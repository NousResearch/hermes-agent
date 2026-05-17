"""Hermes-owned runner for native Codex CLI Kanban workers."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from hermes_cli import kanban_db as kb

SUCCESS_REASON = "review-required: Codex completed; Hermes review required"
FAILURE_REASON = "codex-failed: Codex failed"
OUTPUT_TAIL_CHARS = 4000
DIFF_SUMMARY_CHARS = 12000
SECRET_NAME_RE = re.compile(r"(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)", re.I)


def _tail(text: str, limit: int = OUTPUT_TAIL_CHARS) -> str:
    return text if len(text) <= limit else text[-limit:]


def _coerce_timeout_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _append_log(task_id: str, board: str | None, text: str) -> None:
    if not text:
        return
    log_path = kb.worker_log_path(task_id, board=board)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", errors="replace") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _codex_env(parent: dict[str, str] | None = None) -> dict[str, str]:
    """Pass a minimal non-Hermes-secret environment to Codex CLI."""
    src = dict(parent or os.environ)
    env: dict[str, str] = {}
    allow = {
        "HOME", "USER", "LOGNAME", "SHELL", "PATH", "LANG", "LC_ALL", "TERM",
        "TMPDIR", "CODEX_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
    }
    for key, value in src.items():
        if key in allow or key.startswith("CODEX_"):
            env[key] = value
        elif key.startswith("HERMES_KANBAN_"):
            env[key] = value
        elif key == "HERMES_PROFILE":
            env[key] = value
        elif SECRET_NAME_RE.search(key):
            continue
    return env


def _build_prompt(context: str, workspace: Path) -> str:
    return f"""You are OpenAI Codex CLI running as a Hermes Kanban implementation worker.

Hermes Kanban is the source of truth for task lifecycle. Your job is to execute this one card in the assigned workspace, then leave a clear receipt. The wrapper will update Kanban after you exit.

Workspace:
{workspace}

Operating contract:
- Treat the Kanban card as your goal. Set your own internal goal/plan from it before editing.
- Investigate first. Read relevant repo guidance and existing code before changing files.
- Work only inside the workspace unless the card explicitly authorizes another path.
- Use bounded judgment: implement the scoped goal, run relevant gates, self-review, then stop.
- Do not merge, deploy, force-push, rotate secrets, or touch production data.
- If blocked, explain the blocker and evidence in your final response.
- Final response must include: changed files, commands/gates run, verification result, remaining risks, and next reviewer action.

Kanban task context:
```text
{context}
```
"""


def _run_git(args: list[str], workspace: Path) -> str:
    result = subprocess.run(args, cwd=str(workspace), text=True, capture_output=True, check=False)
    return (result.stdout or "") + (result.stderr or "")


def _git_summary(workspace: Path) -> dict[str, str]:
    status = _run_git(["git", "status", "--short"], workspace)
    diff_name_only = _run_git(["git", "diff", "--name-only"], workspace)
    staged_name_only = _run_git(["git", "diff", "--cached", "--name-only"], workspace)
    untracked = _run_git(["git", "ls-files", "--others", "--exclude-standard"], workspace)
    changed_files = "".join([diff_name_only, staged_name_only, untracked])
    diff_summary = _run_git(["git", "diff", "--stat", "--patch"], workspace)
    staged_summary = _run_git(["git", "diff", "--cached", "--stat", "--patch"], workspace)
    return {
        "status": status,
        "changed_files": changed_files,
        "diff_name_only": diff_name_only,
        "staged_name_only": staged_name_only,
        "untracked": untracked,
        "diff_summary": _tail(diff_summary + staged_summary, DIFF_SUMMARY_CHARS),
    }


def _codex_command(codex_bin: str, workspace: Path, prompt: str, *, model: str | None = None) -> list[str]:
    cmd = [
        codex_bin,
        "--cd", str(workspace),
        "--sandbox", "workspace-write",
        "--ask-for-approval", "never",
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["exec", "-"])
    return cmd


def _metadata(exit_code: int, output: str, workspace: Path, cmd: list[str]) -> dict[str, Any]:
    return {
        "codex": {
            "exit_code": int(exit_code),
            "output_tail": _tail(output),
            "command": cmd,
        },
        "git": _git_summary(workspace),
    }


def run_task(
    task_id: str,
    workspace: Path,
    *,
    board: str | None = None,
    codex_bin: str = "codex",
    model: str | None = None,
    timeout_seconds: int | None = None,
) -> int:
    workspace = workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace is not a directory: {workspace}")

    resolved_codex = shutil.which(codex_bin) or (codex_bin if Path(codex_bin).exists() else None)
    if resolved_codex is None:
        output = f"Codex executable not found: {codex_bin}"
        _append_log(task_id, board, output)
        with kb.connect(board=board) as conn:
            kb.block_task(conn, task_id, reason=FAILURE_REASON, error=output, metadata={"codex": {"exit_code": 127, "output_tail": output}})
        return 127

    with kb.connect(board=board) as conn:
        context = kb.build_worker_context(conn, task_id)

    prompt = _build_prompt(context, workspace)
    cmd = _codex_command(resolved_codex, workspace, prompt, model=model)
    header = "\n".join([
        "=== codex kanban worker ===",
        f"task={task_id}",
        f"board={board or ''}",
        f"workspace={workspace}",
        "command=" + " ".join(cmd[:-1] + ["<prompt-stdin>"]),
        "",
    ])
    _append_log(task_id, board, header)

    timed_out = False
    output_parts: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(workspace),
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_codex_env(),
        )
        assert proc.stdin is not None
        proc.stdin.write(prompt)
        proc.stdin.close()

        assert proc.stdout is not None
        started = __import__("time").time()
        last_heartbeat = 0.0
        for line in proc.stdout:
            output_parts.append(line)
            _append_log(task_id, board, line)
            now = __import__("time").time()
            if now - last_heartbeat >= 60:
                with kb.connect(board=board) as conn:
                    kb.heartbeat_worker(conn, task_id, note="codex worker still running")
                last_heartbeat = now
            if timeout_seconds is not None and now - started > timeout_seconds:
                timed_out = True
                proc.terminate()
                break
        exit_code = proc.wait(timeout=10) if not timed_out else 124
        output = "".join(output_parts)
        if timed_out:
            output += f"\nTimed out after {timeout_seconds} seconds.\n"
            _append_log(task_id, board, f"\nTimed out after {timeout_seconds} seconds.\n")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        output = "".join(output_parts) + _coerce_timeout_output(exc.stdout) + _coerce_timeout_output(exc.stderr) + f"\nTimed out after {exc.timeout} seconds.\n"
    metadata = _metadata(exit_code, output, workspace, cmd[:-1] + ["<prompt-stdin>"])
    if timed_out:
        metadata["codex"]["timed_out"] = True

    with kb.connect(board=board) as conn:
        if exit_code == 0:
            kb.block_task(conn, task_id, reason=SUCCESS_REASON, metadata=metadata)
        else:
            kb.block_task(conn, task_id, reason=FAILURE_REASON, error=_tail(output), metadata=metadata)
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native Codex CLI for one Hermes Kanban task")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--board", default=None)
    parser.add_argument("--codex-bin", default=os.environ.get("HERMES_CODEX_BIN", "codex"))
    parser.add_argument("--model", default=os.environ.get("HERMES_CODEX_MODEL"))
    parser.add_argument("--timeout-seconds", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_task(
        args.task_id,
        Path(args.workspace),
        board=args.board,
        codex_bin=args.codex_bin,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
