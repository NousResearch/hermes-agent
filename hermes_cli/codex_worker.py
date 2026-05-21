"""Codex CLI Kanban worker lane adapter.

The dispatcher spawns this module as a wrapper process.  The wrapper owns the
external Codex CLI subprocess, streams stdout/stderr to the Kanban worker log,
periodically heartbeats the claim, writes structured progress events, and
blocks the task for Hermes review when Codex exits successfully.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli.worker_lanes import WorkerLane

CODEX_OUTPUT_TAIL_BYTES = 8192
CODEX_FIELD_MAX_BYTES = 4096
CODEX_PROGRESS_MAX_ITEMS = 50
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0

_CHECKBOX_RE = re.compile(r"^\s*[-*]\s*\[([ xX])\]\s+(.+?)\s*$")
_ORDINAL_RE = re.compile(r"^\s*([oxOX])\s*\((\d+)\)\s+(.+?)\s*$")


@dataclass(frozen=True)
class CodexLaneConfig:
    name: str
    model: Optional[str] = None
    sandbox: str = "workspace-write"
    approval: str = "never"
    max_concurrency: Optional[int] = 1
    success_policy: str = "block_for_review"
    timeout_seconds: Optional[int] = None
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS


class _TailBuffer:
    def __init__(self, max_bytes: int = CODEX_OUTPUT_TAIL_BYTES) -> None:
        self.max_bytes = int(max_bytes)
        self._buf = b""

    def append(self, text: str) -> None:
        if not text:
            return
        self._buf += text.encode("utf-8", errors="replace")
        if len(self._buf) > self.max_bytes:
            self._buf = self._buf[-self.max_bytes :]

    def text(self) -> str:
        return self._buf.decode("utf-8", errors="replace")


def make_codex_worker_lane(config: dict[str, Any], *, source: str = "config") -> WorkerLane:
    cfg = CodexLaneConfig(
        name=str(config["name"]),
        model=(str(config["model"]) if config.get("model") else None),
        sandbox=str(config.get("sandbox") or "workspace-write"),
        approval=str(config.get("approval") or "never"),
        max_concurrency=(
            int(config["max_concurrency"])
            if config.get("max_concurrency") is not None
            else None
        ),
        success_policy=str(config.get("success_policy") or "block_for_review"),
        timeout_seconds=(
            int(config["timeout_seconds"])
            if config.get("timeout_seconds") is not None
            else None
        ),
    )

    def _spawn(task, workspace: str, *, board: Optional[str] = None) -> Optional[int]:
        return spawn_codex_worker(task, workspace, cfg, board=board)

    return WorkerLane(
        name=cfg.name,
        kind="codex_cli",
        description=f"Codex CLI worker lane ({cfg.model or 'default model'})",
        spawn_fn=_spawn,
        success_policy=cfg.success_policy,
        max_concurrency=cfg.max_concurrency,
        source=source,
        config={
            "type": "codex_cli",
            "model": cfg.model,
            "sandbox": cfg.sandbox,
            "approval": cfg.approval,
            "timeout_seconds": cfg.timeout_seconds,
        },
    )


def _safe_env_for_worker(task, workspace: str, cfg: CodexLaneConfig, *, board: Optional[str]) -> dict[str, str]:
    from hermes_cli import kanban_db as kb

    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "HERMES_HOME",
        "HERMES_KANBAN_HOME",
        "CODEX_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
    }
    env = {k: v for k, v in os.environ.items() if k in allowed and v is not None}
    project_root = str(Path(__file__).resolve().parent.parent)
    existing_pythonpath = env.get("PYTHONPATH") or os.environ.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        project_root
        if not existing_pythonpath
        else project_root + os.pathsep + existing_pythonpath
    )
    env["HERMES_KANBAN_TASK"] = task.id
    env["HERMES_KANBAN_WORKSPACE"] = workspace
    env["HERMES_KANBAN_HOME"] = str(kb.kanban_home())
    env["HERMES_KANBAN_DB"] = str(kb.kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(kb.workspaces_root(board=board))
    env["HERMES_KANBAN_BOARD"] = kb._normalize_board_slug(board) or kb.get_current_board()
    env["HERMES_WORKER_LANE"] = cfg.name
    env["HERMES_WORKER_KIND"] = "codex_cli"
    env["HERMES_PROFILE"] = cfg.name
    if task.current_run_id is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
    if task.claim_lock:
        env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock
    if task.tenant:
        env["HERMES_TENANT"] = task.tenant
    return env


def _path_is_writable_dir(path: Optional[str]) -> bool:
    if not path:
        return False
    try:
        p = Path(path).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".hermes-codex-write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _safe_env_for_codex(workspace: Optional[str] = None) -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "CODEX_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_WORKSPACE",
        "HERMES_KANBAN_BOARD",
        "HERMES_WORKER_LANE",
        "HERMES_WORKER_KIND",
    }
    env = {k: v for k, v in os.environ.items() if k in allowed and v is not None}
    home_writable = _path_is_writable_dir(env.get("HOME"))
    home_rebased = False
    if not home_writable and workspace:
        home = Path(workspace) / ".hermes-codex-home"
        home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        home_writable = True
        home_rebased = True
    codex_home = env.get("CODEX_HOME")
    if codex_home:
        codex_home_writable = _path_is_writable_dir(codex_home)
    else:
        default_codex_home = Path(env["HOME"]).expanduser() / ".codex" if env.get("HOME") else None
        codex_home_writable = bool(default_codex_home and _path_is_writable_dir(str(default_codex_home)))
    if not codex_home_writable and (codex_home or home_rebased or not home_writable) and workspace:
        codex_home = Path(workspace) / ".hermes-codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(codex_home)
    return env


def spawn_codex_worker(
    task,
    workspace: str,
    cfg: CodexLaneConfig,
    *,
    board: Optional[str] = None,
) -> Optional[int]:
    """Spawn the Python wrapper that runs Codex CLI."""
    from hermes_cli import kanban_db as kb

    cmd = [
        sys.executable,
        "-m",
        "hermes_cli.codex_worker",
        "run",
        "--task-id",
        task.id,
        "--lane",
        cfg.name,
        "--workspace",
        workspace,
        "--sandbox",
        cfg.sandbox,
        "--approval",
        cfg.approval,
        "--success-policy",
        cfg.success_policy,
        "--heartbeat-interval",
        str(cfg.heartbeat_interval_seconds),
    ]
    if task.current_run_id is not None:
        cmd.extend(["--run-id", str(task.current_run_id)])
    if task.claim_lock:
        cmd.extend(["--claim-lock", task.claim_lock])
    if cfg.model:
        cmd.extend(["--model", cfg.model])
    if cfg.timeout_seconds is not None:
        cmd.extend(["--timeout-seconds", str(cfg.timeout_seconds)])
    resolved_board = kb._normalize_board_slug(board) or kb.get_current_board()
    cmd.extend(["--board", resolved_board])

    log_dir = kb.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    kb._rotate_worker_log(log_path, kb.DEFAULT_LOG_ROTATE_BYTES)
    env = _safe_env_for_worker(task, workspace, cfg, board=board)

    log_f = open(log_path, "ab")
    try:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell.
            cmd,
            cwd=workspace if os.path.isdir(workspace) else None,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        log_f.close()
    return proc.pid


def build_codex_argv(
    *,
    binary: str,
    workspace: str,
    sandbox: str,
    approval: str,
    model: Optional[str] = None,
) -> list[str]:
    argv = [
        binary,
        "--cd",
        workspace,
        "--sandbox",
        sandbox,
        "--ask-for-approval",
        approval,
    ]
    if model:
        argv.extend(["--model", model])
    argv.extend(["exec", "-"])
    return argv


def parse_progress_items(text: str) -> list[dict[str, Any]]:
    """Parse supported Codex progress/checklist formats."""
    items: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    implicit_index = 1

    def add_item(item: dict[str, Any]) -> None:
        key = str(item["text"]).strip().casefold()
        if key in positions:
            existing = items[positions[key]]
            existing["status"] = item["status"]
            existing["text"] = item["text"]
            return
        positions[key] = len(items)
        items.append(item)

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _ORDINAL_RE.match(line)
        if m:
            mark, idx, item_text = m.groups()
            item_text = item_text.strip()
            if _is_placeholder_progress_text(item_text):
                continue
            # Codex plan output historically used `o` for completed and `x`
            # for the item currently being worked in examples.
            status = "done" if mark.lower() == "o" else "running"
            add_item({
                "index": int(idx),
                "status": status,
                "text": item_text[:500],
            })
            continue
        m = _CHECKBOX_RE.match(line)
        if m:
            mark, item_text = m.groups()
            item_text = item_text.strip()
            if _is_placeholder_progress_text(item_text):
                continue
            status = "done" if mark.lower() == "x" else "pending"
            add_item({
                "index": implicit_index,
                "status": status,
                "text": item_text[:500],
            })
            implicit_index += 1
    return items[:CODEX_PROGRESS_MAX_ITEMS]


def _is_placeholder_progress_text(text: str) -> bool:
    """Return True for template-only checklist entries echoed by Codex CLI."""
    return text.strip() in {"...", "…"}


def _write_log(log_f, text: str) -> None:
    try:
        log_f.write(text)
        log_f.flush()
    except Exception:
        pass


def _record_event(task_id: str, kind: str, payload: dict[str, Any], *, run_id: Optional[int]) -> None:
    from hermes_cli import kanban_db as kb

    try:
        with kb.connect() as conn:
            kb.record_task_event(conn, task_id, kind, payload, run_id=run_id)
    except Exception:
        pass


def _heartbeat(task_id: str, *, run_id: Optional[int], claim_lock: Optional[str], lane: str) -> None:
    from hermes_cli import kanban_db as kb

    try:
        with kb.connect() as conn:
            if claim_lock:
                kb.heartbeat_claim(conn, task_id, claimer=claim_lock)
            kb.heartbeat_worker(
                conn,
                task_id,
                note=f"worker_lane={lane}",
                expected_run_id=run_id,
            )
    except Exception:
        pass


def _cap(text: Optional[str], limit: int = CODEX_FIELD_MAX_BYTES) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated {len(text) - limit} chars]"


def _run_git(args: list[str], workspace: str, *, timeout: float = 5.0) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", workspace, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        return out if out else err
    except Exception as exc:
        return str(exc)


def collect_git_evidence(workspace: str) -> dict[str, Any]:
    if not shutil.which("git"):
        return {"status": "", "changed_files": [], "diff_summary": "git not found"}
    status = _run_git(["status", "--short"], workspace)
    changed_files: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip() if len(line) >= 3 else line.strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed_files.append(path)
    diff_stat = _run_git(["diff", "--stat", "--summary", "HEAD"], workspace)
    if "not a git repository" in status.lower() or "not a git repository" in diff_stat.lower():
        return {
            "status": _cap(status),
            "changed_files": [],
            "diff_summary": "not a git repository",
        }
    untracked = [
        path
        for line, path in zip(status.splitlines(), changed_files)
        if line.startswith("?? ")
    ]
    if untracked:
        untracked_summary = "\n".join(f"?? {path}" for path in untracked[:200])
        diff_stat = (
            diff_stat + "\n" + untracked_summary
            if diff_stat else untracked_summary
        )
    return {
        "status": _cap(status),
        "changed_files": changed_files[:200],
        "diff_summary": _cap(diff_stat),
    }


def _extract_verification_summary(output: str) -> dict[str, Any]:
    blocks: list[list[str]] = []
    current: list[str] = []
    capture = False
    for line in output.splitlines():
        lower = line.strip().lower()
        if lower.startswith("verification:"):
            if capture and current:
                blocks.append(current)
            current = []
            capture = True
            continue
        if capture and lower.endswith(":") and not lower.startswith(("command:", "result:")):
            blocks.append(current)
            current = []
            capture = False
            continue
        if capture:
            current.append(line)
    if capture and current:
        blocks.append(current)

    candidates: list[dict[str, Any]] = []
    for block in blocks:
        lines: list[str] = []
        commands: list[str] = []
        for line in block:
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            is_command = lowered.startswith("- command:") or lowered.startswith("command:")
            is_result = lowered.startswith("- result:") or lowered.startswith("result:")
            value = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            value = value.strip("`").strip()
            if (is_command or is_result) and _is_placeholder_progress_text(value):
                continue
            lines.append(line)
            if is_command and value:
                commands.append(value)
        summary = "\n".join(lines).strip()
        if summary or commands:
            candidates.append({
                "commands": commands[:20],
                "summary": _cap(summary),
            })

    if not candidates:
        return {"commands": [], "summary": ""}
    return candidates[-1]


def _metadata(
    *,
    lane: str,
    task_id: str,
    run_id: Optional[int],
    worker_pid: int,
    claim_lock: Optional[str],
    workspace: str,
    model: Optional[str],
    exit_code: Optional[int],
    timed_out: bool,
    output_tail: str,
    binary_missing: bool = False,
) -> dict[str, Any]:
    succeeded = (exit_code == 0 and not timed_out and not binary_missing)
    return {
        "worker_instance": {
            "worker_lane": lane,
            "worker_kind": "codex_cli",
            "task_id": task_id,
            "run_id": run_id,
            "worker_pid": worker_pid,
            "claim_lock": claim_lock,
            "workspace": workspace,
            "model": model,
        },
        "worker_lane": {
            "name": lane,
            "kind": "codex_cli",
            "task_id": task_id,
            "run_id": run_id,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "binary_missing": binary_missing,
            "output_tail": output_tail,
        },
        "git": collect_git_evidence(workspace),
        "verification": _extract_verification_summary(output_tail),
        "review": {
            "required": succeeded,
            "reason": (
                "Codex completed; Hermes review required"
                if succeeded
                else "Codex did not complete successfully"
            ),
        },
    }


def _finish_blocked(
    *,
    task_id: str,
    run_id: Optional[int],
    reason: str,
    metadata: dict[str, Any],
) -> bool:
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        return kb.block_task(
            conn,
            task_id,
            reason=reason,
            expected_run_id=run_id,
            metadata=metadata,
        )


def build_codex_prompt(task_context: str, *, lane: str, model: Optional[str]) -> str:
    return f"""{task_context.rstrip()}

## External worker instructions

You are Codex CLI running as Hermes Kanban worker lane `{lane}`.
Implement the assigned task in the workspace. Do not mark the Kanban task
done yourself; this wrapper will return your structured receipt to Hermes
and block the task for review.

When finished, print a concise structured receipt:

Progress:
- [x] ...
- [ ] ...

Changed files:
- ...

Verification:
- command: ...
  result: ...

Remaining risks:
- ...

Recommended reviewer action:
- ...
"""


def run_codex_worker(
    *,
    task_id: str,
    lane: str,
    workspace: str,
    sandbox: str,
    approval: str,
    model: Optional[str] = None,
    run_id: Optional[int] = None,
    claim_lock: Optional[str] = None,
    board: Optional[str] = None,
    success_policy: str = "block_for_review",
    timeout_seconds: Optional[float] = None,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> int:
    from hermes_cli import kanban_db as kb

    log_path = kb.worker_log_path(task_id, board=board)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    worker_pid = os.getpid()
    tail = _TailBuffer()
    last_progress_json = ""

    with open(log_path, "a", encoding="utf-8", errors="replace") as log_f:
        header = {
            "worker_lane": lane,
            "worker_kind": "codex_cli",
            "task_id": task_id,
            "run_id": run_id,
            "worker_pid": worker_pid,
            "claim_lock": claim_lock,
            "workspace": workspace,
            "model": model,
        }
        _write_log(log_f, "[codex-worker] " + json.dumps(header, ensure_ascii=False) + "\n")
        _record_event(task_id, "worker_started", header, run_id=run_id)
        _heartbeat(task_id, run_id=run_id, claim_lock=claim_lock, lane=lane)

        codex_bin = shutil.which("codex")
        if not codex_bin:
            msg = "codex binary not found on PATH"
            _write_log(log_f, f"[codex-worker] {msg}\n")
            meta = _metadata(
                lane=lane,
                task_id=task_id,
                run_id=run_id,
                worker_pid=worker_pid,
                claim_lock=claim_lock,
                workspace=workspace,
                model=model,
                exit_code=None,
                timed_out=False,
                output_tail=msg,
                binary_missing=True,
            )
            _record_event(task_id, "worker_failed", meta["worker_lane"], run_id=run_id)
            _finish_blocked(
                task_id=task_id,
                run_id=run_id,
                reason="codex-failed: codex binary not found on PATH",
                metadata=meta,
            )
            return 0

        with kb.connect() as conn:
            task_context = kb.build_worker_context(conn, task_id)
        prompt = build_codex_prompt(task_context, lane=lane, model=model)
        argv = build_codex_argv(
            binary=codex_bin,
            workspace=workspace,
            sandbox=sandbox,
            approval=approval,
            model=model,
        )
        _write_log(log_f, "[codex-worker] exec " + json.dumps(argv, ensure_ascii=False) + "\n")

        try:
            proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell.
                argv,
                cwd=workspace if os.path.isdir(workspace) else None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=_safe_env_for_codex(workspace),
            )
        except OSError as exc:
            msg = f"failed to start codex: {exc}"
            _write_log(log_f, f"[codex-worker] {msg}\n")
            meta = _metadata(
                lane=lane,
                task_id=task_id,
                run_id=run_id,
                worker_pid=worker_pid,
                claim_lock=claim_lock,
                workspace=workspace,
                model=model,
                exit_code=None,
                timed_out=False,
                output_tail=msg,
            )
            _record_event(task_id, "worker_failed", meta["worker_lane"], run_id=run_id)
            _finish_blocked(
                task_id=task_id,
                run_id=run_id,
                reason=f"codex-failed: {exc}",
                metadata=meta,
            )
            return 0
        _record_event(
            task_id,
            "worker_spawned_external",
            {
                "worker_lane": lane,
                "worker_kind": "codex_cli",
                "run_id": run_id,
                "pid": proc.pid,
                "model": model,
            },
            run_id=run_id,
        )
        try:
            assert proc.stdin is not None
            proc.stdin.write(prompt)
            proc.stdin.close()
        except BrokenPipeError:
            pass

        q: "queue.Queue[Optional[str]]" = queue.Queue()

        def _reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    q.put(line)
            finally:
                q.put(None)

        reader = threading.Thread(target=_reader, name="codex-worker-reader", daemon=True)
        reader.start()
        next_heartbeat = time.monotonic() + max(1.0, float(heartbeat_interval))
        timed_out = False
        reader_done = False
        while True:
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                item = ""
            if item is None:
                reader_done = True
            elif item:
                _write_log(log_f, item)
                tail.append(item)
                items = parse_progress_items(tail.text())
                if items:
                    progress_payload = {
                        "lane": lane,
                        "worker_kind": "codex_cli",
                        "items": items,
                    }
                    progress_json = json.dumps(progress_payload, ensure_ascii=False, sort_keys=True)
                    if progress_json != last_progress_json:
                        _record_event(
                            task_id,
                            "worker_progress",
                            progress_payload,
                            run_id=run_id,
                        )
                        last_progress_json = progress_json

            now = time.monotonic()
            if now >= next_heartbeat:
                _heartbeat(task_id, run_id=run_id, claim_lock=claim_lock, lane=lane)
                next_heartbeat = now + max(1.0, float(heartbeat_interval))

            if timeout_seconds is not None and now - started > float(timeout_seconds):
                timed_out = True
                _write_log(log_f, "[codex-worker] timeout exceeded; terminating codex\n")
                try:
                    proc.terminate()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except OSError:
                        pass
                break

            if proc.poll() is not None and reader_done:
                break

        if not timed_out:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
        reader.join(timeout=1)
        exit_code = proc.returncode
        output_tail = tail.text()
        meta = _metadata(
            lane=lane,
            task_id=task_id,
            run_id=run_id,
            worker_pid=worker_pid,
            claim_lock=claim_lock,
            workspace=workspace,
            model=model,
            exit_code=exit_code,
            timed_out=timed_out,
            output_tail=output_tail,
        )
        if timed_out:
            _record_event(task_id, "worker_timed_out", meta["worker_lane"], run_id=run_id)
            _finish_blocked(
                task_id=task_id,
                run_id=run_id,
                reason=f"codex-timeout: exceeded {timeout_seconds}s",
                metadata=meta,
            )
            return 0
        if exit_code == 0:
            _record_event(task_id, "worker_review_required", meta["worker_lane"], run_id=run_id)
            reason = "review-required: Codex completed; Hermes review required"
            _finish_blocked(
                task_id=task_id,
                run_id=run_id,
                reason=reason,
                metadata=meta,
            )
            return 0

        _record_event(task_id, "worker_failed", meta["worker_lane"], run_id=run_id)
        _finish_blocked(
            task_id=task_id,
            run_id=run_id,
            reason=f"codex-failed: exit code {exit_code}",
            metadata=meta,
        )
    return 0


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m hermes_cli.codex_worker")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--task-id", required=True)
    run.add_argument("--lane", required=True)
    run.add_argument("--workspace", required=True)
    run.add_argument("--sandbox", required=True)
    run.add_argument("--approval", required=True)
    run.add_argument("--success-policy", default="block_for_review")
    run.add_argument("--model")
    run.add_argument("--run-id", type=int)
    run.add_argument("--claim-lock")
    run.add_argument("--board")
    run.add_argument("--timeout-seconds", type=float)
    run.add_argument("--heartbeat-interval", type=float, default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.cmd == "run":
        return run_codex_worker(
            task_id=args.task_id,
            lane=args.lane,
            workspace=args.workspace,
            sandbox=args.sandbox,
            approval=args.approval,
            model=args.model,
            run_id=args.run_id,
            claim_lock=args.claim_lock,
            board=args.board,
            success_policy=args.success_policy,
            timeout_seconds=args.timeout_seconds,
            heartbeat_interval=args.heartbeat_interval,
        )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
