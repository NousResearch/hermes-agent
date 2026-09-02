#!/usr/bin/env python3
"""Run one MiniMax Code task through its stable headless JSON contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from shutil import which as find_executable
from typing import Callable, Sequence


Writer = Callable[[str], object]
Runner = Callable[..., subprocess.CompletedProcess[str]]
Which = Callable[[str], str | None]
EXEC_RESULT_STATUSES = {
    "succeeded",
    "failed",
    "blocked",
    "timeout",
    "cancelled",
    "limit_exceeded",
}


def build_command(
    *,
    mcode: str,
    cwd: Path,
    permission: str,
    model: str | None = None,
    session: str | None = None,
    continue_session: bool = False,
    timeout: str | None = None,
    max_steps: int | None = None,
) -> list[str]:
    """Build an argv-only invocation; task text is intentionally excluded."""
    command = [
        mcode,
        "exec",
        "--input",
        "-",
        "--input-format",
        "json",
        "--output-format",
        "json",
        "--cwd",
        str(cwd),
        "--permission",
        permission,
    ]
    if model:
        command.extend(["--model", model])
    if session:
        command.extend(["--session", session])
    if continue_session:
        command.append("--continue")
    if timeout:
        command.extend(["--timeout", timeout])
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    return command


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _is_exec_result(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("schemaVersion") == 1
        and value.get("type") == "exec.result"
        and isinstance(value.get("sessionId"), str)
        and isinstance(value.get("turnId"), str)
        and isinstance(value.get("durationMs"), (int, float))
        and not isinstance(value.get("durationMs"), bool)
        and value.get("status") in EXEC_RESULT_STATUSES
    )


def run_mcode(
    *,
    prompt: str,
    cwd: Path,
    mcode: str = "mcode",
    permission: str = "smart",
    model: str | None = None,
    session: str | None = None,
    continue_session: bool = False,
    timeout: str | None = None,
    max_steps: int | None = None,
    runner: Runner = subprocess.run,
    which: Which = find_executable,
    stdout: Writer = sys.stdout.write,
    stderr: Writer = sys.stderr.write,
) -> int:
    """Execute MCode and preserve its machine result and exit semantics."""
    resolved_mcode = which(mcode)
    if not resolved_mcode:
        stderr("mcode was not found on PATH. Install @minimax-ai/code first.\n")
        return 127
    command = build_command(
        mcode=resolved_mcode,
        cwd=cwd,
        permission=permission,
        model=model,
        session=session,
        continue_session=continue_session,
        timeout=timeout,
        max_steps=max_steps,
    )
    try:
        completed = runner(
            command,
            input=json.dumps({"prompt": prompt}, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            errors="strict",
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        stderr("mcode was not found on PATH. Install @minimax-ai/code first.\n")
        return 127
    except KeyboardInterrupt:
        stderr("mcode run interrupted.\n")
        return 130

    if completed.stderr:
        stderr(completed.stderr)

    try:
        result = json.loads(completed.stdout, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, TypeError, ValueError):
        result = None

    if not _is_exec_result(result):
        if completed.returncode == 0 or not completed.stderr:
            stderr("mcode returned invalid ExecResultV1 JSON.\n")
        return completed.returncode or 1

    stdout(f"{json.dumps(result, ensure_ascii=False, allow_nan=False)}\n")
    if completed.returncode:
        return completed.returncode
    return 0 if result["status"] == "succeeded" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MiniMax Code headlessly and emit one ExecResultV1 JSON object."
    )
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("prompt", nargs="?", help="Task prompt")
    prompt_source.add_argument(
        "--prompt-file", type=Path, help="UTF-8 file containing the task"
    )
    parser.add_argument(
        "--cwd", type=Path, default=Path.cwd(), help="Workspace directory"
    )
    parser.add_argument("--mcode", default="mcode", help="MCode executable path")
    parser.add_argument("--model", help="Temporary provider/model override")
    session = parser.add_mutually_exclusive_group()
    session.add_argument("--session", help="Existing active MCode session ID")
    session.add_argument("--continue", dest="continue_session", action="store_true")
    parser.add_argument(
        "--permission",
        choices=("ask", "smart", "full", "off"),
        default="smart",
        help="MCode permission policy (default: smart)",
    )
    parser.add_argument("--timeout", help="MCode run timeout, such as 30s or 10m")
    parser.add_argument("--max-steps", type=int, help="Maximum assistant steps")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cwd = args.cwd.expanduser().resolve(strict=True)
        if not cwd.is_dir():
            raise ValueError(f"--cwd is not a directory: {cwd}")
        prompt = (
            args.prompt_file.expanduser().read_text(encoding="utf-8")
            if args.prompt_file
            else args.prompt
        )
        if not prompt or not prompt.strip():
            raise ValueError("task prompt cannot be empty")
        if args.max_steps is not None and args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"mcode wrapper error: {exc}\n")
        return 2

    return run_mcode(
        prompt=prompt,
        cwd=cwd,
        mcode=args.mcode,
        permission=args.permission,
        model=args.model,
        session=args.session,
        continue_session=args.continue_session,
        timeout=args.timeout,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
