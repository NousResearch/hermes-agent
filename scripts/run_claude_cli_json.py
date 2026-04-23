#!/usr/bin/env python3
"""Run Claude Code CLI in print mode and emit normalized JSON."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MAX_TURNS = 10
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MODEL = "opus"


def _normalize_allowed_tools(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def build_command(
    *,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    allowed_tools: str | None = None,
    append_system_prompt: str | None = None,
) -> list[str]:
    command = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--max-turns",
        str(max_turns),
    ]
    if model:
        command.extend(["--model", model])
    allowed_tools = _normalize_allowed_tools(allowed_tools)
    if allowed_tools:
        command.extend(["--allowedTools", allowed_tools])
    if append_system_prompt:
        command.extend(["--append-system-prompt", append_system_prompt])
    return command


def run_claude_cli(
    *,
    prompt: str,
    workdir: str | None = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    allowed_tools: str | None = None,
    append_system_prompt: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    claude_path = shutil.which("claude")
    if not claude_path:
        return {
            "success": False,
            "error": "claude CLI not found on PATH",
            "command": None,
            "cwd": workdir or os.getcwd(),
        }

    if not prompt or not prompt.strip():
        return {
            "success": False,
            "error": "prompt is required",
            "command": None,
            "cwd": workdir or os.getcwd(),
        }

    cwd = str(Path(workdir).expanduser()) if workdir else os.getcwd()
    command = build_command(
        prompt=prompt,
        model=model,
        max_turns=max_turns,
        allowed_tools=allowed_tools,
        append_system_prompt=append_system_prompt,
    )

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "error": f"claude CLI timed out after {timeout_seconds}s",
            "command": command,
            "cwd": cwd,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    parsed: dict[str, Any] | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None

    success = completed.returncode == 0 and isinstance(parsed, dict) and not parsed.get("is_error", False)
    result: dict[str, Any] = {
        "success": success,
        "command": command,
        "cwd": cwd,
        "claude_path": claude_path,
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    if parsed is not None:
        result["claude_json"] = parsed
        result["result"] = parsed.get("result")
        result["session_id"] = parsed.get("session_id")
        result["subtype"] = parsed.get("subtype")
        result["total_cost_usd"] = parsed.get("total_cost_usd")
        result["usage"] = parsed.get("usage")
    elif stdout:
        result["error"] = "claude CLI output was not valid JSON"
    elif not stderr:
        result["error"] = "claude CLI produced no output"

    if not success and "error" not in result:
        result["error"] = stderr or result.get("result") or "claude CLI invocation failed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Claude Code CLI in print mode and emit normalized JSON")
    parser.add_argument("prompt", help="Prompt to send to Claude Code CLI")
    parser.add_argument("--workdir", default=None, help="Working directory for claude CLI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model alias/name (default: opus)")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Max turns for claude -p")
    parser.add_argument("--allowed-tools", default=None, help="Pass-through value for --allowedTools")
    parser.add_argument("--append-system-prompt", default=None, help="Additional system prompt text")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Subprocess timeout seconds")
    args = parser.parse_args()

    result = run_claude_cli(
        prompt=args.prompt,
        workdir=args.workdir,
        model=args.model,
        max_turns=args.max_turns,
        allowed_tools=args.allowed_tools,
        append_system_prompt=args.append_system_prompt,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
