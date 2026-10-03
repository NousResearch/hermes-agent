#!/usr/bin/env python3
"""Safely record observed Antigravity CLI stream-json behavior.

The probe runs only in a new temporary workspace.  It does not request
permission bypasses and never asks the agent to modify files.  Its JSON report
is evidence, not a version-independent protocol specification.

Usage:
  python scripts/antigravity_probe.py
  python scripts/antigravity_probe.py --agy /path/to/agy --output report.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_AGY = Path.home() / ".local/bin/agy"
SAFE_PROMPT = "Reply with exactly PONG. Do not use tools."


@dataclass(frozen=True)
class ParsedStream:
    events: list[dict[str, Any]]
    diagnostics: list[str]


def parse_ndjson(output: str) -> ParsedStream:
    """Split stdout into JSON event records and non-JSON diagnostics."""
    events: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            diagnostics.append(line)
            continue
        if isinstance(value, dict):
            events.append(value)
        else:
            diagnostics.append(line)
    return ParsedStream(events=events, diagnostics=diagnostics)


def _ordered(values: list[str]) -> list[str]:
    return sorted(set(values))


def summarize_stream(parsed: ParsedStream) -> dict[str, Any]:
    """Extract only fields actually present in a stream run."""
    conversation_ids: list[str] = []
    event_types: list[str] = []
    step_types: list[str] = []
    tool_names: list[str] = []
    permission_modes: list[str] = []
    denied_actions: list[str] = []
    statuses: list[str] = []
    tools: list[str] = []

    for event in parsed.events:
        name = event.get("event")
        if isinstance(name, str):
            event_types.append(name)
        conversation_id = event.get("conversation_id")
        if isinstance(conversation_id, str):
            conversation_ids.append(conversation_id)
        init = event.get("init")
        if isinstance(init, dict):
            mode = init.get("permission_mode")
            if isinstance(mode, str):
                permission_modes.append(mode)
            tools.extend(item for item in init.get("tools", []) if isinstance(item, str))
        step = event.get("step_update")
        if isinstance(step, dict):
            conversation_id = step.get("conversation_id")
            if isinstance(conversation_id, str):
                conversation_ids.append(conversation_id)
            step_type = step.get("step_type")
            if isinstance(step_type, str):
                step_types.append(step_type)
            tool_name = step.get("tool_name")
            if isinstance(tool_name, str):
                tool_names.append(tool_name)
        result = event.get("result")
        if isinstance(result, dict):
            conversation_id = result.get("conversation_id")
            if isinstance(conversation_id, str):
                conversation_ids.append(conversation_id)
            status = result.get("status")
            if isinstance(status, str):
                statuses.append(status)
            for action in result.get("denied_actions", []):
                if isinstance(action, dict) and isinstance(action.get("action"), str):
                    denied_actions.append(action["action"])

    return {
        "conversation_ids": _ordered(conversation_ids),
        "denied_actions": _ordered(denied_actions),
        "diagnostics": parsed.diagnostics,
        "event_types": _ordered(event_types),
        "event_count": len(parsed.events),
        "permission_modes": _ordered(permission_modes),
        "statuses": _ordered(statuses),
        "step_types": _ordered(step_types),
        "tool_names": _ordered(tool_names),
        "tools": _ordered(tools),
    }


def _run(command: list[str], *, cwd: Path, stdin: str | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=90,
    )
    parsed = parse_ndjson(completed.stdout)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
        "stream": summarize_stream(parsed),
    }


def _stream_command(agy: str, *, conversation_id: str | None = None) -> list[str]:
    command = [
        agy,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--sandbox",
        "--print=",
        "--print-timeout",
        "60s",
    ]
    if conversation_id:
        command.extend(["--conversation", conversation_id])
    return command


def _user_record(prompt: str) -> str:
    return json.dumps({"event": "user", "message": {"role": "user", "content": prompt}}) + "\n"


def run_probe(agy: str) -> dict[str, Any]:
    """Execute bounded, non-destructive protocol experiments in a temp dir."""
    if not shutil.which(agy) and not Path(agy).is_file():
        raise FileNotFoundError(f"Antigravity CLI not found: {agy}")

    with tempfile.TemporaryDirectory(prefix="antigravity-probe-") as temporary:
        cwd = Path(temporary)
        version = _run([agy, "--version"], cwd=cwd)
        help_result = _run([agy, "--help"], cwd=cwd)
        valid = _run(_stream_command(agy), cwd=cwd, stdin=_user_record(SAFE_PROMPT))
        malformed = _run(_stream_command(agy), cwd=cwd, stdin="not json\n")
        print_timeout = _run(
            [*_stream_command(agy)[:-2], "--print-timeout", "1ms"],
            cwd=cwd,
            stdin=_user_record(SAFE_PROMPT),
        )
        tool_permission = _run(
            _stream_command(agy),
            cwd=cwd,
            stdin=_user_record(
                "Invoke run_command exactly once with command pwd. Do not modify files. "
                "Then reply with exactly RAN."
            ),
        )

        conversation_ids = valid["stream"]["conversation_ids"]
        resume: dict[str, Any] = {"skipped": "valid run returned no conversation_id"}
        if conversation_ids:
            resume = _run(
                _stream_command(agy, conversation_id=conversation_ids[0]),
                cwd=cwd,
                stdin=_user_record("Reply with exactly RESUMED. Do not use tools."),
            )

        capabilities = {
            name: _run([agy, *command], cwd=cwd)
            for name, command in {
                "models": ["models"],
                "agents": ["agents"],
                "mcp": ["mcp", "list"],
                "plugins": ["plugin", "list"],
            }.items()
        }

    return {
        "agy_path": agy,
        "experiments": {
            "capabilities": capabilities,
            "help": help_result,
            "malformed_stream_input": malformed,
            "print_timeout": print_timeout,
            "resume": resume,
            "tool_permission": tool_permission,
            "valid_stream_input": valid,
            "version": version,
        },
        "safety": {
            "dangerously_skip_permissions": False,
            "workspace": "fresh temporary directory removed after probe",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agy", default=str(DEFAULT_AGY), help="Path or command name for agy")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    args = parser.parse_args()
    try:
        report = run_probe(args.agy)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 2
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
        print(args.output)
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
