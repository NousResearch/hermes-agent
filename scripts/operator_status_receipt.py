#!/usr/bin/env python3
"""Emit a small operator receipt for the live Hermes gateway.

This script is intentionally read-only. It answers the question that tends to
matter during Hermes maintenance: which checkout is the gateway actually
running, on which branch, with which profile logs?
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ISSUE_PATTERNS = (
    "NoneType",
    "Traceback",
    "Non-retryable client error",
    "responses.stream",
    "finalizer",
    "output=null",
    "response.output=None",
)


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def run_command(args: list[str], *, cwd: Path | None = None) -> CommandResult:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(127, "", str(exc))
    return CommandResult(proc.returncode, proc.stdout, proc.stderr)


def parse_launchctl(text: str) -> dict[str, str]:
    fields = {}
    wanted = {
        "pid",
        "program",
        "working directory",
        "stdout path",
        "stderr path",
        "state",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        key = key.strip()
        if key in wanted:
            fields[key.replace(" ", "_")] = value.strip()
    return fields


def git_info(repo_root: Path) -> dict[str, str]:
    def _git(*args: str) -> str:
        result = run_command(["git", *args], cwd=repo_root)
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    return {
        "root": str(repo_root),
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "--short", "HEAD"),
        "status_short": _git("status", "--short", "--branch"),
    }


def timestamped_issue_lines(path: Path, *, since: str | None = None) -> list[str]:
    if not path.exists():
        return []
    hits = []
    for line in path.read_text(errors="replace").splitlines():
        if since:
            line_stamp = line[:19]
            if len(line_stamp) < 19 or line_stamp[4:5] != "-" or line_stamp[13:14] != ":":
                continue
            if line_stamp < since[:19]:
                continue
        if any(pattern in line for pattern in ISSUE_PATTERNS):
            hits.append(line)
    return hits


def compact_status_flags(status_text: str) -> dict[str, bool]:
    return {
        "slack_configured": "Slack" in status_text and "✓ configured" in status_text,
        "gateway_running": "Gateway Service" in status_text and "✓ running" in status_text,
        "openai_codex_logged_in": "OpenAI Codex" in status_text and "✓ logged in" in status_text,
    }


def build_receipt(profile: str, repo_root: Path, since: str | None = None) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    profile_root = Path.home() / ".hermes" / "profiles" / profile

    status_result = run_command(["hermes", "--profile", profile, "status"], cwd=repo_root)
    launchctl_fields: dict[str, str] = {}
    launchctl_result = CommandResult(0, "", "")
    if platform.system() == "Darwin":
        label = f"gui/{os.getuid()}/ai.hermes.gateway-{profile}"
        launchctl_result = run_command(["launchctl", "print", label], cwd=repo_root)
        if launchctl_result.returncode == 0:
            launchctl_fields = parse_launchctl(launchctl_result.stdout)

    stdout_log = Path(
        launchctl_fields.get("stdout_path")
        or profile_root / "logs" / "gateway.log"
    )
    stderr_log = Path(
        launchctl_fields.get("stderr_path")
        or profile_root / "logs" / "gateway.error.log"
    )
    stdout_hits = timestamped_issue_lines(stdout_log, since=since)
    stderr_hits = timestamped_issue_lines(stderr_log, since=since)

    return {
        "generated_at": now,
        "profile": profile,
        "repo": git_info(repo_root),
        "profile_root": str(profile_root),
        "launchctl": {
            "available": launchctl_result.returncode == 0,
            **launchctl_fields,
        },
        "status_flags": compact_status_flags(status_result.stdout),
        "status_returncode": status_result.returncode,
        "logs": {
            "stdout_path": str(stdout_log),
            "stderr_path": str(stderr_log),
            "since": since,
            "issue_patterns": list(ISSUE_PATTERNS),
            "stdout_issue_count": len(stdout_hits),
            "stderr_issue_count": len(stderr_hits),
            "stdout_recent_issues": stdout_hits[-10:],
            "stderr_recent_issues": stderr_hits[-10:],
        },
    }


def render_markdown(receipt: dict) -> str:
    repo = receipt["repo"]
    launch = receipt["launchctl"]
    logs = receipt["logs"]
    flags = receipt["status_flags"]
    lines = [
        "# Hermes Operator Status Receipt",
        "",
        f"- Generated: `{receipt['generated_at']}`",
        f"- Profile: `{receipt['profile']}`",
        f"- Repo root: `{repo['root']}`",
        f"- Branch: `{repo['branch']}`",
        f"- HEAD: `{repo['head']}`",
        f"- Gateway PID: `{launch.get('pid', 'unknown')}`",
        f"- Gateway cwd: `{launch.get('working_directory', 'unknown')}`",
        f"- Slack configured: `{flags['slack_configured']}`",
        f"- Gateway running: `{flags['gateway_running']}`",
        f"- OpenAI Codex logged in: `{flags['openai_codex_logged_in']}`",
        f"- Log window start: `{logs.get('since') or 'full file scan'}`",
        f"- Stdout issue hits: `{logs['stdout_issue_count']}`",
        f"- Stderr issue hits: `{logs['stderr_issue_count']}`",
        "",
        "## Git Status",
        "",
        "```text",
        repo["status_short"],
        "```",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(receipt: dict, output: Path | None, json_output: Path | None) -> None:
    markdown = render_markdown(receipt)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.getenv("HERMES_PROFILE", "sawyer"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--since", help="Only count timestamped log issues after YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--output", type=Path, help="Markdown receipt output path")
    parser.add_argument("--json-output", type=Path, help="JSON receipt output path")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    receipt = build_receipt(args.profile, args.repo_root.resolve(), since=args.since)
    write_outputs(receipt, args.output, args.json_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
