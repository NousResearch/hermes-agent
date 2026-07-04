"""CLI for the codex-ops plugin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="codex_ops_command")

    subs.add_parser("status", help="Show ledger path, counts, and active settings")

    list_p = subs.add_parser("list", help="List recent recorded Codex terminal runs")
    list_p.add_argument("--limit", "-n", type=int, default=20, help="Number of runs to show (default: 20)")

    show_p = subs.add_parser("show", help="Show one recorded run")
    show_p.add_argument("id", type=int, help="Run id from `hermes codex-ops list`")

    prune_p = subs.add_parser("prune", help="Delete ledger rows older than N days")
    prune_p.add_argument("--days", type=int, default=30, help="Keep rows newer than this many days (default: 30)")

    sanitize_p = subs.add_parser("sanitize", help="Redact high-confidence secrets from text")
    sanitize_p.add_argument("path", nargs="?", default="-", help="File path to sanitize, or '-' for stdin")

    subparser.set_defaults(func=codex_ops_command)


def _print_kv(data: dict[str, Any]) -> None:
    for key, value in data.items():
        print(f"{key}: {value}")


def _cmd_status() -> int:
    info = core.status()
    print("codex-ops status")
    print("================")
    _print_kv(info)
    return 0


def _cmd_list(limit: int) -> int:
    rows = core.list_runs(limit=limit)
    if not rows:
        print("No codex-ops runs recorded yet.")
        return 0
    print(f"{'id':>5}  {'exit':>4}  {'ms':>7}  {'lines':>7}  {'created_at':<25}  command")
    print("-" * 110)
    for row in rows:
        command = (row.get("command") or "").replace("\n", " ")
        if len(command) > 90:
            command = command[:87] + "..."
        print(
            f"{row.get('id', ''):>5}  {str(row.get('exit_code', '')):>4}  "
            f"{row.get('duration_ms', 0):>7}  {row.get('output_lines', 0):>7}  "
            f"{row.get('created_at', ''):<25}  {command}"
        )
    return 0


def _cmd_show(run_id: int) -> int:
    row = core.get_run(run_id)
    if not row:
        print(f"No codex-ops run found with id={run_id}")
        return 1
    print(f"codex-ops run {run_id}")
    print("================")
    for key in (
        "created_at",
        "profile_home",
        "status",
        "exit_code",
        "duration_ms",
        "output_chars",
        "output_lines",
        "output_sha256",
        "workdir",
        "session_id",
        "task_id",
        "tool_call_id",
        "turn_id",
        "error_type",
        "error_message",
    ):
        print(f"{key}: {row.get(key, '')}")
    print("command:")
    print(row.get("command") or "")
    if row.get("summary"):
        print("summary:")
        print(row["summary"])
    return 0


def _cmd_prune(days: int) -> int:
    deleted = core.prune(days)
    print(f"Deleted {deleted} codex-ops ledger row(s) older than {days} day(s).")
    return 0


def _cmd_sanitize(path: str) -> int:
    if path == "-":
        text = sys.stdin.read()
    else:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    print(core.redact_text(text), end="")
    return 0


def codex_ops_command(args: argparse.Namespace) -> int:
    sub = getattr(args, "codex_ops_command", None) or "status"
    if sub == "status":
        return _cmd_status()
    if sub == "list":
        return _cmd_list(limit=int(getattr(args, "limit", 20)))
    if sub == "show":
        return _cmd_show(run_id=int(getattr(args, "id")))
    if sub == "prune":
        return _cmd_prune(days=int(getattr(args, "days", 30)))
    if sub == "sanitize":
        return _cmd_sanitize(path=str(getattr(args, "path", "-")))
    print("usage: hermes codex-ops {status,list,show,prune,sanitize}")
    return 2
