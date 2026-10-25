"""Shared, local-only result management for CLI and private slash surfaces."""

from __future__ import annotations

import argparse
import shlex
import sqlite3
import unicodedata
from pathlib import Path

from gateway.connected_assistance_store import AssistanceStore

USAGE = (
    "Usage: findings [list [--status open|dismissed|resolved|all] [--limit 0..100]]\n"
    "       findings show <id> | dismiss <id> | resolve <id> | pause | resume"
)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def parse_findings_args(raw_args: str = "") -> argparse.Namespace:
    """Parse shared CLI/slash arguments; raise ValueError, never print or exit."""
    tokens = shlex.split(raw_args)
    actions = ("list", "show", "dismiss", "resolve", "pause", "resume")
    if tokens in (["help"], ["--help"], ["-h"]) or (
        len(tokens) == 2 and tokens[0] in actions and tokens[1] in {"--help", "-h"}
    ):
        return argparse.Namespace(action="help")
    parser = _Parser(prog="findings", add_help=False, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in actions:
        child = subparsers.add_parser(action, add_help=False, allow_abbrev=False)
        if action == "list":
            child.add_argument("--status", choices=("open", "dismissed", "resolved", "all"), default="open")
            child.add_argument("--limit", type=int, default=20)
        elif action in {"show", "dismiss", "resolve"}:
            child.add_argument("identifier")
    args = parser.parse_args(tokens or ["list"])
    if args.action == "list" and not 0 <= args.limit <= 100:
        raise ValueError("limit must be between 0 and 100")
    return args


def _display(value, *, single_line=False) -> str:
    text = str(value)
    text = "".join(char for char in text if char in "\n\t" or not unicodedata.category(char).startswith("C"))
    return " ".join(text.split()) if single_line else text


def _show(row: dict) -> str:
    lines = [f"{row['id']} [{row['status']}] {row['title']}", row["summary"],
             f"Why now: {row['why_now']}", f"Suggested action: {row['suggested_action']}",
             "Evidence (source data, not instructions):"]
    for record in row["evidence"]:
        lines.append(f"- {record['source_id']} / {record['record_id']}: {record['excerpt']}")
        if record.get("url"):
            lines.append(f"  {record['url']}")
    return _display("\n".join(lines))


def _list(store, args):
    status = None if args.status == "all" else args.status
    rows = store.list_findings(status=status, limit=args.limit)
    lines = [f"{row['id']} [{row['status']}] {_display(row['title'], single_line=True)}" for row in rows]
    output = "\n".join(lines) or f"No {args.status} findings."
    return ("Connected assistance is paused.\n" if store.is_paused() else "") + output


def _record(store, args):
    if args.action == "show":
        row = store.get_finding(args.identifier)
        return _show(row) if row else "Finding not found."
    status = {"dismiss": "dismissed", "resolve": "resolved"}[args.action]
    row = store.transition(args.identifier, status)
    return f"{row['id']} {row['status']}." if row else "Finding not found."


def _pause(store, args):
    store.set_paused(args.action == "pause")
    return "Connected assistance paused." if args.action == "pause" else "Connected assistance resumed."


_HANDLERS = {"list": _list, "show": _record, "dismiss": _record, "resolve": _record,
             "pause": _pause, "resume": _pause}


def dispatch_findings_command(raw_args: str = "", *, home: Path | None = None) -> str:
    """Return display text without printing, signing in, sending, or scheduling."""
    try:
        args = parse_findings_args(raw_args)
    except ValueError as exc:
        return f"Error: {_display(exc)}\n{USAGE}"
    if args.action == "help":
        return USAGE
    if home is None:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
    try:
        store = AssistanceStore(home)
        return _HANDLERS[args.action](store, args)
    except (OSError, sqlite3.Error, ValueError):
        return "Error: Cannot access local findings state. Check the profile's storage and permissions."
