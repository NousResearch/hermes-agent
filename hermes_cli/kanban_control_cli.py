"""Operator CLI for standalone per-board Kanban pickup control."""

from __future__ import annotations

import argparse

from hermes_cli import kanban_control
from hermes_cli.kanban_output import _err, _json_out


def _actor(args: argparse.Namespace) -> str:
    if getattr(args, "by", None):
        return args.by
    from hermes_cli.profiles import current_profile_name
    return current_profile_name("user") or "user"


def _print_result(args: argparse.Namespace, result: dict) -> int:
    if _json_out(args, result):
        return 0 if result["state"] in {"running", "paused"} else 1
    print(f"Kanban pickups: {result['state']}")
    for row in result["boards"]:
        line = f"  {row['board']}: {row['state']} (revision {row['revision']})"
        if row.get("changed_by"):
            line += f" by {row['changed_by']}"
        if row.get("reason"):
            line += f" — {row['reason']}"
        if row.get("error_code"):
            line += f" [error: {row['error_code']}]"
        print(line)
    return 0 if result["state"] in {"running", "paused"} else 1


def _cmd_pause(args: argparse.Namespace) -> int:
    try:
        result = kanban_control.set_control_scope(
            "paused", changed_by=_actor(args), reason=args.reason,
            board=getattr(args, "board", None),
        )
    except ValueError as exc:
        return _err(f"kanban pause: {exc}", 2)
    return _print_result(args, result)


def _cmd_resume(args: argparse.Namespace) -> int:
    try:
        result = kanban_control.set_control_scope(
            "running", changed_by=_actor(args), reason=args.reason,
            board=getattr(args, "board", None),
        )
    except ValueError as exc:
        return _err(f"kanban resume: {exc}", 2)
    return _print_result(args, result)


def _cmd_pause_status(args: argparse.Namespace) -> int:
    try:
        result = kanban_control.control_status(board=getattr(args, "board", None))
    except ValueError as exc:
        return _err(f"kanban pause-status: {exc}", 2)
    return _print_result(args, result)
