"""Human CLI for listing and resolving core-owned protocol retry holds."""

from __future__ import annotations

import json
import os
import sys

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_output import _err


def _list_holds(args) -> int:
    """Read active holds and their concurrency tokens from one SQLite snapshot."""
    with kbc.connect_closing() as conn:
        conn.execute("BEGIN")
        try:
            task_ids = [
                row["task_id"] for row in conn.execute(
                    "SELECT task_id FROM task_retry_holds ORDER BY created_at, task_id"
                ).fetchall()
            ]
            holds = [hold for task_id in task_ids if (hold := kb.get_retry_hold(conn, task_id)) is not None]
        finally:
            conn.rollback()

    if args.json:
        print(json.dumps({"holds": holds}, ensure_ascii=False, sort_keys=True))
    elif not holds:
        print("No active protocol retry holds.")
    else:
        for hold in holds:
            assignee = hold["assignee"] or "-"
            print(
                f"{hold['task_id']} status={hold['status']} assignee={assignee} "
                f"failed_run_id={hold['failed_run_id']} hold_token={hold['hold_token']}"
            )
    return 0


def _resolve_hold(args) -> int:
    """Resolve through the existing fenced API; no CLI-side task mutations."""
    if os.environ.get("HERMES_KANBAN_TASK", "").strip():
        return _err("kanban holds resolve is operator-only; run it outside a worker")

    with kbc.connect_closing() as conn:
        receipt = kb.resolve_retry_hold(
            conn,
            args.task_id,
            expected_hold_token=args.expected_hold_token,
            action=args.action,
            reason=args.reason,
        )
    if receipt.get("ok") is not True:
        reason = receipt.get("reason", "resolution_refused")
        hint = "; run `hermes kanban holds list` for the current token" if reason == "stale_hold_token" else ""
        return _err(f"cannot resolve retry hold for {args.task_id}: {reason}{hint}")

    verb = "Already resolved" if receipt.get("idempotent") else "Resolved"
    print(
        f"{verb} {args.task_id}: {receipt['action']} -> {receipt['status']} "
        f"(event {receipt['resolution_event_id']})"
    )
    return 0


def cmd_holds(args) -> int:
    """Dispatch ``hermes kanban holds list|resolve``."""
    action = getattr(args, "holds_action", None)
    if action == "list":
        return _list_holds(args)
    if action == "resolve":
        return _resolve_hold(args)
    print("Usage: hermes kanban holds {list|resolve} --help", file=sys.stderr)
    return 2
