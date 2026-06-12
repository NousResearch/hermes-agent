"""``hermes delegation`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()``. The handler is injected so
this module does not import ``main`` (cycle avoidance).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from hermes_cli.colors import Colors, color


def _fmt_ts(ts: float) -> str:
    """Format a Unix timestamp to local HH:MM:SS."""
    if not ts:
        return "—"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    return dt.strftime("%H:%M:%S")


def _color_state(state: str) -> str:
    if state == "running":
        return color("running", Colors.GREEN)
    elif state == "queued":
        return color("queued", Colors.YELLOW)
    return state


# ---------------------------------------------------------------------------
# Parser builder
# ---------------------------------------------------------------------------

def build_delegation_parser(subparsers, *, cmd_delegation: Callable) -> None:
    """Attach the ``delegation`` subcommand (and its sub-actions) to ``subparsers``."""
    delegation_parser = subparsers.add_parser(
        "delegation",
        help="Async delegation management",
        description="List and inspect background subagent tasks (delegate_task background=True).",
    )
    delegation_subparsers = delegation_parser.add_subparsers(dest="delegation_command")

    # delegation list
    delegation_list = delegation_subparsers.add_parser(
        "list", help="List running and queued async delegations"
    )
    delegation_list.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )

    # delegation status
    delegation_status = delegation_subparsers.add_parser(
        "status", help="Show details for a specific delegation"
    )
    delegation_status.add_argument("delegation_id", help="Delegation ID (e.g. deleg_abc12345)")
    delegation_status.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )


# ---------------------------------------------------------------------------
# Command implementation (called by cmd_delegation in main.py)
# ---------------------------------------------------------------------------

def delegation_command(args) -> int:
    """
    Implement ``hermes delegation list`` and ``hermes delegation status``.

    Returns exit code (0 = ok, 1 = error/not found).
    """
    from tools.async_delegation import (
        count_queued,
        count_running,
        get_detail,
        get_queued_details,
        get_running_details,
    )

    if args.delegation_command == "list":
        running = get_running_details()
        queued = get_queued_details()

        if args.json:
            print(json.dumps({
                "running": running,
                "queued": queued,
                "running_count": len(running),
                "queued_count": len(queued),
            }))
            return 0

        max_children = 3  # display only; actual config may differ
        print(color("Async Delegations", Colors.BOLD))
        print(f"  Running: {len(running)}/{max_children}   Queued: {len(queued)}")
        print()

        if not running and not queued:
            print(color("  (no active async delegations)", Colors.DIM))
            return 0

        if running:
            print(color("  Running:", Colors.BOLD))
            for d in running:
                alive = color("●", Colors.GREEN) if d.get("is_alive") else color("○", Colors.DIM)
                ts = _fmt_ts(d.get("dispatch_time", 0))
                goal = d.get("goal") or "(no goal)"
                print(f"  {alive} {d['delegation_id']}  [{ts}]  {goal}")

        if queued:
            print(color("  Queued:", Colors.BOLD))
            for d in queued:
                ts = _fmt_ts(d.get("enqueue_time", 0))
                goal = d.get("goal") or "(no goal)"
                print(f"    ○ {d['delegation_id']}  [{ts}]  {goal}")

        return 0

    if args.delegation_command == "status":
        detail = get_detail(args.delegation_id)

        if args.json:
            if detail is None:
                print(json.dumps({"error": "not found", "delegation_id": args.delegation_id}))
                return 1
            print(json.dumps(detail))
            return 0

        if detail is None:
            print(color(f"delegation '{args.delegation_id}' not found", Colors.RED))
            return 1

        state_label = _color_state(detail["state"])
        print(color("Async Delegation", Colors.BOLD))
        print(f"  ID:       {detail['delegation_id']}")
        print(f"  State:    {state_label}")
        print(f"  Goal:     {detail.get('goal') or '(none)'}")

        if detail["state"] == "running":
            ts = _fmt_ts(detail.get("dispatch_time", 0))
            alive = "alive" if detail.get("is_alive") else "finished"
            print(f"  Started:  {ts}  ({alive})")
        elif detail["state"] == "queued":
            ts = _fmt_ts(detail.get("enqueue_time", 0))
            print(f"  Queued:   {ts}")

        return 0

    return 0