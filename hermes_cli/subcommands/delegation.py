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
    elif state in ("completed", "dispatched"):
        return color(state, Colors.CYAN)
    elif state in ("cancelled", "timed_out", "error"):
        return color(state, Colors.RED)
    return state


# ---------------------------------------------------------------------------
# Parser builder
# ---------------------------------------------------------------------------

def build_delegation_parser(subparsers, *, cmd_delegation: Callable) -> None:
    """Attach the ``delegation`` subcommand (and its sub-actions) to ``subparsers``."""
    delegation_parser = subparsers.add_parser(
        "delegation",
        help="Async delegation management",
        description="List, inspect, cancel, and retrieve results of background subagent tasks (delegate_task background=True).",
    )
    delegation_subparsers = delegation_parser.add_subparsers(dest="delegation_command")

    # delegation list
    delegation_list = delegation_subparsers.add_parser(
        "list", help="List running and queued async delegations"
    )
    delegation_list.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    delegation_list.add_argument(
        "--completed", action="store_true", help="Also show recently completed delegations"
    )

    # delegation status
    delegation_status = delegation_subparsers.add_parser(
        "status", help="Show details for a specific delegation"
    )
    delegation_status.add_argument("delegation_id", help="Delegation ID (e.g. deleg_abc12345)")
    delegation_status.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )

    # delegation cancel
    delegation_cancel = delegation_subparsers.add_parser(
        "cancel", help="Cancel a queued or running delegation"
    )
    delegation_cancel.add_argument("delegation_id", help="Delegation ID to cancel")
    delegation_cancel.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )

    # delegation result
    delegation_result = delegation_subparsers.add_parser(
        "result", help="Retrieve the stored result of a completed delegation"
    )
    delegation_result.add_argument("delegation_id", help="Delegation ID")
    delegation_result.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    delegation_result.add_argument(
        "--clear", action="store_true", help="Remove the result from storage after retrieval"
    )

    # delegation completed
    delegation_completed = delegation_subparsers.add_parser(
        "completed", help="List recently completed delegations"
    )
    delegation_completed.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    delegation_completed.add_argument(
        "--limit", type=int, default=20, help="Max items to return (default: 20)"
    )
    delegation_completed.add_argument(
        "--clear", action="store_true", help="Clear all stored results"
    )


# ---------------------------------------------------------------------------
# Command implementation (called by cmd_delegation in main.py)
# ---------------------------------------------------------------------------

def delegation_command(args) -> int:
    """
    Implement ``hermes delegation`` subcommands.

    Returns exit code (0 = ok, 1 = error/not found).
    """
    from tools.async_delegation import (
        cancel,
        clear_completed,
        clear_result,
        count_queued,
        count_running,
        get_detail,
        get_queued_details,
        get_result,
        get_running_details,
        list_completed,
    )

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------
    if args.delegation_command == "list":
        running = get_running_details()
        queued = get_queued_details()

        if args.json:
            payload: dict = {
                "running": running,
                "queued": queued,
                "running_count": len(running),
                "queued_count": len(queued),
            }
            if getattr(args, "completed", False):
                payload["completed"] = list_completed()
            print(json.dumps(payload))
            return 0

        try:
            from hermes_cli.config import CLI_CONFIG
            max_children = int(CLI_CONFIG.get("delegation", {}).get("max_async_children", 3))
        except Exception:
            max_children = 3

        print(color("Async Delegations", Colors.BOLD))
        print(f"  Running: {len(running)}/{max_children}   Queued: {len(queued)}")
        print()

        if not running and not queued:
            print(color("  (no active async delegations)", Colors.DIM))
        else:
            if running:
                print(color("  Running:", Colors.BOLD))
                for d in running:
                    alive = color("●", Colors.GREEN) if d.get("is_alive") else color("○", Colors.DIM)
                    ts = _fmt_ts(d.get("dispatch_time", 0))
                    goal = d.get("goal") or "(no goal)"
                    timeout = f"  ⏱{d['timeout_seconds']}s" if d.get("timeout_seconds") else ""
                    print(f"  {alive} {d['delegation_id']}  [{ts}]{timeout}  {goal}")

            if queued:
                print(color("  Queued:", Colors.BOLD))
                for d in queued:
                    ts = _fmt_ts(d.get("enqueue_time", 0))
                    goal = d.get("goal") or "(no goal)"
                    timeout = f"  ⏱{d['timeout_seconds']}s" if d.get("timeout_seconds") else ""
                    print(f"    ○ {d['delegation_id']}  [{ts}]{timeout}  {goal}")

        if getattr(args, "completed", False):
            completed = list_completed()
            if completed:
                print()
                print(color("  Completed (recent):", Colors.BOLD))
                for d in completed:
                    ts = _fmt_ts(d.get("completion_time", 0))
                    status_label = _color_state(d.get("status", ""))
                    dur = f"  ({d['duration_seconds']:.1f}s)" if d.get("duration_seconds") is not None else ""
                    goal = d.get("goal") or "(no goal)"
                    print(f"    {d['delegation_id']}  [{ts}] {status_label}{dur}  {goal}")

        return 0

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------
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
            timeout = f"  timeout in {detail['timeout_seconds']}s" if detail.get("timeout_seconds") else ""
            print(f"  Started:  {ts}  ({alive}){timeout}")
        elif detail["state"] == "queued":
            ts = _fmt_ts(detail.get("enqueue_time", 0))
            timeout = f"  (timeout: {detail['timeout_seconds']}s)" if detail.get("timeout_seconds") else ""
            print(f"  Queued:   {ts}{timeout}")
        else:
            # Completed state
            ts = _fmt_ts(detail.get("completion_time", 0))
            dur = f"  ({detail['duration_seconds']:.1f}s)" if detail.get("duration_seconds") is not None else ""
            print(f"  Finished: {ts}{dur}")
            print(f"  Result:   {'available — use `hermes delegation result <id>`' if detail.get('result_available') else 'not stored'}")

        return 0

    # ------------------------------------------------------------------
    # cancel
    # ------------------------------------------------------------------
    if args.delegation_command == "cancel":
        result = cancel(args.delegation_id)

        if args.json:
            print(json.dumps(result))
            return 0 if result["cancelled"] else 1

        if not result["cancelled"]:
            print(color(f"delegation '{args.delegation_id}' not found or already completed", Colors.RED))
            return 1

        state = result["state"]
        if state == "queued":
            print(color(f"✓ delegation '{args.delegation_id}' removed from queue", Colors.GREEN))
        else:
            print(color(f"✓ cancel requested for running delegation '{args.delegation_id}'", Colors.YELLOW))
            print("  The subagent will be marked cancelled when it completes.")
        return 0

    # ------------------------------------------------------------------
    # result
    # ------------------------------------------------------------------
    if args.delegation_command == "result":
        result = get_result(args.delegation_id)

        if args.json:
            if result is None:
                print(json.dumps({"error": "not found", "delegation_id": args.delegation_id}))
                return 1
            print(json.dumps(result))
            if getattr(args, "clear", False):
                clear_result(args.delegation_id)
            return 0

        if result is None:
            print(color(f"No stored result for delegation '{args.delegation_id}'", Colors.RED))
            print("  (Results are only stored for completed delegations; running/queued tasks have no result yet.)")
            return 1

        status_label = _color_state(result.get("status", ""))
        ts = _fmt_ts(result.get("completion_time", 0))
        dur = f"  ({result['duration_seconds']:.1f}s)" if result.get("duration_seconds") is not None else ""
        print(color("Async Delegation Result", Colors.BOLD))
        print(f"  ID:       {result.get('delegation_id', '')}")
        print(f"  Status:   {status_label}")
        print(f"  Finished: {ts}{dur}")
        print(f"  Goal:     {result.get('goal') or '(none)'}")
        print()
        print(color("  Result:", Colors.BOLD))
        inner = result.get("result", "")
        if isinstance(inner, dict):
            print("  " + json.dumps(inner, indent=2).replace("\n", "\n  "))
        else:
            print(f"  {inner}")

        if getattr(args, "clear", False):
            clear_result(args.delegation_id)
            print()
            print(color("  (result cleared from storage)", Colors.DIM))
        return 0

    # ------------------------------------------------------------------
    # completed
    # ------------------------------------------------------------------
    if args.delegation_command == "completed":
        if getattr(args, "clear", False):
            n = clear_completed()
            if args.json:
                print(json.dumps({"cleared": n}))
            else:
                print(color(f"Cleared {n} stored result(s).", Colors.GREEN))
            return 0

        items = list_completed(limit=getattr(args, "limit", 20))

        if args.json:
            print(json.dumps({"completed": items, "count": len(items)}))
            return 0

        if not items:
            print(color("  (no stored completed delegations)", Colors.DIM))
            return 0

        print(color("Completed Delegations", Colors.BOLD))
        for d in items:
            ts = _fmt_ts(d.get("completion_time", 0))
            status_label = _color_state(d.get("status", ""))
            dur = f"  ({d['duration_seconds']:.1f}s)" if d.get("duration_seconds") is not None else ""
            goal = d.get("goal") or "(no goal)"
            print(f"  {d['delegation_id']}  [{ts}] {status_label}{dur}  {goal}")
        return 0

    return 0
