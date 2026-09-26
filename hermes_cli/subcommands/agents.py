"""``hermes agents`` cross-process, profile-scoped fleet monitor."""

from __future__ import annotations

import json
import argparse
import math
import sys
import time
from typing import Any

from hermes_constants import get_hermes_home


def _positive_interval(value: str) -> float:
    try:
        interval = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("interval must be a number") from exc
    if not math.isfinite(interval) or interval < 0.2:
        raise argparse.ArgumentTypeError("interval must be finite and at least 0.2 seconds")
    return interval


def build_agents_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "agents",
        help="Monitor delegated agents across this profile",
        description=(
            "Show live delegated agents owned by every Hermes process sharing "
            "the active profile. This fleet view is read-only."
        ),
    )
    parser.add_argument(
        "--once", action="store_true", help="Render one table and exit"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit one machine-readable snapshot and exit"
    )
    parser.add_argument(
        "--interval", type=_positive_interval, default=1.5,
        help="Refresh interval in seconds for live mode (default: 1.5)",
    )
    parser.set_defaults(func=cmd_agents)


def _snapshot() -> dict[str, Any]:
    from tools.delegation_live_log import scan_live_delegations

    return {
        "schema_version": 1,
        "observed_at": time.time(),
        "profile_home": str(get_hermes_home()),
        "children": scan_live_delegations(),
    }


def _age(timestamp: Any, observed_at: float) -> str:
    try:
        parsed = float(timestamp)
        if not math.isfinite(parsed):
            return "?"
        seconds = max(0, int(observed_at - parsed))
    except (TypeError, ValueError, OverflowError):
        return "?"
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _table(snapshot: dict[str, Any]):
    from rich.table import Table
    from rich.text import Text

    children = snapshot["children"]
    if not children:
        return "No live delegated agents in this profile."
    table = Table(title="Hermes agent fleet", expand=True)
    table.add_column("Session / PID", no_wrap=True)
    table.add_column("Agent", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Age", justify="right", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Last tool", no_wrap=True)
    table.add_column("Goal", overflow="ellipsis")
    observed_at = float(snapshot["observed_at"])
    for child in children:
        owner = child.get("owner_session_id") or "unknown-session"
        owner = f"{owner} / {child.get('owner_pid', '?')}"
        agent = child.get("subagent_id") or (
            f"{child.get('delegation_id', '?')}#{child.get('task_index', '?')}"
        )
        table.add_row(
            Text(owner),
            Text(str(agent)),
            Text(str(child.get("status") or "?")),
            Text(_age(child.get("updated_at"), observed_at)),
            Text(str(child.get("model") or "?")),
            Text(str(child.get("last_tool") or "—")),
            Text(str(child.get("goal") or "")),
        )
    table.caption = "Read-only fleet view • transcripts are listed by --json"
    return table


def cmd_agents(args) -> int:
    """Render a one-shot/JSON snapshot or continuously refresh on a TTY."""
    if args.json:
        print(json.dumps(_snapshot(), ensure_ascii=False, sort_keys=True))
        return 0

    from rich.console import Console

    console = Console()
    if args.once or not sys.stdout.isatty():
        console.print(_table(_snapshot()))
        return 0

    from rich.live import Live

    try:
        with Live(_table(_snapshot()), console=console, refresh_per_second=4) as live:
            while True:
                time.sleep(args.interval)
                live.update(_table(_snapshot()), refresh=True)
    except KeyboardInterrupt:
        return 130
