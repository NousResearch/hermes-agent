#!/usr/bin/env python3
"""Moments gate for a cron job: wake the agent only when a planned moment is due.

Fill MOMENTS, then pass this file's text as ``script_content`` to ``cronjob_manage``
(schedule e.g. ``every 15m`` with a ``repeat`` cap covering the trip). Each tick:
- prints {"wakeAgent": false} while nothing is due (no model call, no tokens);
- when one or more moments are due and not yet handled, prints
  {"wakeAgent": true, "context": {"due": [...], "now": ...}} so the agent sees only those moments;
- remembers handled moments in $HERMES_CRON_STATE_DIR/moments.json, so each fires once.

A moment: {"id": unique str, "at": ISO-8601 time WITH offset, "label": what it is,
"window_minutes": how long after `at` it still counts (default 90), "context": anything the agent
needs to act (flight number, addresses, links)}.
"""
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

MOMENTS = [
    # {"id": "checkin", "at": "2026-11-26T08:00:00-08:00", "label": "Online check-in opens",
    #  "window_minutes": 180, "context": {"flight": "NH7", "airline_url": "https://..."}},
]


def due_moments(moments, handled, now):
    """Moments whose time has come, that are still inside their window and were not handled yet."""
    due = []
    for moment in moments:
        at = datetime.fromisoformat(moment["at"])
        if at.tzinfo is None:
            raise ValueError(f"moment {moment['id']!r} needs a UTC offset in 'at'")
        window = timedelta(minutes=int(moment.get("window_minutes", 90)))
        if moment["id"] not in handled and at <= now <= at + window:
            due.append(moment)
    return due


def main() -> int:
    state_dir = Path(os.environ.get("HERMES_CRON_STATE_DIR") or Path(__file__).with_suffix(".state"))
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "moments.json"
    handled = set(json.loads(state_file.read_text())) if state_file.exists() else set()
    now = datetime.now(UTC)
    due = due_moments(MOMENTS, handled, now)
    if not due:
        print(json.dumps({"wakeAgent": False}))
        return 0
    state_file.write_text(json.dumps(sorted(handled | {m["id"] for m in due})))
    print(json.dumps({"wakeAgent": True, "context": {"due": due, "now": now.isoformat()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
