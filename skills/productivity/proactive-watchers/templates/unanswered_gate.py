#!/usr/bin/env python3
"""Unanswered-message gate for a cron job: wake the agent only when someone is waiting on the user.

Pass this file's text as ``script_content`` to ``cronjob_manage`` (schedule e.g. ``every 15m``,
weekdays). Each tick it asks the connected app for recent messages addressed to the user through
``hermes connectors call`` (read-only), keeps the newest timestamp it has seen in
$HERMES_CRON_STATE_DIR, and:
- prints {"wakeAgent": false} when nothing new is waiting (no model call, no tokens);
- prints {"wakeAgent": true, "context": {"waiting": [...]}} with only the new messages otherwise,
  so the agent drafts replies for those and nothing else.

Set TOOL / ARGS to a read tool of the user's connected app (find one with
``hermes connectors call`` --help or the app's tool list) and MESSAGES_KEY / TS_KEY to where the
response keeps the message list and each message's timestamp. A failed call exits non-zero, which
the scheduler reports as a watchdog alert instead of waking the agent.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

TOOL = "slack.search_messages"           # <app>.<read tool>
ARGS = {"query": "to:me is:dm", "count": 20}
MESSAGES_KEY = "messages"                # list of messages in the tool's JSON response
TS_KEY = "ts"                            # per-message sortable timestamp field
BATCH = 10                               # messages handed to the agent per tick


def fetch(tool, args):
    """The connector tool's JSON response; raises on a failed call so the job alerts."""
    out = subprocess.run(["hermes", "connectors", "call", tool, "--args", json.dumps(args)],
                         capture_output=True, text=True, encoding="utf-8", timeout=120, check=True)
    return json.loads(out.stdout)


def new_messages(response, last_seen):
    """Messages newer than ``last_seen``, oldest first."""
    items = response.get(MESSAGES_KEY) or []
    if isinstance(items, dict):  # some apps nest the list: {"messages": {"matches": [...]}}
        items = next((v for v in items.values() if isinstance(v, list)), [])
    fresh = [m for m in items if isinstance(m, dict) and str(m.get(TS_KEY, "")) > last_seen]
    return sorted(fresh, key=lambda m: str(m.get(TS_KEY, "")))


def main() -> int:
    state_dir = Path(os.environ.get("HERMES_CRON_STATE_DIR") or Path(__file__).with_suffix(".state"))
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "last_seen.txt"
    last_seen = state_file.read_text().strip() if state_file.exists() else ""
    fresh = new_messages(fetch(TOOL, ARGS), last_seen)
    if not fresh:
        print(json.dumps({"wakeAgent": False}))
        return 0
    batch = fresh[:BATCH]  # the cursor stops at the last message shown, so the rest come next tick
    state_file.write_text(str(batch[-1].get(TS_KEY, "")))
    print(json.dumps({"wakeAgent": True, "context": {"waiting": batch, "source": TOOL}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
