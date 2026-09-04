#!/usr/bin/env python3
"""Reminder queue poller — prints due reminders to stdout for no_agent delivery.

Designed to run as a no_agent cron job: the scheduler delivers stdout verbatim
via the job's configured delivery channel (v1 — the entry's origin field is
recorded for future per-chat routing).  Empty stdout = silent run (no
delivery).  This means
quiet nights cost zero tokens — the poller is pure Python, no LLM.

Behaviour:
  1. Read the pending reminder queue.
  2. Find entries whose due_at has passed.
  3. Print each due reminder's message to stdout (one per line, with a header).
  4. Mark each fired entry (moves to fired.log, removed from queue).
  5. If no reminders are due, print nothing → silent run.

Catch-up policy: fire late.  If the machine was asleep at due_at, the reminder
fires on the next poll tick.  This is the default — a skip-if-stale policy is
TBD per the issue.

Usage (standalone, for testing):
    python reminder_poller.py

Usage (as no_agent cron job):
    hermes cron create '* * * * *' "reminder poller" \
            --name reminder-poller \
            --script reminder_poller.py \
            --no-agent \
            --deliver telegram

Setup (one-time):
    copy this script to <HERMES_HOME>/scripts/reminder_poller.py
    then create the cron job above.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Bootstrap import path — the no_agent cron mechanism sanitizes PYTHONPATH,
# so we add the hermes-agent root ourselves.  When this script is copied to
# HERMES_HOME/scripts/, the hermes-agent root is found via HERMES_HOME env
# var (set by the scheduler) or the platform-default path.
def _bootstrap_imports() -> None:
    # When run from the repo (cron/scripts/), the root is two levels up.
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent  # cron/scripts/ -> cron/ -> root
    if (repo_root / "hermes_constants.py").exists():
        sys.path.insert(0, str(repo_root))
        return
    # When run from HERMES_HOME/scripts/, find the hermes-agent install.
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if hermes_home:
        candidate = Path(hermes_home) / "hermes-agent"
        if (candidate / "hermes_constants.py").exists():
            sys.path.insert(0, str(candidate))
            return
    # Last resort: try the platform-default HERMES_HOME
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        if local_appdata:
            candidate = Path(local_appdata) / "hermes" / "hermes-agent"
            if (candidate / "hermes_constants.py").exists():
                sys.path.insert(0, str(candidate))
                return


_bootstrap_imports()

from hermes_time import now as _now  # noqa: E402
from cron.reminder_queue import (  # noqa: E402
    add_reminder,
    due_now,
    mark_fired,
    next_occurrence,
)


def main() -> int:
    """Print due reminders, mark fired, re-arm recurring entries.

    Empty stdout = silent.
    """
    now = _now()
    due = due_now(now)
    if not due:
        return 0  # silent run — no stdout, no delivery

    lines = []
    for entry in due:
        rid = entry.get("id", "?")
        msg = entry.get("message", "")
        due_at = entry.get("due_at", "?")
        lines.append(f"⏰ Reminder ({rid}): {msg}\n  Due: {due_at}")
        # Mark fired AFTER capturing the line — if mark_fired fails we still
        # deliver (better to double-fire than to lose a reminder).
        try:
            mark_fired(rid)
        except Exception as exc:
            # Don't crash the poller — print a warning to stderr so the
            # user sees it in logs but the reminder still delivers.
            print(f"WARNING: failed to mark {rid} as fired: {exc}", file=sys.stderr)
            continue
        # Re-arm recurring entries (repeat-flag): compute the next occurrence
        # strictly after this poll tick and re-add with the same identity.
        recurring = entry.get("recurring")
        if recurring:
            try:
                next_due = next_occurrence(recurring, now)
            except Exception as exc:
                print(
                    f"WARNING: recurring rule for {rid} invalid "
                    f"({exc}); dropped after this fire",
                    file=sys.stderr,
                )
                continue
            try:
                add_reminder(
                    due_at=next_due,
                    message=msg,
                    origin=entry.get("origin") or None,
                    recurring=recurring,
                )
            except Exception as exc:
                print(
                    f"WARNING: failed to re-arm recurring reminder {rid}: {exc}",
                    file=sys.stderr,
                )

    print("\n\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
