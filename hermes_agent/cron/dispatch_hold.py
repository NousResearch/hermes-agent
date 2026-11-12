"""A gateway drain holds cron dispatch; a one-shot due during the hold keeps its grace window.

While the gateway drains (a restart or ``hermes update`` waiting on in-flight turns, up to
``agent.restart_after_turn_timeout``), the in-process ticker skips dispatch (``can_dispatch``).
The one-shot grace (120s) kept counting from the due time, so a one-shot due mid-drain was retired
as missed by the next gateway's first tick whenever the drain outlasted the grace: an update that
waited three minutes on a Kanban turn dropped the reminder the user scheduled for that minute.

Each held tick stamps ``<cron store>/dispatch_held.json`` (``since`` = first held tick, ``last`` =
latest). It crosses the restart: the next gateway reads it from the same store. A one-shot that
came due during the hold (and was not already expired when it began) counts its grace from the
hold's last tick instead of its due time, so it fires once when dispatch resumes; a hold whose
gateway never came back within the grace still retires it (#93526: no stale reminders after
downtime).
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Optional

_FILE = "dispatch_held.json"
# A held tick refreshes ``last`` once per ticker interval; the hold may have lasted this much longer.
TICK_SLACK_SECONDS = 60.0
# The longest a restart drain waits on in-flight turns by default (agent.restart_after_turn_timeout).
MAX_EXTENSION_SECONDS = 1800.0
# This process's ongoing holds: cron dir -> first held tick. Kept in memory so a failed write
# mid-drain cannot reset ``since``; cleared when the ticker dispatches again (note_dispatching).
_hold_since: dict[str, float] = {}


def note_held(now: Optional[float] = None) -> None:
    """Record that this tick skipped dispatch for the active cron store (a drain is holding it)."""
    from hermes_agent.cron.jobs import _current_cron_store
    from utils import atomic_json_write

    cron_dir = _current_cron_store().cron_dir
    now = time.time() if now is None else now
    if str(cron_dir) not in _hold_since:
        held = read(cron_dir)
        # Another process's hold still being stamped continues; anything older starts a new hold.
        since = now
        if held and now - held["last"] <= 2 * TICK_SLACK_SECONDS:
            since = held["since"]
        _hold_since[str(cron_dir)] = since
    with contextlib.suppress(OSError):  # an unwritable store is reported by the tick itself
        atomic_json_write(Path(cron_dir) / _FILE, {"since": _hold_since[str(cron_dir)], "last": now})


def note_dispatching() -> None:
    """The ticker dispatches the active cron store again: this process's hold on it has ended."""
    from hermes_agent.cron.jobs import _current_cron_store

    _hold_since.pop(str(_current_cron_store().cron_dir), None)


def note_held_profiles(profiles) -> None:
    """``note_held`` for each ``(name, home)`` a multiplex ticker serves, inside its cron scope."""
    from hermes_agent.cron.scheduler_provider import _profile_cron_scope

    for _name, home in profiles:
        with _profile_cron_scope(home):
            note_held()


def read(cron_dir: Path) -> Optional[dict]:
    try:
        data = json.loads((Path(cron_dir) / _FILE).read_text(encoding="utf-8-sig"))
        return {"since": float(data["since"]), "last": float(data["last"])}
    except (OSError, ValueError, TypeError, KeyError):
        return None


def grace_starts_at(cron_dir: Path, due_at: float, grace: float) -> float:
    """When a one-shot due at *due_at* starts its *grace*: the end of a hold it came due in,
    else its due time. Only a one-shot due within MAX_EXTENSION_SECONDS of the hold's end is
    extended: an external drain has no time limit, and a reminder due hours into it is stale."""
    held = read(cron_dir)
    if (held and held["since"] - grace <= due_at <= held["last"] + TICK_SLACK_SECONDS
            and due_at >= held["last"] - MAX_EXTENSION_SECONDS):
        return max(due_at, held["last"] + TICK_SLACK_SECONDS)
    return due_at
