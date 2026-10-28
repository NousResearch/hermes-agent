"""Kanban dispatch waits out a provider usage limit instead of spawning workers into it.

A worker that dies on a plan usage limit records a provider hold in its profile
(``agent.usage_hold``). Every other ready card for that profile would spawn a worker that hits
the same wall, burn a request, and requeue as ``rate_limited``. The dispatcher asks here first:
a card whose worker would call a held provider stays ``ready`` (no claim, no failure counted)
and dispatches on the first tick after the reset. The board gets ONE ``usage_held`` event per
card per hold, not one per tick.

Only what the assignee profile's config.yaml (or the card's own provider override) states is
held; a configured fallback chain means the worker's normal fallback runs instead.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

EVENT_KIND = "usage_held"


def _profile_config(home: Path) -> dict[str, Any]:
    from hermes_cli.config_effective import load_user_config_effective

    return load_user_config_effective(home / "config.yaml")


def hold_for_card(row: sqlite3.Row, assignee: str) -> Optional[dict[str, Any]]:
    """The live usage hold the card's worker would run into, or None."""
    from agent.usage_hold import active_hold, configured_route
    from hermes_cli.profiles import get_profile_dir

    try:
        home = get_profile_dir(assignee)
    except ValueError:
        return None
    provider, base_url, has_fallback = configured_route(_profile_config(home))
    keys = row.keys()
    override = str(row["provider_override"] or "") if "provider_override" in keys else ""
    if override and override != provider:
        provider, base_url = override, ""
    if has_fallback or not provider:
        return None
    return active_hold(provider, base_url, home=home)


def note_held_card(conn: sqlite3.Connection, task_id: str, hold: dict[str, Any], *, dry_run: bool) -> None:
    """One board event per card per hold: a repeat tick inside the same window writes nothing."""
    from hermes_cli import kanban_db as _kb

    if dry_run:
        return
    payload = {"provider": str(hold.get("provider") or ""), "until": int(float(hold["until"]))}
    with _kb.write_txn(conn):
        last = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
            (task_id,)).fetchone()
        if last is not None and last["kind"] == EVENT_KIND and last["payload"] == _kb._json_or_null(payload):
            return
        _kb._append_event(conn, task_id, EVENT_KIND, payload)
    from agent.usage_hold import format_until

    logger.warning("kanban dispatcher: %s waits for %s's usage limit to reset at %s",
                   task_id, payload["provider"], format_until(hold))
