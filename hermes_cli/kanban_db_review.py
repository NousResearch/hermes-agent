"""Review routing: who owns a handoff, and who reviewed the card last.

Split out of ``kanban_db`` because that module is over its file cap and may only
shrink. Both rules here answer one question the review lane depends on, so a
topical sibling is the shape the repo asks for.
"""
from __future__ import annotations

import sqlite3
from typing import Optional


def default_reviewer_profile() -> Optional[str]:
    """``kanban.reviewer_profile`` when it names a profile this host can spawn.

    A worker's ``kanban_request_review`` call carries ``reviewer`` only when the
    model happens to supply one, and :func:`prior_reviewer` answers ``None`` on a
    FIRST review. The card then kept the profile that DID the work, and the
    review lane spawns the row's own assignee
    (``_dispatch_lane_task(..., row["assignee"], lane="review")``), so the author
    graded itself while the reviewer lane sat idle. This is the code-side default
    the config key already promises.

    Unset, unreadable, or a name with no live profile directory returns ``None``,
    which keeps the previous behaviour rather than parking a card on an assignee
    the dispatcher can never spawn.
    """
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.profiles import profile_exists

        cfg = load_config_readonly() or {}
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        name = (kanban_cfg.get("reviewer_profile") or "").strip()
        if not name:
            return None
        return name if profile_exists(name) else None
    except Exception:  # health: allow BLE001 -- a config fault must never misroute a handoff
        return None


def prior_reviewer(conn: sqlite3.Connection, task_id: str):
    """Reviewer recorded by the latest ``changes_requested`` run's event.

    ``None`` = first review (no such run); ``False`` = a run exists but its
    provenance is missing or malformed, which the caller must refuse rather than
    guess at.
    """
    from hermes_cli.kanban_db import _json_dict, _latest_event, _row_get

    changes_run = conn.execute(
        "SELECT id FROM task_runs "
        "WHERE task_id = ? AND outcome = 'changes_requested' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if changes_run is None:
        return None
    changes_event = _latest_event(conn, task_id, "changes_requested", changes_run["id"])
    reviewer = _json_dict(_row_get(changes_event, "payload")).get("reviewer")
    return reviewer if isinstance(reviewer, str) and reviewer.strip() else False
