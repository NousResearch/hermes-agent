"""Conditional goal persistence shared by CLI managers and plugin facades."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GoalChangedDuringEvaluation(RuntimeError):
    """A competing writer won; no later stage may evaluate or mutate its goal."""


def save_if_unchanged(session_id: str, state, expected: str | None) -> bool | None:
    """Return False on contention, None on unavailable storage, True on commit.

    Compare decoded state first for legacy JSON compatibility, then use the exact
    stored bytes in the atomic write so a concurrent mutation always wins.
    """
    from hermes_cli import goals

    db = goals._get_session_db()
    if db is None or not session_id:
        return None
    try:
        key = goals._meta_key(session_id)
        raw = db.get_meta(key)
        observed = goals.GoalState.from_json(raw).to_json() if raw else None
        if observed != expected:
            return False
        value = state.to_json()
        if raw is None:
            changed = db._write_rowcount(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO NOTHING",
                (key, value),
            )
        else:
            changed = db._write_rowcount(
                "UPDATE state_meta SET value = ? WHERE key = ? AND value = ?",
                (value, key, raw),
            )
        return changed == 1
    except Exception as exc:
        logger.exception("GoalManager: conditional persistence failed: %s", exc)
        return None
