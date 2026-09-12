"""Durable skill-review cadence on the existing profile state_meta store.

A normal conversation owns a counter across its compression lineage. Kanban
workers share one profile counter across cards; it schedules a review of the
current card, not a replay of unrelated cards or another profile's history.
Memory cadence remains owned by the existing turn prologue.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
_PREFIX = "skill-review-cadence:"


def _scope(agent: Any):
    from hermes_state import SessionDB
    db = getattr(agent, "_session_db", None)
    sid = getattr(agent, "session_id", None)
    if not isinstance(db, SessionDB) or not isinstance(sid, str) or not sid:
        return None
    row = db.get_session(sid) or {}
    if row.get("source") == "kanban":
        key = _PREFIX + "kanban"
    else:
        lineage = db.get_compression_lineage(sid)
        key = _PREFIX + "session:" + (lineage[0] if lineage else sid)
    return db, key


def prepare_skill_review_cadence(agent: Any) -> None:
    """A reused agent may change conversation; never carry unsaved effort across it."""
    if getattr(agent, "_persist_disabled", False) or getattr(agent, "_delegate_depth", 0):
        return
    sid = getattr(agent, "session_id", None)
    previous_sid = vars(agent).get("_review_cadence_session")
    if previous_sid and previous_sid != sid:
        try:
            db = getattr(agent, "_session_db", None)
            if db is None or previous_sid not in db.get_compression_lineage(sid):
                agent._iters_since_skill = 0
                agent._review_cadence_reset = False
        except Exception:
            # Unknown lineage must not couple unrelated conversations.
            agent._iters_since_skill = 0
            agent._review_cadence_reset = False
    agent._review_cadence_session = sid


def reset_skill_review_cadence(agent: Any) -> None:
    """Keep foreground skill_manage's established reset semantics."""
    agent._iters_since_skill = 0
    agent._review_cadence_reset = True


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _advance(db, key: str, delta: int, *, reset: bool, interval: int, claim: bool) -> tuple[int, bool]:
    def write(conn):
        row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
        total = (0 if reset else _count(row[0] if row else None)) + delta
        due = claim and total >= interval
        db.set_meta(key, str(0 if due else total), cursor=conn)
        return total, due
    return db._execute_write(write)


def schedule_turn_review(agent: Any, messages: list, *, final_response: Any,
                         interrupted: bool, review_memory: bool = False) -> None:
    """Commit effort, atomically claim a due review, refund rejected admission.

    No schema/provider mutation: the review still receives the same turn snapshot.
    Persistence failures degrade to the existing live-agent counter and are logged.
    """
    if (getattr(agent, "skip_background_review", False)
            or getattr(agent, "_persist_disabled", False) or getattr(agent, "_delegate_depth", 0)):
        return
    from agent.background_review import load_background_review_settings
    if not load_background_review_settings()[0]:
        return
    interval = _count(getattr(agent, "_skill_nudge_interval", 0))
    allowed = interval > 0 and "skill_manage" in getattr(agent, "valid_tool_names", ())
    may_spawn = bool(final_response) and not interrupted
    total = _count(getattr(agent, "_iters_since_skill", 0))
    due = allowed and may_spawn and total >= interval
    scope = None
    if allowed:
        try:
            scope = _scope(agent)
            if scope is not None:
                db, key = scope
                # A foreground write resets only its conversation, never other cards'
                # concurrently accumulated effort on the profile-wide Kanban clock.
                reset = vars(agent).get("_review_cadence_reset", False) and key != _PREFIX + "kanban"
                total, due = _advance(db, key, total, reset=reset, interval=interval, claim=may_spawn)
                agent._iters_since_skill = 0
                agent._review_cadence_reset = False
        except Exception:
            scope = None
            logger.warning("Could not persist skill-review cadence; using live counter", exc_info=True)
    if not may_spawn or not (review_memory or due):
        return
    if due and scope is None:
        agent._iters_since_skill = 0
    try:
        accepted = agent._spawn_background_review(
            messages_snapshot=list(messages), review_memory=review_memory, review_skills=due,
        ) is not False  # existing plugin callbacks may return None
    except Exception:
        accepted = False
        logger.warning("Background review admission failed", exc_info=True)
    if due and not accepted:
        if scope is None:
            agent._iters_since_skill += total
        else:
            try:
                _advance(*scope, total, reset=False, interval=interval, claim=False)
            except Exception:
                logger.warning("Could not refund rejected review cadence claim", exc_info=True)
