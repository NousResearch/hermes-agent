"""Optimistic goal evaluation, with no database lock held over gates or model calls."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hermes_cli.goals import GoalManager, GoalState
    from hermes_state import SessionDB

logger = logging.getLogger(__name__)


class GoalEvaluationConflict(Exception):
    """A command or another evaluator changed the state being judged."""


class GoalEvaluationPersistenceError(Exception):
    """The evaluator could not read or commit durable state."""


def _read_goal(session_id: str, db: SessionDB) -> GoalState | None:
    from hermes_cli.goals import GoalState, _meta_key

    try:
        raw = db.get_meta(_meta_key(session_id))
        return GoalState.from_json(raw) if raw else None
    except Exception as exc:
        raise GoalEvaluationPersistenceError from exc


def assert_goal_snapshot(session_id: str, expected: str, db: SessionDB) -> None:
    current = _read_goal(session_id, db)
    if current is None or current.to_json() != expected:
        raise GoalEvaluationConflict


def _reload_if_readable(manager: GoalManager, db: SessionDB | None) -> None:
    if db is not None:
        try:
            manager._state = _read_goal(manager.session_id, db)
        except GoalEvaluationPersistenceError:
            # An unreadable DB is not evidence that the goal was deleted.
            logger.warning("Could not reload goal %s; retaining in-memory state", manager.session_id)


def evaluate_goal_snapshot(manager: GoalManager, last_response: str, **kwargs: Any) -> dict[str, Any]:
    from hermes_cli.goals import GoalState, _decision, _get_session_db, _meta_key

    state = manager.state
    if state is None or state.status != "active":
        return _decision(state.status if state else None, False, None, "inactive", "no active goal", "")
    expected = state.to_json()
    db = None
    try:
        try:
            db = _get_session_db()
        except Exception as exc:
            raise GoalEvaluationPersistenceError from exc
        if db is None:
            raise GoalEvaluationPersistenceError("session DB unavailable")
        assert_goal_snapshot(manager.session_id, expected, db)
        # Commands can use the original manager while the judge runs. Only this
        # isolated evaluator stages writes, so those commands still persist normally.
        evaluator = object.__new__(type(manager))
        evaluator.__dict__ = manager.__dict__.copy()
        evaluator._state = GoalState.from_json(expected)
        evaluator._evaluation_snapshot = (db, expected)
        decision = evaluator._evaluate_after_turn(last_response, **kwargs)
        result = evaluator.state

        def commit(conn):
            row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (_meta_key(manager.session_id),)).fetchone()
            if not row or not row[0] or GoalState.from_json(row[0]).to_json() != expected:
                raise GoalEvaluationConflict
            if result.to_json() != expected:
                result.mutation_id = uuid.uuid4().hex
                conn.execute("UPDATE state_meta SET value = ? WHERE key = ?", (result.to_json(), _meta_key(manager.session_id)))

        try:
            db._execute_write(commit)
        except GoalEvaluationConflict:
            raise
        except Exception as exc:
            raise GoalEvaluationPersistenceError from exc
        manager._state = result
        return decision
    except GoalEvaluationConflict:
        _reload_if_readable(manager, db)
        return _decision(
            manager.state.status if manager.state else None, False, None, "interrupted",
            "goal changed during evaluation", "Goal state changed during evaluation; the current goal was preserved.",
        )
    except GoalEvaluationPersistenceError:
        logger.warning("Goal evaluation could not read or commit durable state for %s", manager.session_id, exc_info=True)
        _reload_if_readable(manager, db)
        return _decision(
            manager.state.status if manager.state else None, False, None, "persistence_failed",
            "goal state could not be read or saved", "Goal state could not be read or saved; automatic continuation stopped.",
        )
