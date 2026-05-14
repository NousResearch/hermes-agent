"""Tests for /goal --confirm mode (pending_confirmation status).

Covers GoalState, GoalManager, and the confirm flow.
"""
import json
import os
import tempfile
import time

import pytest

# ---------------------------------------------------------------------------
# Ensure the package is importable from the checkout
# ---------------------------------------------------------------------------
import sys

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hermes_cli.goals import GoalState, GoalManager, DEFAULT_MAX_TURNS, save_goal, load_goal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_session() -> str:
    """Return a unique session id that won't collide with other tests."""
    return f"test_confirm_{time.time_ns()}"


# ---------------------------------------------------------------------------
# GoalState
# ---------------------------------------------------------------------------

class TestGoalStateConfirmMode:
    def test_default_confirm_mode_is_false(self):
        s = GoalState(goal="test")
        assert s.confirm_mode is False

    def test_confirm_mode_can_be_set(self):
        s = GoalState(goal="test", confirm_mode=True)
        assert s.confirm_mode is True

    def test_to_json_includes_confirm_mode(self):
        s = GoalState(goal="test", confirm_mode=True)
        data = json.loads(s.to_json())
        assert data["confirm_mode"] is True

    def test_from_json_with_confirm_mode(self):
        raw = json.dumps({
            "goal": "test",
            "status": "pending_confirmation",
            "turns_used": 0,
            "max_turns": 20,
            "created_at": 0.0,
            "last_turn_at": 0.0,
            "confirm_mode": True,
        })
        s = GoalState.from_json(raw)
        assert s.confirm_mode is True
        assert s.status == "pending_confirmation"

    def test_from_json_backward_compat_no_confirm_mode(self):
        """Old data without confirm_mode field should default to False."""
        raw = json.dumps({
            "goal": "test",
            "status": "active",
            "turns_used": 0,
            "max_turns": 20,
            "created_at": 0.0,
            "last_turn_at": 0.0,
        })
        s = GoalState.from_json(raw)
        assert s.confirm_mode is False


# ---------------------------------------------------------------------------
# GoalManager.set() with confirm
# ---------------------------------------------------------------------------

class TestGoalManagerSetConfirm:
    def test_set_without_confirm_is_active(self):
        mgr = GoalManager(session_id=_fresh_session())
        state = mgr.set("do something")
        assert state.status == "active"
        assert state.confirm_mode is False

    def test_set_with_confirm_is_pending_confirmation(self):
        mgr = GoalManager(session_id=_fresh_session())
        state = mgr.set("do something", confirm=True)
        assert state.status == "pending_confirmation"
        assert state.confirm_mode is True


# ---------------------------------------------------------------------------
# GoalManager.confirm()
# ---------------------------------------------------------------------------

class TestGoalManagerConfirm:
    def test_confirm_transitions_to_active(self):
        mgr = GoalManager(session_id=_fresh_session())
        mgr.set("do something", confirm=True)
        state = mgr.confirm()
        assert state is not None
        assert state.status == "active"

    def test_confirm_on_active_goal_returns_none(self):
        mgr = GoalManager(session_id=_fresh_session())
        mgr.set("do something")
        result = mgr.confirm()
        assert result is None

    def test_confirm_on_no_goal_returns_none(self):
        mgr = GoalManager(session_id=_fresh_session())
        result = mgr.confirm()
        assert result is None

    def test_confirm_is_idempotent(self):
        """Second confirm() after first one succeeds should return None."""
        mgr = GoalManager(session_id=_fresh_session())
        mgr.set("do something", confirm=True)
        mgr.confirm()
        result = mgr.confirm()
        assert result is None


# ---------------------------------------------------------------------------
# is_active() / has_goal()
# ---------------------------------------------------------------------------

class TestGoalManagerStatusChecks:
    def test_is_active_false_for_pending_confirmation(self):
        mgr = GoalManager(session_id=_fresh_session())
        mgr.set("do something", confirm=True)
        assert mgr.is_active() is False

    def test_is_active_true_after_confirm(self):
        mgr = GoalManager(session_id=_fresh_session())
        mgr.set("do something", confirm=True)
        mgr.confirm()
        assert mgr.is_active() is True

    def test_has_goal_true_for_pending_confirmation(self):
        mgr = GoalManager(session_id=_fresh_session())
        mgr.set("do something", confirm=True)
        assert mgr.has_goal() is True

    def test_has_goal_true_for_active(self):
        mgr = GoalManager(session_id=_fresh_session())
        mgr.set("do something")
        assert mgr.has_goal() is True


# ---------------------------------------------------------------------------
# status_line()
# ---------------------------------------------------------------------------

class TestGoalManagerStatusLine:
    def test_status_line_pending_confirmation(self):
        mgr = GoalManager(session_id=_fresh_session())
        mgr.set("do something", confirm=True)
        line = mgr.status_line()
        assert "⏳" in line
        assert "awaiting confirmation" in line

    def test_status_line_active(self):
        mgr = GoalManager(session_id=_fresh_session())
        mgr.set("do something")
        line = mgr.status_line()
        assert "⊙" in line
        assert "active" in line


# ---------------------------------------------------------------------------
# resume() for pending_confirmation
# ---------------------------------------------------------------------------

class TestGoalManagerResume:
    def test_resume_pending_confirmation_is_confirm(self):
        mgr = GoalManager(session_id=_fresh_session())
        mgr.set("do something", confirm=True)
        state = mgr.resume()
        assert state is not None
        assert state.status == "active"


# ---------------------------------------------------------------------------
# evaluate_after_turn() skips pending_confirmation
# ---------------------------------------------------------------------------

class TestGoalManagerEvaluate:
    def test_evaluate_skips_pending_confirmation(self):
        mgr = GoalManager(session_id=_fresh_session())
        mgr.set("do something", confirm=True)
        decision = mgr.evaluate_after_turn("some response")
        assert decision["should_continue"] is False


# ---------------------------------------------------------------------------
# Persistence roundtrip
# ---------------------------------------------------------------------------

class TestGoalPersistence:
    def test_roundtrip_pending_confirmation(self):
        sid = _fresh_session()
        mgr = GoalManager(session_id=sid)
        mgr.set("do something", confirm=True)

        # Reload from disk
        loaded = load_goal(sid)
        assert loaded is not None
        assert loaded.status == "pending_confirmation"
        assert loaded.confirm_mode is True

    def test_roundtrip_after_confirm(self):
        sid = _fresh_session()
        mgr = GoalManager(session_id=sid)
        mgr.set("do something", confirm=True)
        mgr.confirm()

        # Reload from disk
        loaded = load_goal(sid)
        assert loaded is not None
        assert loaded.status == "active"
        assert loaded.confirm_mode is True


def test_slash_command_input_should_not_confirm_pending_goal():
    """Regression: slash commands such as /goal clear must not confirm a pending goal."""
    # Mirrors the process_loop guard before _check_goal_confirmation.
    from cli import _looks_like_slash_command

    assert _looks_like_slash_command("/goal clear") is True
    assert _looks_like_slash_command("确认开始") is False
