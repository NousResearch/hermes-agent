"""Tests for CLI goal-continuation interrupt handling.

Covers:
- Ctrl+C during a /goal turn auto-pauses the goal (no more continuations).
- Empty/whitespace-only responses skip outcome application (no phantom continuations).
- A clean response applies the primary model's structured outcome.

These tests exercise ``_maybe_continue_goal_after_turn`` directly on a
minimal ``HermesCLI`` stub (pattern used elsewhere in tests/cli).
"""

from __future__ import annotations

import queue
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes stay hermetic."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal module's DB cache so it re-resolves HERMES_HOME each test.
    from hermes_cli import goals
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _make_cli_with_goal(session_id: str, goal_text: str = "build a thing"):
    """Build a minimal HermesCLI stub with an active goal wired in."""
    from cli import HermesCLI
    from hermes_cli.goals import GoalManager

    cli = HermesCLI.__new__(HermesCLI)
    # State the hook + helpers touch directly.
    cli._pending_input = queue.Queue()
    cli._last_turn_interrupted = False
    cli.conversation_history = []
    # `_get_goal_manager()` reads `self.session_id` directly, not
    # `self.agent.session_id`. Match the production lookup.
    cli.session_id = session_id
    cli.agent = MagicMock()
    cli.agent.session_id = session_id

    mgr = GoalManager(session_id=session_id, default_max_turns=5)
    mgr.set(goal_text)
    turn_id = f"test-turn-{uuid.uuid4().hex}"
    generation_id = mgr.begin_model_turn(turn_id)
    cli._last_goal_turn_id = turn_id
    cli._last_goal_generation_id = generation_id
    cli._last_goal_turn_failed = False
    cli._goal_test_authority = {
        "originating_turn_id": turn_id,
        "goal_generation_id": generation_id,
    }
    cli.agent._current_turn_id = turn_id
    cli.agent._current_goal_generation_id = generation_id
    cli._goal_manager = mgr
    return cli, mgr


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


class TestInterruptAutoPause:
    def test_interrupted_turn_pauses_goal_and_skips_continuation(self, hermes_home):
        """Ctrl+C mid-turn must auto-pause the goal, not queue another round."""
        sid = f"sid-interrupt-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        # Simulate an interrupted turn with a partial assistant reply.
        cli._last_turn_interrupted = True
        cli.conversation_history = [
            {"role": "user", "content": "kickoff"},
            {"role": "assistant", "content": "starting work..."},
        ]

        cli._maybe_continue_goal_after_turn()

        # Pending input must NOT contain a continuation prompt.
        assert cli._pending_input.empty(), (
            "Interrupted turn should not enqueue a continuation prompt"
        )

        # Goal should be paused, not active.
        state = mgr.state
        assert state is not None
        assert state.status == "paused"
        assert "interrupt" in (state.paused_reason or "").lower()

    def test_interrupted_turn_is_resumable(self, hermes_home):
        """After auto-pause from Ctrl+C, /goal resume puts it back to active."""
        sid = f"sid-resume-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = True
        cli.conversation_history = [
            {"role": "assistant", "content": "partial"},
        ]
        cli._maybe_continue_goal_after_turn()
        assert mgr.state.status == "paused"

        mgr.resume()
        assert mgr.state.status == "active"


class TestEmptyResponseSkip:
    def test_empty_response_does_not_apply_outcome(self, hermes_home):
        """Whitespace-only replies skip lifecycle evaluation."""
        sid = f"sid-empty-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = False
        cli.conversation_history = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "   \n\n   "},
        ]

        cli._maybe_continue_goal_after_turn()

        # No continuation queued; goal still active (neither paused nor done).
        assert cli._pending_input.empty()
        assert mgr.state.status == "active"
        assert mgr.state.active_model_turn_id is None

    def test_no_assistant_message_skipped(self, hermes_home):
        """Conversation with zero assistant replies skips lifecycle evaluation."""
        sid = f"sid-noassistant-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = False
        cli.conversation_history = [
            {"role": "user", "content": "go"},
        ]

        cli._maybe_continue_goal_after_turn()

        assert cli._pending_input.empty()
        assert mgr.state.status == "active"
        assert mgr.state.active_model_turn_id is None

    def test_failed_turn_revokes_pending_outcome(self, hermes_home):
        sid = f"sid-failed-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        assert mgr.record_model_outcome(
            "complete", "must not leak", **cli._goal_test_authority
        )
        cli._last_goal_turn_failed = True
        cli.conversation_history = [
            {"role": "assistant", "content": "provider error"},
        ]

        cli._maybe_continue_goal_after_turn()

        assert cli._pending_input.empty()
        assert mgr.state.status == "active"
        assert mgr.state.pending_model_outcome is None
        assert mgr.state.active_model_turn_id is None


class TestHealthyTurnStillRuns:
    def test_clean_response_enqueues_model_authored_continuation(
        self, hermes_home,
    ):
        """Sanity check: the hook still works in the happy path."""
        sid = f"sid-healthy-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = False
        cli.conversation_history = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "did some work, more to do"},
        ]

        mgr.record_model_outcome(
            "continue", "needs more steps", **cli._goal_test_authority
        )
        cli._maybe_continue_goal_after_turn()

        # Continuation prompt must be queued.
        assert not cli._pending_input.empty()
        queued = cli._pending_input.get_nowait()
        assert "Continuing toward your standing goal" in queued
        assert mgr.state.status == "active"

    def test_clean_response_marks_done_when_primary_model_records_completion(self, hermes_home):
        sid = f"sid-done-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = False
        cli.conversation_history = [
            {"role": "assistant", "content": "all finished, here's the result"},
        ]

        mgr.record_model_outcome(
            "complete", "goal satisfied", **cli._goal_test_authority
        )
        cli._maybe_continue_goal_after_turn()

        assert cli._pending_input.empty()
        assert mgr.state.status == "done"

    def test_queued_user_turn_preempts_enqueue_but_consumes_completion(
        self, hermes_home
    ):
        sid = f"sid-preempt-complete-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli.conversation_history = [
            {"role": "assistant", "content": "verified and complete"},
        ]
        cli._pending_input.put("new user instruction")
        assert mgr.record_model_outcome(
            "complete", "verified", **cli._goal_test_authority
        )

        cli._maybe_continue_goal_after_turn()

        assert mgr.state.status == "done"
        assert cli._pending_input.qsize() == 1
        assert cli._pending_input.get_nowait() == "new user instruction"

    def test_queued_user_turn_suppresses_only_automatic_continuation(
        self, hermes_home
    ):
        sid = f"sid-preempt-continue-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli.conversation_history = [
            {"role": "assistant", "content": "more work remains"},
        ]
        cli._pending_input.put("change the next step")
        assert mgr.record_model_outcome(
            "continue", "more work", **cli._goal_test_authority
        )

        cli._maybe_continue_goal_after_turn()

        assert mgr.state.status == "active"
        assert mgr.state.turns_used == 1
        assert mgr.state.active_model_turn_id is None
        assert cli._pending_input.qsize() == 1
        assert cli._pending_input.get_nowait() == "change the next step"


class TestPrimaryModelDraft:
    def test_plain_goal_queues_primary_model_kickoff(self, hermes_home):
        sid = f"sid-kickoff-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid, goal_text="old goal")

        cli._handle_goal_command("/goal finish the migration")

        assert mgr.state.goal == "finish the migration"
        queued = cli._pending_input.get_nowait()
        assert "Begin working toward your standing goal" in queued
        assert "goal_outcome" in queued

    def test_goal_draft_queues_primary_model_workspace_turn(self, hermes_home):
        sid = f"sid-draft-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid, goal_text="old goal")

        cli._handle_goal_draft("migrate auth to JWT")

        assert mgr.state.goal == "migrate auth to JWT"
        assert not mgr.state.has_contract()
        queued = cli._pending_input.get_nowait()
        assert "Author your standing-goal workspace" in queued
        assert "migrate auth to JWT" in queued
        assert "goal_contract through the todo tool" in queued
        assert "record goal_outcome" in queued


class TestInterruptFlagLifecycle:
    def test_chat_resets_flag_at_entry(self, hermes_home):
        """chat() must reset _last_turn_interrupted at the top of each turn.

        This guards against stale flag state: if turn N was interrupted and
        turn N+1 runs clean, the hook must not see True from N.
        """
        # We can't run chat() end-to-end here, but we can assert the reset
        # is the first thing after the secret-capture registration by
        # inspecting the source shape.
        from cli import HermesCLI
        import inspect

        src = inspect.getsource(HermesCLI.chat)
        # Look for an explicit reset near the top of chat().
        head = src.split("if not self._ensure_runtime_credentials", 1)[0]
        assert "self._last_turn_interrupted = False" in head, (
            "chat() must reset _last_turn_interrupted before run_conversation "
            "runs — otherwise a prior turn's interrupt state leaks into the "
            "next turn's goal hook decision."
        )
