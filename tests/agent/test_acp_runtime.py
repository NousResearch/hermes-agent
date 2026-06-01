"""Tests for agent.acp_runtime — run_acp_client_turn().

Tests cover the happy path, crash/retire policy, skill nudge increment,
memory sync, and background review trigger.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.acp_runtime import run_acp_client_turn
from agent.transports.acp_client_session import TurnResult


# ---------------------------------------------------------------------------
# Helpers — mock agent
# ---------------------------------------------------------------------------


def _make_agent(
    *,
    api_mode: str = "acp_client",
    acp_command: str = "fake-acp",
    acp_args: list = None,
    session_cwd: str = "/tmp",
    _iters_since_skill: int = 0,
    _skill_nudge_interval: int = 10,
    valid_tool_names: set = None,
    has_stream_delta: bool = False,
) -> SimpleNamespace:
    agent = SimpleNamespace(
        api_mode=api_mode,
        acp_command=acp_command,
        acp_args=list(acp_args or []),
        session_cwd=session_cwd,
        _acp_session=None,
        _iters_since_skill=_iters_since_skill,
        _skill_nudge_interval=_skill_nudge_interval,
        valid_tool_names=valid_tool_names or set(),
        _fire_stream_delta=MagicMock() if has_stream_delta else None,
        _sync_external_memory_for_turn=MagicMock(),
        _spawn_background_review=MagicMock(),
    )
    return agent


def _mock_session(turn_result: TurnResult) -> MagicMock:
    """Return a mock ACPClientSession that returns the given TurnResult."""
    mock = MagicMock()
    mock.run_turn.return_value = turn_result
    mock.close.return_value = None
    return mock


def _inject_session(agent, session_mock: MagicMock) -> None:
    """Inject a pre-built session into the agent."""
    agent._acp_session = session_mock


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_returns_final_response_from_turn(self):
        """Happy path: final_text from TurnResult becomes final_response."""
        agent = _make_agent()
        turn = TurnResult(final_text="hello from ACP", projected_messages=[
            {"role": "assistant", "content": "hello from ACP"}
        ])
        _inject_session(agent, _mock_session(turn))
        messages = [{"role": "user", "content": "hi"}]

        result = run_acp_client_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages,
            effective_task_id="t1",
        )

        assert result["final_response"] == "hello from ACP"
        assert result["completed"] is True
        assert result["partial"] is False
        assert result["error"] is None
        assert result["api_calls"] == 1

    def test_projected_messages_spliced_into_messages(self):
        """Projected messages from TurnResult are appended to messages."""
        agent = _make_agent()
        turn = TurnResult(
            final_text="Hi there",
            projected_messages=[{"role": "assistant", "content": "Hi there"}],
        )
        _inject_session(agent, _mock_session(turn))
        messages = [{"role": "user", "content": "hello"}]

        run_acp_client_turn(
            agent,
            user_message="hello",
            original_user_message="hello",
            messages=messages,
            effective_task_id="t1",
        )

        assert len(messages) == 2
        assert messages[-1] == {"role": "assistant", "content": "Hi there"}

    def test_lazy_session_created_on_first_call(self):
        """A new ACPClientSession is created when _acp_session is None."""
        agent = _make_agent()
        assert agent._acp_session is None

        mock_session = _mock_session(TurnResult(final_text="ok"))
        with patch(
            "agent.transports.acp_client_session.ACPClientSession.__init__",
            return_value=None,
        ), patch(
            "agent.transports.acp_client_session.ACPClientSession.run_turn",
            return_value=TurnResult(final_text="ok"),
        ), patch(
            "agent.transports.acp_client_session.ACPClientSession.close",
        ):
            result = run_acp_client_turn(
                agent,
                user_message="test",
                original_user_message="test",
                messages=[],
                effective_task_id="t1",
            )

        assert result["final_response"] == "ok"
        # Session was created and stored
        assert agent._acp_session is not None

    def test_session_reused_across_calls(self):
        """Existing _acp_session is reused, no new session created."""
        agent = _make_agent()
        session_mock = _mock_session(TurnResult(final_text="ok"))
        _inject_session(agent, session_mock)

        # Run two turns — the pre-injected session should be reused
        run_acp_client_turn(
            agent,
            user_message="turn1",
            original_user_message="turn1",
            messages=[],
            effective_task_id="t1",
        )
        run_acp_client_turn(
            agent,
            user_message="turn2",
            original_user_message="turn2",
            messages=[],
            effective_task_id="t2",
        )
        assert session_mock.run_turn.call_count == 2


# ---------------------------------------------------------------------------
# Tests: crash / retire policy
# ---------------------------------------------------------------------------


class TestRetirePolicy:
    def test_should_retire_true_closes_and_nils_session(self):
        """When turn.should_retire is True, session is closed and set to None."""
        agent = _make_agent()
        turn = TurnResult(
            final_text="", should_retire=True, error="agent crashed"
        )
        session_mock = _mock_session(turn)
        _inject_session(agent, session_mock)

        result = run_acp_client_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[],
            effective_task_id="t1",
        )

        session_mock.close.assert_called_once()
        assert agent._acp_session is None

    def test_crash_exception_nils_session_and_returns_error(self):
        """Uncaught exception from run_turn() sets session to None and returns error."""
        agent = _make_agent()
        session_mock = MagicMock()
        session_mock.run_turn.side_effect = RuntimeError("subprocess died")
        _inject_session(agent, session_mock)

        result = run_acp_client_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[],
            effective_task_id="t1",
        )

        assert result["completed"] is False
        assert result["partial"] is True
        assert "ACP client turn failed" in result["final_response"]
        assert agent._acp_session is None

    def test_no_retire_keeps_session_alive(self):
        """When should_retire=False, session is kept for reuse."""
        agent = _make_agent()
        turn = TurnResult(final_text="ok", should_retire=False)
        session_mock = _mock_session(turn)
        _inject_session(agent, session_mock)

        run_acp_client_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[],
            effective_task_id="t1",
        )

        session_mock.close.assert_not_called()
        assert agent._acp_session is session_mock


# ---------------------------------------------------------------------------
# Tests: skill nudge counter
# ---------------------------------------------------------------------------


class TestSkillNudge:
    def test_tool_iterations_added_to_iters_since_skill(self):
        """turn.tool_iterations is added to agent._iters_since_skill."""
        agent = _make_agent(_iters_since_skill=3, _skill_nudge_interval=20)
        turn = TurnResult(final_text="ok", tool_iterations=4)
        _inject_session(agent, _mock_session(turn))

        run_acp_client_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[],
            effective_task_id="t1",
        )

        assert agent._iters_since_skill == 7  # 3 + 4

    def test_skill_nudge_resets_counter(self):
        """When _skill_nudge_interval is reached, _iters_since_skill resets."""
        agent = _make_agent(
            _iters_since_skill=8,
            _skill_nudge_interval=10,
            valid_tool_names={"skill_manage"},
        )
        turn = TurnResult(final_text="ok", tool_iterations=3)
        _inject_session(agent, _mock_session(turn))

        run_acp_client_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[],
            effective_task_id="t1",
            should_review_memory=False,
        )

        # 8 + 3 = 11 >= 10 → nudge triggered → counter reset
        assert agent._iters_since_skill == 0


# ---------------------------------------------------------------------------
# Tests: memory sync
# ---------------------------------------------------------------------------


class TestMemorySync:
    def test_memory_sync_called_on_successful_turn(self):
        """_sync_external_memory_for_turn called on non-interrupted, non-error turn."""
        agent = _make_agent()
        turn = TurnResult(final_text="ok", interrupted=False, error=None)
        _inject_session(agent, _mock_session(turn))

        run_acp_client_turn(
            agent,
            user_message="q",
            original_user_message="q",
            messages=[],
            effective_task_id="t1",
        )

        agent._sync_external_memory_for_turn.assert_called_once_with(
            original_user_message="q",
            final_response="ok",
            interrupted=False,
        )

    def test_memory_sync_skipped_on_interrupted_turn(self):
        """Memory sync skipped when turn is interrupted."""
        agent = _make_agent()
        turn = TurnResult(final_text="", interrupted=True)
        _inject_session(agent, _mock_session(turn))

        run_acp_client_turn(
            agent,
            user_message="q",
            original_user_message="q",
            messages=[],
            effective_task_id="t1",
        )

        agent._sync_external_memory_for_turn.assert_not_called()

    def test_memory_sync_skipped_on_error_turn(self):
        """Memory sync skipped when turn has an error."""
        agent = _make_agent()
        turn = TurnResult(final_text="", error="something broke")
        _inject_session(agent, _mock_session(turn))

        run_acp_client_turn(
            agent,
            user_message="q",
            original_user_message="q",
            messages=[],
            effective_task_id="t1",
        )

        agent._sync_external_memory_for_turn.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: background review
# ---------------------------------------------------------------------------


class TestBackgroundReview:
    def test_background_review_spawned_when_triggered(self):
        """_spawn_background_review called when should_review_memory=True."""
        agent = _make_agent()
        turn = TurnResult(final_text="good answer", interrupted=False, error=None)
        _inject_session(agent, _mock_session(turn))

        run_acp_client_turn(
            agent,
            user_message="q",
            original_user_message="q",
            messages=[],
            effective_task_id="t1",
            should_review_memory=True,
        )

        agent._spawn_background_review.assert_called_once()
        call_kwargs = agent._spawn_background_review.call_args[1]
        assert call_kwargs["review_memory"] is True

    def test_background_review_skipped_when_no_final_text(self):
        """_spawn_background_review not called when final_text is empty."""
        agent = _make_agent()
        turn = TurnResult(final_text="", interrupted=False, error=None)
        _inject_session(agent, _mock_session(turn))

        run_acp_client_turn(
            agent,
            user_message="q",
            original_user_message="q",
            messages=[],
            effective_task_id="t1",
            should_review_memory=True,
        )

        agent._spawn_background_review.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: cwd resolution — HERMES_ACP_SESSION_CWD env var
# ---------------------------------------------------------------------------


class TestCwdResolution:
    """run_acp_client_turn must pass the correct cwd to run_turn.

    Priority (highest first):
      1. agent.session_cwd (explicitly set on the agent)
      2. HERMES_ACP_SESSION_CWD env var
      3. os.getcwd() fallback

    This is the mechanism that lets the janet_test gateway route each ACP
    session into its sandbox directory (where CLAUDE.md and
    .claude/settings.local.json live) without adding a config key to the
    provider resolver chain.
    """

    def _capture_cwd_from_run_turn(self, agent) -> str:
        """Run one turn and return the cwd that was passed to run_turn."""
        captured: list[str] = []
        session_mock = MagicMock()

        def _fake_run_turn(user_input, *, cwd=None, **kwargs):
            captured.append(cwd or "")
            return TurnResult(final_text="ok")

        session_mock.run_turn.side_effect = _fake_run_turn
        _inject_session(agent, session_mock)

        run_acp_client_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[],
            effective_task_id="t1",
        )
        return captured[0] if captured else ""

    def test_session_cwd_attribute_wins_over_env(self, monkeypatch):
        """agent.session_cwd takes priority over HERMES_ACP_SESSION_CWD."""
        monkeypatch.setenv("HERMES_ACP_SESSION_CWD", "/env/sandbox")
        agent = _make_agent(session_cwd="/agent/sandbox")

        cwd = self._capture_cwd_from_run_turn(agent)
        assert cwd == "/agent/sandbox"

    def test_env_var_used_when_session_cwd_absent(self, monkeypatch):
        """HERMES_ACP_SESSION_CWD is used when agent has no session_cwd."""
        monkeypatch.setenv("HERMES_ACP_SESSION_CWD", "/env/sandbox")
        agent = _make_agent()
        # Remove session_cwd so it's absent (not just None)
        del agent.session_cwd

        cwd = self._capture_cwd_from_run_turn(agent)
        assert cwd == "/env/sandbox"

    def test_env_var_used_when_session_cwd_is_none(self, monkeypatch):
        """HERMES_ACP_SESSION_CWD is used when agent.session_cwd is None."""
        monkeypatch.setenv("HERMES_ACP_SESSION_CWD", "/env/sandbox")
        agent = _make_agent(session_cwd=None)

        cwd = self._capture_cwd_from_run_turn(agent)
        assert cwd == "/env/sandbox"

    def test_env_var_empty_falls_back_to_getcwd(self, monkeypatch):
        """Empty HERMES_ACP_SESSION_CWD falls through to os.getcwd()."""
        monkeypatch.setenv("HERMES_ACP_SESSION_CWD", "")
        agent = _make_agent(session_cwd=None)

        import os
        cwd = self._capture_cwd_from_run_turn(agent)
        assert cwd == os.getcwd()

    def test_no_env_var_falls_back_to_getcwd(self, monkeypatch):
        """Absent HERMES_ACP_SESSION_CWD falls through to os.getcwd()."""
        monkeypatch.delenv("HERMES_ACP_SESSION_CWD", raising=False)
        agent = _make_agent(session_cwd=None)

        import os
        cwd = self._capture_cwd_from_run_turn(agent)
        assert cwd == os.getcwd()
