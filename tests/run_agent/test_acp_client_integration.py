"""Integration test for the acp_client runtime path through AIAgent.

Verifies that:
  - api_mode='acp_client' is accepted on AIAgent construction
  - run_conversation() takes the early-return path and never enters the
    chat completions loop
  - Projected messages from a fake ACP session land in the messages list
  - tool_iterations from the ACP session tick the skill nudge counter
  - _user_turn_count and _turns_since_memory increment pre-loop (not doubled)
  - The user message appears exactly once (no duplicate from acp_runtime)
  - The returned dict has the same shape as the chat_completions path
  - Session retirement on should_retire=True drops _acp_session
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import run_agent
from agent.transports.acp_client_session import ACPClientSession, TurnResult


@pytest.fixture
def fake_session(monkeypatch):
    """Replace ACPClientSession with a stub that returns a fixed TurnResult,
    so we can drive AIAgent without spawning a real ACP subprocess."""

    def fake_run_turn(self, user_input: str, **kwargs):
        return TurnResult(
            final_text=f"acp-echo: {user_input}",
            projected_messages=[
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "tool_1", "type": "function",
                                 "function": {"name": "bash",
                                              "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "tool_1", "content": "done"},
                {"role": "assistant", "content": f"acp-echo: {user_input}"},
            ],
            tool_iterations=1,
            interrupted=False,
            error=None,
            should_retire=False,
        )

    monkeypatch.setattr(ACPClientSession, "run_turn", fake_run_turn)
    monkeypatch.setattr(
        ACPClientSession, "ensure_started", lambda self, **kw: "acp-session-stub"
    )


def _make_acp_agent():
    """Construct an AIAgent in acp_client mode without contacting any real
    provider. Pass api_mode explicitly so the constructor accepts it directly."""
    return run_agent.AIAgent(
        api_key="stub",
        base_url="https://stub.invalid",
        provider="openai",
        api_mode="acp_client",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


class TestApiModeAccepted:
    def test_api_mode_is_acp_client(self):
        agent = _make_acp_agent()
        assert agent.api_mode == "acp_client"


class TestRunConversationAcpPath:
    def test_run_conversation_returns_acp_shape(self, fake_session):
        """Return dict matches the chat_completions shape contract."""
        agent = _make_acp_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hello there")
        assert result["final_response"] == "acp-echo: hello there"
        assert result["completed"] is True
        assert result["partial"] is False
        assert result["error"] is None
        assert result["api_calls"] == 1

    def test_projected_messages_are_spliced(self, fake_session):
        """Projected messages from TurnResult land in the returned messages list."""
        agent = _make_acp_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hello")
        msgs = result["messages"]
        # User message + 3 projected (assistant tool_call + tool + assistant text)
        assert len(msgs) >= 4
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"
        # Final assistant message with the echoed text should be in there
        final = [m for m in msgs if m.get("role") == "assistant"
                 and m.get("content") == "acp-echo: hello"]
        assert final, f"expected final assistant message in {msgs}"

    def test_nudge_counters_tick(self, fake_session):
        """The skill nudge counter accumulates tool_iterations across turns.
        _user_turn_count is incremented by run_conversation pre-loop (not by
        acp_runtime), so it must be exactly 1 after one turn."""
        agent = _make_acp_agent()
        agent._iters_since_skill = 0
        agent._user_turn_count = 0
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("first")
        assert agent._iters_since_skill == 1  # one tool_iteration in fake turn
        # _user_turn_count is incremented pre-loop, not by the ACP helper —
        # confirms we delegate counter management to the standard flow.
        assert agent._user_turn_count == 1
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("second")
        assert agent._iters_since_skill == 2
        assert agent._user_turn_count == 2

    def test_user_message_not_duplicated(self, fake_session):
        """Regression guard: the user message must appear exactly once.

        The standard run_conversation() pre-loop appends it, and acp_runtime
        must NOT append again (see acp_runtime.py NOTE comment at line 59-61).
        """
        agent = _make_acp_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("ping unique 99999")
        user_count = sum(
            1 for m in result["messages"]
            if m.get("role") == "user" and m.get("content") == "ping unique 99999"
        )
        assert user_count == 1, (
            f"user message appeared {user_count}× in {result['messages']} — "
            "acp_runtime must not re-append the user message"
        )

    def test_chat_completions_loop_is_not_entered(self, fake_session):
        """The early-return at api_mode='acp_client' must bypass the regular
        API call loop entirely. Confirmed by patching the SDK client and
        asserting chat.completions.create is never called."""
        agent = _make_acp_agent()
        with patch.object(agent, "client") as client_mock, patch.object(
            agent, "_spawn_background_review", return_value=None
        ):
            agent.run_conversation("hi")
        assert not client_mock.chat.completions.create.called, (
            "chat.completions.create was called — early-return did not fire"
        )

    def test_background_review_NOT_invoked_below_threshold(self, fake_session):
        """A single turn shouldn't trigger background review — counters
        haven't reached the nudge interval (default 10)."""
        agent = _make_acp_agent()
        agent._memory_nudge_interval = 10
        agent._skill_nudge_interval = 10
        agent._iters_since_skill = 0
        with patch.object(agent, "_spawn_background_review",
                          return_value=None) as spawn:
            agent.run_conversation("ping")
        assert not spawn.called, (
            "_spawn_background_review fired below threshold — "
            "check acp_runtime.py skill nudge gate condition"
        )


class TestErrorHandling:
    def test_session_exception_returns_partial_with_error(self, monkeypatch):
        def boom_run_turn(self, user_input, **kwargs):
            raise RuntimeError("acp subprocess died")

        monkeypatch.setattr(ACPClientSession, "ensure_started",
                            lambda self, **kw: "s1")
        monkeypatch.setattr(ACPClientSession, "run_turn", boom_run_turn)

        agent = _make_acp_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hi")
        assert result["completed"] is False
        assert result["partial"] is True
        assert "acp subprocess died" in result["error"]

    def test_interrupted_turn_marked_partial(self, monkeypatch):
        def interrupted_turn(self, user_input, **kwargs):
            return TurnResult(
                final_text="",
                projected_messages=[],
                tool_iterations=0,
                interrupted=True,
                error="user interrupted",
                should_retire=False,
            )

        monkeypatch.setattr(ACPClientSession, "ensure_started",
                            lambda self, **kw: "s1")
        monkeypatch.setattr(ACPClientSession, "run_turn", interrupted_turn)

        agent = _make_acp_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hi")
        assert result["completed"] is False
        assert result["partial"] is True
        assert result["error"] == "user interrupted"


class TestSessionRetirementOnRunAgent:
    """run_agent.py side: when run_turn returns should_retire=True, the
    AIAgent must close + null _acp_session so the next turn respawns."""

    def test_should_retire_drops_session(self, monkeypatch):
        closes = {"count": 0}

        def fake_run_turn(self, user_input, **kwargs):
            return TurnResult(
                final_text="",
                projected_messages=[],
                tool_iterations=0,
                interrupted=True,
                error="turn timed out after 600.0s",
                should_retire=True,
            )

        def fake_close(self):
            closes["count"] += 1

        monkeypatch.setattr(ACPClientSession, "ensure_started",
                            lambda self, **kw: "s1")
        monkeypatch.setattr(ACPClientSession, "run_turn", fake_run_turn)
        monkeypatch.setattr(ACPClientSession, "close", fake_close)

        agent = _make_acp_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hi")

        # The session was closed and cleared
        assert closes["count"] == 1
        assert getattr(agent, "_acp_session", "MISSING") is None
        # Partial result was still returned (caller still sees the error)
        assert result["partial"] is True
        assert result["error"] == "turn timed out after 600.0s"

    def test_normal_turn_keeps_session(self, fake_session):
        """fake_session fixture returns should_retire=False (default).
        The session must stay attached for the next turn to reuse."""
        agent = _make_acp_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("hi")
        # Session was lazily created and still attached.
        assert getattr(agent, "_acp_session", None) is not None

    def test_exception_path_also_drops_session(self, monkeypatch):
        """Even if run_turn raises (not just sets should_retire), we must
        drop the session — a thrown exception is the strongest possible
        signal the process is dead."""
        closes = {"count": 0}

        def boom_run_turn(self, user_input, **kwargs):
            raise RuntimeError("acp segfaulted")

        def fake_close(self):
            closes["count"] += 1

        monkeypatch.setattr(ACPClientSession, "ensure_started",
                            lambda self, **kw: "s1")
        monkeypatch.setattr(ACPClientSession, "run_turn", boom_run_turn)
        monkeypatch.setattr(ACPClientSession, "close", fake_close)

        agent = _make_acp_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hi")

        assert closes["count"] == 1
        assert agent._acp_session is None
        assert result["completed"] is False
        assert "acp segfaulted" in result["error"]
