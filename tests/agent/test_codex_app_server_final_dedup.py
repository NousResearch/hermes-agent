"""The codex_app_server turn must not deliver its final answer twice (#74248).

The live app-server event bridge routes every completed ``agentMessage``
through ``_emit_interim_assistant_message``, and that item can BE the turn's
final answer. ``conversation_loop`` communicates "already shown" to the
gateway via the ``response_previewed`` key on the turn result, but the
app-server early-return path never set it — so the gateway defaulted it to
``False`` and sent the same text again. On Discord that surfaces as two
replies about a second apart: one unreferenced (bridge), one reply-referenced
(normal final path).
"""

from __future__ import annotations

import pytest

from agent.codex_runtime import _final_text_was_streamed


class _Agent:
    """Minimal stand-in exposing the real prefix-match semantics."""

    def __init__(self, streamed: str = ""):
        self._current_streamed_assistant_text = streamed

    def _interim_content_was_streamed(self, content: str) -> bool:
        streamed = self._current_streamed_assistant_text or ""
        return bool(streamed) and (content or "").startswith(streamed)


class TestFinalTextWasStreamed:
    def test_final_answer_already_streamed_is_marked_previewed(self):
        """The #74248 shape: the bridge already delivered the final text."""
        assert _final_text_was_streamed(_Agent("The answer is 42."), "The answer is 42.") is True

    def test_streamed_prefix_of_final_counts(self):
        """Final may be the streamed text plus a trailing delta."""
        assert _final_text_was_streamed(_Agent("The answer"), "The answer is 42.") is True

    def test_distinct_commentary_does_not_suppress_the_final(self):
        """Mid-turn commentary must not mark an unrelated final as delivered."""
        assert _final_text_was_streamed(_Agent("Working on it..."), "The answer is 42.") is False

    def test_nothing_streamed_is_not_previewed(self):
        assert _final_text_was_streamed(_Agent(""), "The answer is 42.") is False

    @pytest.mark.parametrize("final_text", ["", None])
    def test_empty_final_is_not_previewed(self, final_text):
        assert _final_text_was_streamed(_Agent("anything"), final_text) is False

    def test_agent_without_the_probe_fails_open(self):
        """Fail toward a benign duplicate, never a suppressed answer."""
        assert _final_text_was_streamed(object(), "The answer is 42.") is False

    def test_probe_errors_fail_open(self):
        class _Boom:
            def _interim_content_was_streamed(self, content):
                raise RuntimeError("boom")

        assert _final_text_was_streamed(_Boom(), "The answer is 42.") is False


# ---------- turn-level: the returned result contract ----------


from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.codex_runtime import run_codex_app_server_turn


def _make_turn(**overrides):
    turn = SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-1",
        turn_id="turn-1",
        projected_messages=[{"role": "assistant", "content": "The answer is 42."}],
        tool_iterations=0,
        final_text="The answer is 42.",
        should_retire=False,
    )
    for key, value in overrides.items():
        setattr(turn, key, value)
    return turn


def _make_agent(turn, streamed=""):
    agent = MagicMock()
    # Pre-seed the session so run_codex_app_server_turn skips the spawn block.
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.return_value = turn
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = None
    agent._session_db_created = True
    agent.session_id = "sess-codex"
    agent._interrupt_requested = False
    # A MagicMock probe returns a truthy MagicMock — pin the real semantics.
    agent._interim_content_was_streamed = (
        lambda content: bool(streamed) and (content or "").startswith(streamed)
    )
    return agent


def _run(agent):
    return run_codex_app_server_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}],
        effective_task_id="task-1",
    )


class TestTurnResultContract:
    def test_streamed_final_is_previewed_on_the_turn_result(self):
        """The #74248 shape, observed at the run_codex_app_server_turn
        contract the gateway consumes — not just the helper."""
        turn = _make_turn()
        result = _run(_make_agent(turn, streamed="The answer is 42."))

        assert result["completed"] is True
        assert result["response_previewed"] is True
        assert result["final_response"] == "The answer is 42."

    def test_unrelated_commentary_is_not_previewed(self):
        turn = _make_turn()
        result = _run(_make_agent(turn, streamed="Working on it..."))

        assert result["completed"] is True
        assert result["response_previewed"] is False

    def test_aborted_turn_is_never_previewed(self):
        """Codex substitutes sentinel text (<turn_aborted>) on an
        interrupted/errored turn. Even when the streamed probe would match,
        the partial result must not be marked previewed — the gateway would
        suppress delivering it at all."""
        turn = _make_turn(
            interrupted=True,
            error="turn aborted",
            final_text="<turn_aborted>",
            projected_messages=[],
        )
        # Probe that would match anything — the completed gate must win.
        agent = _make_agent(turn)
        agent._interim_content_was_streamed = lambda content: True

        result = _run(agent)

        assert result["completed"] is False
        assert result["partial"] is True
        assert result["response_previewed"] is False
