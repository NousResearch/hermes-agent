"""Focused tests for the bounded ``pre_turn_finalize`` plugin continuation hook.

The hook is a generic lifecycle extension point (no provider heuristic): after the
model produces a plain-text final and every built-in stop gate accepted it, a plugin
may return ``{"action": "continue", "message": ...}`` to send the turn back to the
agent loop — at most once per turn. No real provider/API calls are made here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_finalizer import _drop_verification_continuation_scaffolding
from agent.turn_stop_gates import (
    MAX_PRE_TURN_FINALIZE_NUDGES,
    apply_stop_gates,
)
from hermes_cli.plugins import get_pre_turn_finalize_continue_message


class _StubAgent:
    """Minimal surface ``apply_stop_gates`` touches."""

    def __init__(self):
        self.session_id = "sess-1"
        self.platform = "cli"
        self.model = "test/model"
        self.provider = "test-provider"
        self._turn_file_mutation_paths = set()
        self._verification_stop_nudges = 0
        self._pre_verify_nudges = 0
        self._pre_turn_finalize_nudges = 0
        self._kanban_stop_nudges = 0
        self._session_messages = None
        self._current_turn_id = ""
        self.interim_emitted = []

    def _emit_interim_assistant_message(self, msg):
        self.interim_emitted.append(msg)

    def _interim_content_was_streamed(self, text):
        return False

    def _flush_messages_to_session_db(self, messages, history):
        return None

    def _emit_status(self, text):
        return None


def _run_gates(agent, final_response="done", **kwargs):
    final_msg = {"role": "assistant", "content": final_response}
    messages = [{"role": "user", "content": "do work"}]
    verdict = apply_stop_gates(
        agent,
        final_msg,
        final_response=final_response,
        messages=messages,
        conversation_history=[],
        pending_verification_response=None,
        pending_verification_response_previewed=False,
        finish_reason=kwargs.pop("finish_reason", "stop"),
        api_call_count=kwargs.pop("api_call_count", 4),
        turn_id=kwargs.pop("turn_id", "turn-1"),
        **kwargs,
    )
    return verdict, final_msg, messages


@pytest.fixture
def quiet_gates(monkeypatch):
    """Neutralize the other three gates; opt into pre_turn_finalize only per test."""
    monkeypatch.setattr(
        "agent.turn_stop_gates._verify_on_stop_nudge", lambda agent: None
    )
    monkeypatch.setattr(
        "agent.turn_stop_gates._pre_verify_nudge", lambda agent, fr, attempt: None
    )
    monkeypatch.setattr(
        "agent.turn_stop_gates._kanban_stop_nudge", lambda agent, messages: None
    )


class TestNoHookRegistered:
    def test_finalize_unchanged_without_hook(self, quiet_gates, monkeypatch):
        """(1) Hook absent → byte/semantics identical to the pre-hook behavior."""
        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook", lambda name: False
        )
        agent = _StubAgent()
        verdict, _final_msg, messages = _run_gates(agent)
        assert verdict.continue_turn is False
        assert verdict.final_response == "done"
        assert verdict.pending_verification_response is None
        assert [m["role"] for m in messages] == ["user"]
        assert agent._pre_turn_finalize_nudges == 0

    def test_none_result_finalizes(self, quiet_gates, monkeypatch):
        """(2) Hook returning None → normal finalize."""
        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook",
            lambda name: name == "pre_turn_finalize",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_turn_finalize_continue_message",
            lambda **kwargs: None,
        )
        agent = _StubAgent()
        verdict, _final_msg, messages = _run_gates(agent)
        assert verdict.continue_turn is False
        assert verdict.final_response == "done"
        assert [m["role"] for m in messages] == ["user"]


class TestContinuation:
    def test_valid_continue_loops_once(self, quiet_gates, monkeypatch):
        """(3,4) Valid directive on attempt 0 → loop continues, candidate kept as the
        interim row, synthetic user nudge appended after it."""
        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook",
            lambda name: name == "pre_turn_finalize",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_turn_finalize_continue_message",
            lambda **kwargs: "Keep going and finish.",
        )
        agent = _StubAgent()
        verdict, final_msg, messages = _run_gates(agent, final_response="tiny")
        assert verdict.continue_turn is True
        assert verdict.final_response is None
        # Candidate preserved only as the budget-exhaustion fallback.
        assert verdict.pending_verification_response == "tiny"
        assert final_msg["finish_reason"] == "pre_turn_finalize_continue"
        # Interim assistant row persisted + emitted, then the synthetic nudge.
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[1]["content"] == "tiny"
        assert not messages[1].get("_pre_turn_finalize_synthetic")
        assert messages[2]["content"] == "Keep going and finish."
        assert messages[2].get("_pre_turn_finalize_synthetic") is True
        assert agent.interim_emitted and agent.interim_emitted[0]["content"] == "tiny"
        assert agent._pre_turn_finalize_nudges == 1

    def test_second_attempt_cannot_continue(self, quiet_gates, monkeypatch):
        """(5) Even an always-continue plugin cannot re-continue on attempt 1."""
        calls = []

        def _helper(**kwargs):
            calls.append(kwargs)
            return "Keep going forever."

        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook",
            lambda name: name == "pre_turn_finalize",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_turn_finalize_continue_message", _helper
        )
        agent = _StubAgent()
        agent._pre_turn_finalize_nudges = 1
        verdict, _final_msg, messages = _run_gates(agent)
        assert verdict.continue_turn is False
        assert verdict.final_response == "done"
        assert calls == []
        assert [m["role"] for m in messages] == ["user"]

    def test_bound_is_one(self):
        assert MAX_PRE_TURN_FINALIZE_NUDGES == 1


class TestFailOpen:
    @pytest.mark.parametrize(
        "directive",
        [
            None,
            "just a string",
            42,
            {"action": "continue"},  # no message
            {"action": "continue", "message": ""},  # empty message
            {"action": "continue", "message": "   "},  # blank message
            {"action": "observe", "message": "Keep going."},  # unknown action
            {"decision": "approve"},  # approval shape, not a stop block
        ],
    )
    def test_invalid_directives_ignored(self, quiet_gates, monkeypatch, directive):
        """(6) Invalid/empty directives finalize instead of looping."""
        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook",
            lambda name: name == "pre_turn_finalize",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [directive],
        )
        agent = _StubAgent()
        verdict, _final_msg, messages = _run_gates(agent)
        assert verdict.continue_turn is False
        assert verdict.final_response == "done"
        assert [m["role"] for m in messages] == ["user"]

    def test_exception_finalizes(self, quiet_gates, monkeypatch):
        """(7) A raising hook never breaks the agent; the turn finalizes."""
        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook",
            lambda name: name == "pre_turn_finalize",
        )

        def _boom(**kwargs):
            raise RuntimeError("plugin exploded")

        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook", _boom
        )
        agent = _StubAgent()
        verdict, _final_msg, _messages = _run_gates(agent)
        assert verdict.continue_turn is False
        assert verdict.final_response == "done"

    def test_first_valid_wins_and_claude_shape_accepted(self, monkeypatch):
        """Aggregation mirrors pre_verify: first non-empty wins; the Claude-Code Stop
        block shape is accepted for convention parity."""
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [
                None,
                {"action": "observe"},
                {"decision": "block", "reason": "Finish the job first."},
                {"action": "continue", "message": "Second wins never."},
            ],
        )
        assert (
            get_pre_turn_finalize_continue_message() == "Finish the job first."
        )

    def test_message_is_stripped(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [
                {"action": "continue", "message": "  padded  "}
            ],
        )
        assert get_pre_turn_finalize_continue_message() == "padded"


class TestOrdering:
    def test_verify_on_stop_runs_first(self, monkeypatch):
        """(8) A firing verify-on-stop gate wins; pre_turn_finalize is not consulted."""
        monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")
        finalize_calls = []
        monkeypatch.setattr(
            "agent.verification_stop.build_verify_on_stop_nudge",
            lambda **kwargs: "verify it",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_turn_finalize_continue_message",
            lambda **kwargs: finalize_calls.append(kwargs) or "should not fire",
        )
        agent = _StubAgent()
        agent._turn_file_mutation_paths = {"changed.py"}
        verdict, final_msg, _messages = _run_gates(agent)
        assert verdict.continue_turn is True
        assert final_msg["finish_reason"] == "verification_required"
        assert finalize_calls == []
        assert agent._pre_turn_finalize_nudges == 0

    def test_pre_verify_runs_first(self, quiet_gates, monkeypatch):
        """(8) A firing pre_verify gate wins; pre_turn_finalize is not consulted."""
        monkeypatch.setattr(
            "agent.turn_stop_gates._pre_verify_nudge",
            lambda agent, fr, attempt: "run the tests",
        )
        finalize_calls = []
        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook", lambda name: True
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_turn_finalize_continue_message",
            lambda **kwargs: finalize_calls.append(kwargs) or "should not fire",
        )
        agent = _StubAgent()
        verdict, final_msg, _messages = _run_gates(agent)
        assert verdict.continue_turn is True
        assert final_msg["finish_reason"] == "verify_hook_continue"
        assert finalize_calls == []
        assert agent._pre_turn_finalize_nudges == 0

    def test_kanban_guard_runs_first(self, quiet_gates, monkeypatch):
        """(8) A firing kanban guard wins; pre_turn_finalize is not consulted."""
        monkeypatch.setattr(
            "agent.turn_stop_gates._kanban_stop_nudge",
            lambda agent, messages: "call kanban_complete",
        )
        finalize_calls = []
        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook", lambda name: True
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_turn_finalize_continue_message",
            lambda **kwargs: finalize_calls.append(kwargs) or "should not fire",
        )
        agent = _StubAgent()
        verdict, final_msg, _messages = _run_gates(agent)
        assert verdict.continue_turn is True
        assert final_msg["finish_reason"] == "kanban_terminal_required"
        assert finalize_calls == []


class TestHistoryHygiene:
    def test_synthetic_nudge_stripped_candidate_kept(self, quiet_gates, monkeypatch):
        """(9) The synthetic nudge is removed from durable history; the real
        assistant candidate is never deleted."""
        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook",
            lambda name: name == "pre_turn_finalize",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_turn_finalize_continue_message",
            lambda **kwargs: "Keep going.",
        )
        agent = _StubAgent()
        verdict, _final_msg, messages = _run_gates(agent, final_response="tiny")
        assert verdict.continue_turn is True
        _drop_verification_continuation_scaffolding(messages)
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["content"] == "tiny"


class TestPayload:
    def test_documented_fields_forwarded(self, quiet_gates, monkeypatch):
        """(10) The documented payload fields reach the hook intact."""
        seen = {}

        def _capture(hook_name, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook",
            lambda name: name == "pre_turn_finalize",
        )
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _capture)
        agent = _StubAgent()
        agent.session_id = "sess-9"
        agent._current_turn_id = "fallback-turn"
        verdict, _final_msg, _messages = _run_gates(
            agent,
            final_response="almost done",
            finish_reason="stop",
            api_call_count=7,
            turn_id="turn-9",
        )
        assert verdict.continue_turn is False
        assert seen["session_id"] == "sess-9"
        assert seen["turn_id"] == "turn-9"
        assert seen["platform"] == "cli"
        assert seen["model"] == "test/model"
        assert seen["provider"] == "test-provider"
        assert seen["final_response"] == "almost done"
        assert seen["finish_reason"] == "stop"
        assert seen["api_call_count"] == 7
        assert seen["attempt"] == 0
        assert "conversation_history" not in seen

    def test_attempt_increments(self, quiet_gates, monkeypatch):
        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return "Keep going."

        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook",
            lambda name: name == "pre_turn_finalize",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_turn_finalize_continue_message", _capture
        )
        agent = _StubAgent()
        _run_gates(agent)
        assert seen["attempt"] == 0


def _e2e_agent(tmp_path, monkeypatch, **kwargs):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id="pre-turn-finalize-e2e",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=kwargs.get("max_iterations", 3),
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    return instance


def _e2e_response(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


class TestEndToEnd:
    def test_one_continuation_then_final(self, tmp_path, monkeypatch):
        """One continue → second answer becomes the final; the synthetic nudge does
        not leak into durable history."""
        agent = _e2e_agent(tmp_path, monkeypatch, max_iterations=3)
        answers = iter([_e2e_response("frag"), _e2e_response("full final answer")])
        agent._interruptible_api_call = lambda _kwargs: next(answers)
        agent._handle_max_iterations = MagicMock(return_value="summary")
        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook",
            lambda name: name == "pre_turn_finalize",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_turn_finalize_continue_message",
            MagicMock(side_effect=["finish the work", None]),
        )
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            result = agent.run_conversation("do the task")
        assert result["final_response"] == "full final answer"
        roles = [m["role"] for m in result["messages"]]
        assert roles[0] == "user"
        assert "assistant" in roles
        assert not any(
            isinstance(m, dict) and m.get("_pre_turn_finalize_synthetic")
            for m in result["messages"]
        )
        # The first candidate survives as an interim assistant row.
        assert any(
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and m.get("content") == "frag"
            for m in result["messages"]
        )

    def test_always_continue_hooks_fires_only_once(self, tmp_path, monkeypatch):
        """An always-continue plugin gets exactly one continuation per turn."""
        agent = _e2e_agent(tmp_path, monkeypatch, max_iterations=4)
        answers = iter(
            [_e2e_response("frag"), _e2e_response("second"), _e2e_response("third")]
        )
        agent._interruptible_api_call = lambda _kwargs: next(answers)
        agent._handle_max_iterations = MagicMock(return_value="summary")
        hook = MagicMock(return_value="keep going")
        monkeypatch.setattr(
            "hermes_cli.lifecycle.has_hook",
            lambda name: name == "pre_turn_finalize",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_turn_finalize_continue_message", hook
        )
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            result = agent.run_conversation("do the task")
        assert hook.call_count == 1
        assert result["final_response"] == "second"
