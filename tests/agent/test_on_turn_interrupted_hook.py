"""Regression tests for the interrupted-turn content hook (#108808).

``post_llm_call`` is gated on a final, non-interrupted response, and
``on_session_end`` carries no message body, so before this hook no plugin could
observe *what* an interrupted turn said — only that one happened. A memory /
logging plugin could therefore never durably capture the user message (or the
partial assistant text already streamed) when a turn was cut short by
``/stop``, a mid-turn interrupt, or a new message.

``on_turn_interrupted`` fires from ``finalize_turn`` whenever the turn ends
interrupted, over the ordinary plugin hook bus.
"""

from __future__ import annotations

from agent.turn_finalizer import finalize_turn
from hermes_cli import plugins
from hermes_cli.plugins import VALID_HOOKS, PluginContext, PluginManifest

USER_MESSAGE = "summarize the release notes"
PARTIAL_ASSISTANT_TEXT = "Here is the summary so far: the release adds"


class _StubBudget:
    used = 1
    max_total = 90
    remaining = 89


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface ``finalize_turn`` reads from."""

    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.platform = "cli"
        self.quiet_mode = True
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._persist_disabled = False
        self._current_streamed_assistant_text = ""
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"
        self.persisted_messages = None

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = [dict(m) for m in messages]

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        # Mirrors production: the interrupt message stops being readable here, so the
        # hook payload must be captured before this point.
        self._interrupt_message = None

    def _sync_external_memory_for_turn(self, **k):
        pass


def _interrupted_messages():
    """A turn interrupted before any assistant row was committed."""
    return [{"role": "user", "content": USER_MESSAGE}]


def _finalize(agent, messages, *, interrupted, final_response=None):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=interrupted,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message=USER_MESSAGE,
        original_user_message=USER_MESSAGE,
        _should_review_memory=False,
        _turn_exit_reason="interrupted_by_user",
    )


def _register_hook(name: str):
    """Register a capturing callback through the real plugin context."""
    captured = []

    def _cb(**kwargs):
        captured.append(kwargs)

    manager = plugins.get_plugin_manager()
    ctx = PluginContext(manifest=PluginManifest(name="capture_plugin", source="user"), manager=manager)
    ctx.register_hook(name, _cb)
    return captured


def test_on_turn_interrupted_is_a_valid_hook():
    assert "on_turn_interrupted" in VALID_HOOKS


def test_registering_on_turn_interrupted_is_not_an_unknown_hook(caplog):
    """A plugin must be able to register the hook without the unknown-hook warning."""
    with caplog.at_level("WARNING"):
        _register_hook("on_turn_interrupted")

    assert not [
        record for record in caplog.records if "unknown hook" in record.getMessage()
    ]


def test_interrupted_turn_dispatches_user_message_and_partial_response():
    captured = _register_hook("on_turn_interrupted")
    agent = _StubAgent()
    agent._interrupt_requested = True
    agent._interrupt_message = "user interrupt"
    agent._current_streamed_assistant_text = PARTIAL_ASSISTANT_TEXT

    _finalize(agent, _interrupted_messages(), interrupted=True, final_response=None)

    assert len(captured) == 1
    payload = captured[0]
    assert payload["user_message"] == USER_MESSAGE
    assert payload["assistant_response"] == PARTIAL_ASSISTANT_TEXT
    assert payload["session_id"] == "sess-1"
    assert payload["task_id"] == "task-1"
    assert payload["turn_id"] == "turn-1"
    assert payload["turn_exit_reason"] == "interrupted_by_user"
    assert payload["interrupt_message"] == "user interrupt"
    assert payload["model"] == "stub/model"
    assert payload["platform"] == "cli"
    assert [m["role"] for m in payload["conversation_history"]] == ["user"]
    assert payload["conversation_history"][0]["content"] == USER_MESSAGE


def test_interrupted_turn_without_streamed_text_reports_empty_response():
    captured = _register_hook("on_turn_interrupted")
    agent = _StubAgent()

    _finalize(agent, _interrupted_messages(), interrupted=True, final_response=None)

    assert len(captured) == 1
    assert captured[0]["assistant_response"] == ""
    assert captured[0]["user_message"] == USER_MESSAGE


def test_clean_turn_keeps_existing_hook_semantics():
    """A completed turn fires post_llm_call only — the new hook stays interrupt-only."""
    interrupted_calls = _register_hook("on_turn_interrupted")
    post_llm_calls = _register_hook("post_llm_call")
    agent = _StubAgent()

    _finalize(agent, _interrupted_messages(), interrupted=False, final_response="All done.")

    assert interrupted_calls == []
    assert len(post_llm_calls) == 1
    assert post_llm_calls[0]["assistant_response"] == "All done."


def test_interrupted_turn_still_skips_post_llm_call():
    """Pre-existing contract: content hooks that require a final response skip interrupts."""
    interrupted_calls = _register_hook("on_turn_interrupted")
    post_llm_calls = _register_hook("post_llm_call")
    agent = _StubAgent()

    _finalize(agent, _interrupted_messages(), interrupted=True, final_response=None)

    assert post_llm_calls == []
    assert len(interrupted_calls) == 1


def test_detached_fork_turn_publishes_no_interrupted_hook():
    """Detached forks must not publish turns under the parent's session id (#15165 rule)."""
    captured = _register_hook("on_turn_interrupted")
    agent = _StubAgent()
    agent._persist_disabled = True

    _finalize(agent, _interrupted_messages(), interrupted=True, final_response=None)

    assert captured == []
