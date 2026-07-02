"""Verify that external memory sync receives the pre-transform response.

When a `transform_llm_output` plugin appends display-only content (citations,
disclaimers), the external memory provider must still receive the raw LLM
output — matching what ``_persist_session`` and the in-context messages list
already store.  (#57282)
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_finalizer import finalize_turn


class _FakeAgent:
    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self.sync_calls: list[dict] = []

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_a, **_kw):
        pass

    def _safe_print(self, *_a, **_kw):
        pass

    def _save_trajectory(self, *_a, **_kw):
        pass

    def _cleanup_task_resources(self, *_a, **_kw):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **kwargs):
        self.sync_calls.append(kwargs)


_BASE_KWARGS = dict(
    effective_task_id="task-test",
    turn_id="turn-test",
    user_message="What is the answer?",
    _should_review_memory=False,
    _turn_exit_reason="completed",
)


def test_memory_sync_receives_pre_transform_response(monkeypatch):
    """External memory sync must get the raw LLM response, not plugin-appended content."""
    raw_response = "Here is the answer."
    appended_suffix = "\n\n[Source: example.com]"

    # Mock transform_llm_output to append display-only content
    def _fake_invoke_hook(hook_name, **kwargs):
        if hook_name == "transform_llm_output":
            return [raw_response + appended_suffix]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_invoke_hook)

    agent = _FakeAgent()
    messages = [
        {"role": "user", "content": "What is the answer?"},
    ]

    result = finalize_turn(
        agent,
        final_response=raw_response,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        original_user_message="What is the answer?",
        conversation_history=[],
        **_BASE_KWARGS,
    )

    # The result returned to the caller should have the transformed response
    assert result["final_response"] == raw_response + appended_suffix
    assert result["response_transformed"] is True

    # But the memory sync should have received the PRE-transform response
    assert len(agent.sync_calls) == 1
    assert agent.sync_calls[0]["final_response"] == raw_response


def test_memory_sync_receives_unchanged_response_when_no_transform(monkeypatch):
    """When no transform hook fires, memory sync gets the original response."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])

    agent = _FakeAgent()
    messages = [
        {"role": "user", "content": "Hello"},
    ]

    result = finalize_turn(
        agent,
        final_response="Hi there!",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        original_user_message="Hello",
        conversation_history=[],
        **_BASE_KWARGS,
    )

    assert result["final_response"] == "Hi there!"
    assert result["response_transformed"] is False
    assert len(agent.sync_calls) == 1
    assert agent.sync_calls[0]["final_response"] == "Hi there!"


def test_memory_sync_passes_interrupted_flag(monkeypatch):
    """Interrupted turns pass interrupted=True to memory sync (which skips sync internally)."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])

    agent = _FakeAgent()
    messages = [
        {"role": "user", "content": "Hello"},
    ]

    finalize_turn(
        agent,
        final_response="Partial...",
        api_call_count=1,
        interrupted=True,
        failed=False,
        messages=messages,
        original_user_message="Hello",
        conversation_history=[],
        **_BASE_KWARGS,
    )

    # The call IS made, but interrupted=True causes _sync_external_memory_for_turn
    # to return early without contacting the memory provider
    assert len(agent.sync_calls) == 1
    assert agent.sync_calls[0]["interrupted"] is True
    # Pre-transform response is still passed (even though it won't be used)
    assert agent.sync_calls[0]["final_response"] == "Partial..."
