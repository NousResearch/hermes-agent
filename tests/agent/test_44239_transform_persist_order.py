"""Regression tests for #44239 — transform_llm_output vs. persistence order.

The ``transform_llm_output`` hook fires in ``finalize_turn``. Before the
fix, ``_persist_session`` ran *before* the hook and the transformed text
was never written back into the assistant message, so the user saw the
transformed response while ``result["messages"]``, the JSON log, and the
SQLite session DB all kept the raw model output — which was then replayed
on resume / next turn.

These tests drive ``finalize_turn`` directly with a stub agent and a
patched ``hermes_cli.plugins.invoke_hook`` and assert INVARIANTS:

* the transformed text is synced into the turn's last assistant message
  before persistence (delivered text == persisted text);
* persistence still runs after ``transform_llm_output`` and before
  ``post_llm_call`` (observability plugins may read the session store
  expecting the turn to be there);
* an untransformed response persists the raw model output unchanged;
* the sync never crosses the turn boundary or rewrites a tool-call /
  non-text assistant message.
"""

import copy
from types import SimpleNamespace

import hermes_cli.plugins as plugins_mod
from agent.turn_finalizer import (
    _sync_final_response_to_last_assistant,
    finalize_turn,
)


class _StubAgent:
    """Bare-minimum agent surface that finalize_turn touches on the happy
    path (final_response present, not interrupted, budget remaining, no
    footer/explainer, no skill/memory review)."""

    def __init__(self):
        self.max_iterations = 100
        self.iteration_budget = SimpleNamespace(remaining=99, used=1, max_total=100)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.platform = "cli"
        self.session_id = "sess-44239"
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "ok"
        self.session_cost_source = "none"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self._tool_guardrail_halt_decision = None
        self._turn_failed_file_mutations = None
        self._interrupt_message = None
        self._stream_callback = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = set()
        self.events = []          # ordered record of persist + hook firings
        self.persisted_messages = None  # deep copy taken at persist time

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("budget-exhaustion path not expected in these tests")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, messages, user_message, completed):
        pass

    def _cleanup_task_resources(self, task_id):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.events.append("persist")
        self.persisted_messages = copy.deepcopy(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **kwargs):
        pass


def _run_finalize(monkeypatch, transform_result, messages):
    agent = _StubAgent()

    def fake_invoke_hook(hook_name, **kwargs):
        agent.events.append(hook_name)
        if hook_name == "transform_llm_output":
            return [transform_result] if transform_result is not None else []
        return []

    monkeypatch.setattr(plugins_mod, "invoke_hook", fake_invoke_hook)

    result = finalize_turn(
        agent,
        final_response=messages[-1]["content"],
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="hi",
        original_user_message="hi",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )
    return agent, result


def test_transformed_response_is_persisted(monkeypatch):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "raw model output"},
    ]
    agent, result = _run_finalize(
        monkeypatch, "[RENDERED] raw model output", messages,
    )

    assert result["final_response"] == "[RENDERED] raw model output"
    assert result["response_transformed"] is True
    # In-memory history handed back to the caller matches what was shown.
    assert result["messages"][-1]["content"] == "[RENDERED] raw model output"
    # ...and so does what hit the session store.
    assert agent.persisted_messages[-1]["content"] == "[RENDERED] raw model output"


def test_hook_persist_ordering(monkeypatch):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "raw model output"},
    ]
    agent, _ = _run_finalize(monkeypatch, "[RENDERED] x", messages)

    # Transform must precede persist (so the transform is durable);
    # persist must precede post_llm_call (observability plugins may read
    # the session store inside that hook).
    assert agent.events.index("transform_llm_output") < agent.events.index("persist")
    assert agent.events.index("persist") < agent.events.index("post_llm_call")


def test_untransformed_response_persists_raw(monkeypatch):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "raw model output"},
    ]
    agent, result = _run_finalize(monkeypatch, None, messages)

    assert result["response_transformed"] is False
    assert agent.persisted_messages[-1]["content"] == "raw model output"


def test_sync_skips_tool_call_assistant_message():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}],
        },
    ]
    assert _sync_final_response_to_last_assistant(messages, "new") is False
    assert messages[1]["content"] == ""


def test_sync_does_not_cross_turn_boundary():
    messages = [
        {"role": "assistant", "content": "prior turn answer"},
        {"role": "user", "content": "this turn"},
        {"role": "tool", "content": "{}", "tool_call_id": "c1"},
    ]
    assert _sync_final_response_to_last_assistant(messages, "new") is False
    assert messages[0]["content"] == "prior turn answer"


def test_sync_skips_non_text_content():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "multimodal"}]},
    ]
    assert _sync_final_response_to_last_assistant(messages, "new") is False


def test_sync_updates_last_assistant_text_message():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "raw"},
        {"role": "tool", "content": "{}", "tool_call_id": "c1"},
    ]
    assert _sync_final_response_to_last_assistant(messages, "new") is True
    assert messages[1]["content"] == "new"
