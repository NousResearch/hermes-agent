from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from agent.turn_finalizer import finalize_turn


class DummyAgent:
    def __init__(self):
        self.max_iterations = 10
        self.iteration_budget = SimpleNamespace(remaining=9, used=1, max_total=10)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.platform = "test"
        self.session_id = "session-1"
        self.valid_tool_names = set()
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._interrupt_message = None
        self._stream_callback = object()
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)

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
        self.session_cost_source = "test"

        self.persisted_messages = None
        self.memory_sync_messages = None
        self.memory_sync_response = None

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _handle_max_iterations(self, messages, api_call_count):  # pragma: no cover
        raise AssertionError("unexpected max-iteration summary")

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        self.scaffolding_dropped = True

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = deepcopy(messages)
        self.persisted_history = deepcopy(conversation_history)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def _sync_external_memory_for_turn(self, *, original_user_message, final_response, interrupted, messages):
        self.memory_sync_response = final_response
        self.memory_sync_messages = deepcopy(messages)

    def _spawn_background_review(self, *_args, **_kwargs):
        pass

    def clear_interrupt(self):
        self.cleared_interrupt = True


def _run_finalizer(agent, messages, *, final_response="raw answer"):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=messages,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="hello",
        original_user_message="hello",
        _should_review_memory=False,
        _turn_exit_reason="text_response(stop)",
    )


def test_transform_llm_output_persists_final_assistant_message(monkeypatch):
    def fake_invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            assert kwargs["response_text"] == "raw answer"
            return ["[rendered] raw answer"]
        if name == "post_llm_call":
            assert kwargs["assistant_response"] == "[rendered] raw answer"
            assert kwargs["conversation_history"][-1]["content"] == "[rendered] raw answer"
            return []
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
    agent = DummyAgent()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "raw answer"},
    ]

    result = _run_finalizer(agent, messages)

    assert result["final_response"] == "[rendered] raw answer"
    assert result["messages"][-1]["content"] == "[rendered] raw answer"
    assert agent.persisted_messages is not None
    assert agent.memory_sync_messages is not None
    assert agent.persisted_messages[-1]["content"] == "[rendered] raw answer"
    assert agent.memory_sync_messages[-1]["content"] == "[rendered] raw answer"
    assert agent.memory_sync_response == "[rendered] raw answer"


def test_post_llm_call_response_override_persists_final_assistant_message(monkeypatch):
    def fake_invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            return []
        if name == "post_llm_call":
            assert kwargs["assistant_response"] == "raw answer"
            return [{"response": "patched response"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
    agent = DummyAgent()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "raw answer"},
    ]

    result = _run_finalizer(agent, messages)

    assert result["final_response"] == "patched response"
    assert result["messages"][-1]["content"] == "patched response"
    assert agent.persisted_messages is not None
    assert agent.memory_sync_messages is not None
    assert agent.persisted_messages[-1]["content"] == "patched response"
    assert agent.memory_sync_messages[-1]["content"] == "patched response"
    assert agent.memory_sync_response == "patched response"
