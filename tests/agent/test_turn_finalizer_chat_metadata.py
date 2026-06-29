from __future__ import annotations

from dataclasses import dataclass

from agent.turn_finalizer import finalize_turn


@dataclass
class _Budget:
    remaining: int = 10
    used: int = 1
    max_total: int = 20


class _Compressor:
    last_prompt_tokens = 123


class _FakeAgent:
    max_iterations = 10
    iteration_budget = _Budget()
    model = "auxiliary-model"
    provider = "auxiliary-provider"
    base_url = "https://aux.example"
    _last_chat_model = "chat-model"
    _last_chat_provider = "chat-provider"
    _last_chat_base_url = "https://chat.example"
    session_input_tokens = 1
    session_output_tokens = 2
    session_cache_read_tokens = 0
    session_cache_write_tokens = 0
    session_reasoning_tokens = 0
    session_prompt_tokens = 1
    session_completion_tokens = 2
    session_total_tokens = 3
    session_estimated_cost_usd = 0.0
    session_cost_status = "ok"
    session_cost_source = "test"
    session_id = "session-1"
    context_compressor = _Compressor()
    _tool_guardrail_halt_decision = None
    _response_was_previewed = False
    _interrupt_message = None
    _stream_callback = None
    _skill_nudge_interval = 0
    _iters_since_skill = 0
    valid_tool_names = set()
    platform = "telegram"

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *_args, **_kwargs):
        pass

    def _persist_session(self, *_args, **_kwargs):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def _sync_external_memory_for_turn(self, *_args, **_kwargs):
        pass

    def _spawn_background_review(self, *_args, **_kwargs):
        pass

    def clear_interrupt(self):
        pass


def test_finalize_turn_returns_chat_metadata_not_mutated_aux_model():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "done"},
    ]
    result = finalize_turn(
        _FakeAgent(),
        final_response="done",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id=None,
        turn_id="turn-1",
        user_message="hi",
        original_user_message="hi",
        _should_review_memory=False,
        _turn_exit_reason="text_response",
    )

    assert result["model"] == "auxiliary-model"
    assert result["provider"] == "auxiliary-provider"
    assert result["base_url"] == "https://aux.example"
    assert result["chat_model"] == "chat-model"
    assert result["chat_provider"] == "chat-provider"
    assert result["chat_base_url"] == "https://chat.example"
