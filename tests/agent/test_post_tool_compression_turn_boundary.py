from types import SimpleNamespace

from agent.turn_preflight import compress_after_tool_results


def test_post_tool_compression_reanchors_the_active_user_boundary(monkeypatch):
    compressed = [
        {"role": "user", "content": "compressed history"},
        {"role": "user", "content": "current ask"},
        {"role": "assistant", "tool_calls": [{"id": "2"}]},
        {"role": "tool", "content": "fresh result", "tool_call_id": "2"},
    ]

    class Compressor:
        last_prompt_tokens = 100
        threshold_tokens = 50

        @staticmethod
        def should_compress(_tokens):
            return True

    agent = SimpleNamespace(
        context_compressor=Compressor(),
        compression_enabled=True,
        _clear_context_overflow_warn=lambda: None,
        _safe_print=lambda *_args: None,
        _compress_context=lambda *_args, **_kwargs: (compressed, "system"),
        _persist_user_message_idx=4,
    )
    monkeypatch.setattr(
        "agent.turn_preflight.conversation_history_after_compression",
        lambda _agent, _messages, _history: [],
    )
    monkeypatch.setattr(
        "agent.conversation_loop._should_skip_model_call_for_reference_handoff",
        lambda _messages, _user_message: False,
    )

    verdict = compress_after_tool_results(
        agent,
        messages=[{"role": "user", "content": "current ask"}],
        system_message="system",
        user_message="current ask",
        active_system_prompt="system",
        conversation_history=[],
        compression_attempts=0,
        max_compression_attempts=1,
        effective_task_id="task",
        final_response="",
        turn_exit_reason=None,
        current_turn_user_idx=0,
    )

    assert verdict.messages is compressed
    assert verdict.current_turn_user_idx == 1
    assert agent._persist_user_message_idx == 1
