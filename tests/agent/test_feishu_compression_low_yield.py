from types import SimpleNamespace

from agent.conversation_compression import (
    LOW_YIELD_COMPRESSION_MESSAGE,
    compress_context,
)


class FakeCompressor:
    compression_count = 0
    _last_compress_aborted = False
    _last_summary_error = None
    _last_aux_model_failure_model = None
    _last_aux_model_failure_error = None

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
        return list(messages)


class FakeTodoStore:
    def format_for_injection(self):
        return ""


def test_feishu_low_yield_compression_returns_original_messages():
    warnings = []
    messages = [
        {"role": "user", "content": "查一下"},
        {"role": "assistant", "content": "需要工具"},
    ]
    agent = SimpleNamespace(
        platform="feishu",
        session_id="sid",
        model="gemini-3-flash",
        compression_enabled=True,
        context_compressor=FakeCompressor(),
        _compression_feasibility_checked=True,
        _memory_manager=None,
        _session_db=None,
        _todo_store=FakeTodoStore(),
        _cached_system_prompt="system",
        _last_compression_summary_warning=None,
        _last_aux_fallback_warning_key=None,
        _emit_status=lambda msg: None,
        _emit_warning=warnings.append,
        _build_system_prompt=lambda system_message: system_message or "system",
        _invalidate_system_prompt=lambda: None,
        commit_memory_session=lambda _messages: None,
    )

    returned, system_prompt = compress_context(agent, messages, "system", approx_tokens=80_000)

    assert returned is messages
    assert system_prompt == "system"
    assert warnings == [LOW_YIELD_COMPRESSION_MESSAGE]
    assert agent._last_compression_low_yield["before_messages"] == 2
    assert agent._last_compression_low_yield["after_messages"] == 2

