"""Pre-compression memory-provider context wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.conversation_compression import compress_context


def test_pre_compress_provider_context_reaches_compressor():
    """MemoryProvider.on_pre_compress() output is fed into compression."""
    messages = [
        {"role": "user", "content": "summarize this long session"},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "tail"},
    ]

    memory_manager = MagicMock()
    memory_manager.on_pre_compress.return_value = "provider extracted: project=alpha"

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor._last_compress_aborted = False
    compressor._last_summary_error = None
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor.compression_count = 1

    agent = SimpleNamespace(
        api_mode="chat_completions",
        _memory_manager=memory_manager,
        context_compressor=compressor,
        _compression_feasibility_checked=True,
        _session_db=None,
        session_id="s1",
        compression_in_place=False,
        model="test/model",
        platform="cli",
        tools=None,
        _todo_store=SimpleNamespace(format_for_injection=lambda: ""),
        _cached_system_prompt=None,
        _last_compaction_in_place=False,
        _emit_status=lambda *_args, **_kwargs: None,
        _emit_warning=lambda *_args, **_kwargs: None,
        _invalidate_system_prompt=lambda: None,
        _build_system_prompt=lambda system_message: f"rebuilt: {system_message}",
        event_callback=None,
    )

    compressed, new_prompt = compress_context(
        agent,
        messages,
        "system prompt",
        approx_tokens=120_000,
    )

    assert compressed == compressor.compress.return_value
    assert new_prompt == "rebuilt: system prompt"
    memory_manager.on_pre_compress.assert_called_once_with(messages)
    assert compressor.compress.call_args.kwargs["pre_compress_context"] == (
        "provider extracted: project=alpha"
    )
