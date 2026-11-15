"""A completed Telegram TTS tool result needs no caption-only second model round."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import agent.turn_tool_round as round_module
from agent.turn_tool_round import _single_tts_media_response, run_tool_round


def _call(name="text_to_speech", call_id="call-tts"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _message(name="text_to_speech", call_id="call-tts"):
    return SimpleNamespace(content="", tool_calls=[_call(name, call_id)])


def _tool_result(call_id="call-tts", **overrides):
    payload = {
        "success": True,
        "media_tag": "[[audio_as_voice]]\nMEDIA:/tmp/reply.ogg",
        **overrides,
    }
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload)}


def test_single_tts_media_response_requires_telegram_success_and_one_call():
    agent = SimpleNamespace(platform="telegram")
    messages = [_tool_result()]
    result = _single_tts_media_response(agent, _message(), messages)
    assert result is not None and result.endswith("MEDIA:/tmp/reply.ogg")

    assert _single_tts_media_response(SimpleNamespace(platform="desktop"), _message(), messages) is None
    assert _single_tts_media_response(agent, _message(name="read_file"), messages) is None
    assert _single_tts_media_response(agent, _message(), [_tool_result(success=False)]) is None
    assert _single_tts_media_response(agent, _message(), [_tool_result(media_tag="")]) is None

    mixed = _message()
    mixed.tool_calls.append(_call(name="read_file", call_id="call-read"))
    assert _single_tts_media_response(agent, mixed, messages) is None


def test_run_tool_round_breaks_with_native_tts_media(monkeypatch):
    agent = SimpleNamespace(
        platform="telegram",
        quiet_mode=True,
        verbose_logging=False,
        log_prefix="",
        valid_tool_names={"text_to_speech"},
        stream_delta_callback=None,
        _incremental_persistence_failed=False,
        _tool_guardrail_halt_decision=None,
        iteration_budget=SimpleNamespace(remaining=10, refund=lambda: None),
        _session_messages=[],
    )
    agent._cap_delegate_task_calls = lambda calls: calls
    agent._deduplicate_tool_calls = lambda calls: calls
    agent._flush_messages_to_session_db = lambda messages, history: True
    agent._emit_interim_assistant_message = lambda message: None
    agent._touch_activity = lambda message: None

    def _execute(assistant_message, messages, task_id, api_count):
        messages.append(_tool_result())

    agent._execute_tool_calls = _execute
    monkeypatch.setattr(round_module, "validate_tool_calls", lambda *a, **k: SimpleNamespace(
        action="ok", mixed_invalid_batch=False,
    ))
    monkeypatch.setattr(round_module, "stage_tool_call_message", lambda *a, **k: (
        {"role": "assistant", "content": "", "tool_calls": []}, False,
    ))
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_harness.finish_tool_round",
        lambda agent: None,
    )
    monkeypatch.setattr(round_module, "compress_after_tool_results", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the redundant post-TTS model/compression path must not run"),
    ))
    # The round imports one error-copy helper from the full conversation loop.
    # Isolate that unrelated bootstrap; this test exercises the tool-round contract.
    conversation_loop = ModuleType("agent.conversation_loop")
    conversation_loop._invalid_tool_name_error_content = lambda *a: "invalid tool"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent.conversation_loop", conversation_loop)

    verdict = run_tool_round(
        agent,
        assistant_message=_message(),
        finish_reason="tool_calls",
        messages=[],
        conversation_history=[],
        api_call_count=1,
        effective_task_id="task",
        user_message="Sprachnachricht bitte",
        system_message="",
        active_system_prompt="",
        compression_attempts=0,
        max_compression_attempts=1,
        final_response=None,
        failed=False,
        _turn_exit_reason="unknown",
        truncated_tool_call_retries=0,
        current_turn_user_idx=0,
    )

    assert verdict.action == "break"
    assert verdict.final_response == "[[audio_as_voice]]\nMEDIA:/tmp/reply.ogg"
    assert verdict._turn_exit_reason == "tts_media_ready"
    assert verdict.failed is False

    # The ordinary finalizer closes the durable tool-result tail with the media
    # response; the normal gateway parser then delivers exactly one voice file.
    from agent.turn_finalizer import _close_transcript_tail
    from gateway.platforms.base import BasePlatformAdapter
    _close_transcript_tail(agent, verdict.messages, verdict.final_response, False, False)
    assert verdict.messages[-1]["role"] == "assistant"
    assert verdict.messages[-1]["content"] == verdict.final_response
    media, remaining_text = BasePlatformAdapter.extract_media(verdict.final_response)
    assert media == [("/tmp/reply.ogg", True)]
    assert not remaining_text.strip()
