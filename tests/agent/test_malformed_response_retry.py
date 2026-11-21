"""Gemini ``MALFORMED_RESPONSE`` is a broken generation, not a refusal.

CLIProxyAPI reports it as ``finish_reason: content_filter`` with
``native_finish_reason: malformed_response``; the refusal path never retried it and told the
user "safety refusal". It now ends the call normally and takes a bounded corrected retry, the
way Antigravity's own agent treats a bad generation.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from agent import turn_empty_response as ter
from agent.gemini_native_adapter import translate_gemini_response, translate_stream_event
from agent.message_sanitization import is_malformed_response, native_finish_reason
from agent.transports import get_transport


def _completion(finish_reason, native=None, content=None):
    choice = {"index": 0, "finish_reason": finish_reason,
              "message": {"role": "assistant", "content": content}}
    if native:
        choice["native_finish_reason"] = native
    return ChatCompletion.model_validate({
        "id": "c1", "object": "chat.completion", "created": 0, "model": "gemini", "choices": [choice]})


def _transport():
    import agent.transports.chat_completions  # registers the transport
    return get_transport("chat_completions")


def test_cpa_malformed_content_filter_is_not_a_refusal():
    normalized = _transport().normalize_response(_completion("content_filter", "malformed_response"))
    assert normalized.finish_reason == "stop"


def test_malformed_after_partial_text_continues_instead_of_finishing():
    """Text streamed before the break is not a finished answer: the continuation path keeps it."""
    normalized = _transport().normalize_response(
        _completion("content_filter", "malformed_response", content="Let me read the config file."))
    assert normalized.finish_reason == "length"


def test_plain_content_filter_stays_a_refusal():
    assert _transport().normalize_response(_completion("content_filter")).finish_reason == "content_filter"


def test_stream_assembly_keeps_the_native_reason():
    from agent import chat_completion_helpers as cch

    chunk = ChatCompletionChunk.model_validate({
        "id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gemini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter",
                     "native_finish_reason": "malformed_response"}]})
    native = native_finish_reason(chunk.choices[0])
    assert native == "malformed_response"

    call = cch._StreamingCall.__new__(cch._StreamingCall)
    call.agent = MagicMock()
    resp = call._finish_chat_stream(None, "assistant", [], [], {}, "content_filter", "gemini", None,
                                    flush_pending=lambda: None, native_finish=native)
    assert is_malformed_response(resp)
    assert _transport().normalize_response(resp).finish_reason == "stop"


def test_native_gemini_adapter_keeps_the_raw_reason():
    for reason in ("MALFORMED_RESPONSE", "MALFORMED_FUNCTION_CALL"):
        resp = translate_gemini_response({"candidates": [{"finishReason": reason}]}, "gemini")
        assert is_malformed_response(resp)
        chunks = translate_stream_event({"candidates": [{"finishReason": reason}]}, "gemini", {})
        assert is_malformed_response(chunks[-1])
    ok = translate_gemini_response({"candidates": [{"finishReason": "STOP"}]}, "gemini")
    assert not is_malformed_response(ok)


def _agent():
    a = MagicMock()
    a._current_streamed_assistant_text = ""
    a._has_content_after_think_block.return_value = False
    a._strip_think_blocks.side_effect = lambda t: t or ""
    a._malformed_response_retries = 0
    a._last_content_with_tools = None  # no housekeeping reply to reuse
    a._empty_response_retry_hint = None
    a._fallback_chain = []
    a._try_activate_fallback.return_value = False
    return a


def _recover(agent, response, messages=None):
    msg = SimpleNamespace(reasoning=None, reasoning_content=None, reasoning_details=None)
    return ter.recover_empty_response(
        agent, msg, response, "stop", final_response="",
        messages=messages or [{"role": "user", "content": "hi"}],
        api_messages=[], conversation_history=[], active_system_prompt="sys", api_call_count=5,
        turn_exit_reason=None, preflight_compression_blocked=False,
    )


def test_malformed_reply_gets_bounded_corrected_retries():
    agent = _agent()
    response = SimpleNamespace(choices=[SimpleNamespace(native_finish_reason="malformed_response")])
    with patch.object(ter, "_retry_empty") as generic_retry, \
            patch.object(ter, "_terminal_empty", return_value="(empty)") as terminal:
        for remaining in (2, 1, 0):
            assert _recover(agent, response).action == "continue"
            tail = [{"role": "tool", "content": "result"}]
            ter.add_empty_response_retry_hint(agent, tail)
            assert "MALFORMED_RESPONSE" in tail[-1]["content"]
            assert f"Retries remaining: {remaining}." in tail[-1]["content"]
        verdict = _recover(agent, response)
    assert verdict.action == "break"
    terminal.assert_called_once()
    generic_retry.assert_not_called()  # no extra full-context resends after the corrected ones


def test_machinery_turn_still_ends_quietly():
    """A notification turn answered with nothing stays silent: no paid corrected retries."""
    agent = _agent()
    agent._build_assistant_message.return_value = {"role": "assistant"}
    response = SimpleNamespace(choices=[SimpleNamespace(native_finish_reason="malformed_response")])
    notice = [{"role": "user", "content": "done", "display_kind": "process_complete"}]
    assert _recover(agent, response, notice).action == "break"
    assert agent._malformed_response_retries == 0
    assert agent._empty_response_retry_hint is None
