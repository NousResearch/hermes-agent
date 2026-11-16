"""Production Bedrock transport/history replay regressions."""

import sys
from types import ModuleType

import pytest


class _FakeAgent:
    verbose_logging = False
    reasoning_callback = None
    stream_delta_callback = None
    _stream_callback = None

    @staticmethod
    def _extract_reasoning(message):
        return getattr(message, "reasoning_content", None)

    @staticmethod
    def _strip_think_blocks(text):
        return text

    @staticmethod
    def _needs_thinking_reasoning_pad():
        return False

    @staticmethod
    def _split_responses_tool_id(value):
        return value, None

    @staticmethod
    def _deterministic_call_id(name, args, index):
        return f"call-{index}"

    @staticmethod
    def _derive_responses_function_call_id(value, response_item_id=None):
        return value


def _raw_response():
    return {
        "output": {"message": {"role": "assistant", "content": [
            {"reasoningContent": {"redactedContent": b"r1"}},
            {"toolUse": {"toolUseId": "t1", "name": "one", "input": {"n": 1}}},
            {"reasoningContent": {"redactedContent": b"r2"}},
            {"toolUse": {"toolUseId": "t2", "name": "two", "input": {"n": 2}}},
        ]}},
        "stopReason": "tool_use",
    }


def _stream_response():
    return {"stream": [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
            "reasoningContent": {"redactedContent": b"r1"},
        }}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockStart": {"contentBlockIndex": 1, "start": {
            "toolUse": {"toolUseId": "t1", "name": "one"},
        }}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {
            "toolUse": {"input": '{"n":1}'},
        }}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"contentBlockStart": {"contentBlockIndex": 2, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 2, "delta": {
            "reasoningContent": {"redactedContent": b"r2"},
        }}},
        {"contentBlockStop": {"contentBlockIndex": 2}},
        {"contentBlockStart": {"contentBlockIndex": 3, "start": {
            "toolUse": {"toolUseId": "t2", "name": "two"},
        }}},
        {"contentBlockDelta": {"contentBlockIndex": 3, "delta": {
            "toolUse": {"input": '{"n":2}'},
        }}},
        {"contentBlockStop": {"contentBlockIndex": 3}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]}


@pytest.mark.parametrize("streaming", [False, True])
def test_bedrock_transport_preserves_reasoning_and_order(streaming):
    from agent.bedrock_adapter import normalize_converse_response, normalize_converse_stream_events, convert_messages_to_converse
    from agent.transports.bedrock import BedrockTransport

    raw = _stream_response() if streaming else _raw_response()
    adapter_response = (
        normalize_converse_stream_events(raw)
        if streaming else normalize_converse_response(raw)
    )
    normalized = BedrockTransport().normalize_response(adapter_response)
    assert normalized.provider_data["reasoning_details"] == [
        {"type": "redacted_thinking", "data": "cjE="},
        {"type": "redacted_thinking", "data": "cjI="},
    ]

    # Exercise the real history builder against the transport's canonical
    # NormalizedResponse, rather than the adapter-only SimpleNamespace.
    sys.modules.setdefault("requests", ModuleType("requests"))
    from agent.chat_completion_helpers import build_assistant_message
    history = build_assistant_message(_FakeAgent(), normalized, "tool_calls")
    assert history["reasoning_details"] == normalized.provider_data["reasoning_details"]
    assert history["bedrock_content_blocks"] == normalized.provider_data["bedrock_content_blocks"]
    _system, messages = convert_messages_to_converse([
        {"role": "user", "content": "go"}, history,
    ])
    blocks = messages[1]["content"]
    assert [next(iter(block)) for block in blocks] == [
        "reasoningContent", "toolUse", "reasoningContent", "toolUse"
    ]
    assert [block["reasoningContent"]["redactedContent"] for block in blocks if "reasoningContent" in block] == [b"r1", b"r2"]
