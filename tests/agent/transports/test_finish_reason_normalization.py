"""Uppercase wire finish reasons (STOP / MAX_TOKENS from Gemini-fronting
OpenAI-compatible gateways) must fold to the lowercase OpenAI contract at both
wire-intake choke points — the chat_completions transport and the streaming
chunk loop — so stop handling and length recovery see the values they compare
against.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.message_sanitization import normalize_finish_reason
from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_constants import PARTIAL_STREAM_STUB_ID


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("Tool_Calls", "tool_calls"),
        ("end", "stop"),
        ("function_call", "tool_calls"),
        ("content_filter", "content_filter"),  # contract values pass through byte-identical
    ],
)
def test_normalize_finish_reason_folds_to_contract(raw, expected):
    assert normalize_finish_reason(raw) == expected


@pytest.mark.parametrize("raw", [None, "", 24])
def test_normalize_finish_reason_passes_falsy_and_non_string_unchanged(raw):
    # Callers keep their ``or "stop"`` defaults; Poolside int reasons are untouched.
    assert normalize_finish_reason(raw) is raw


def _fake_response(finish_reason):
    msg = SimpleNamespace(content="hello", tool_calls=None, refusal=None)
    choice = SimpleNamespace(finish_reason=finish_reason, message=msg)
    return SimpleNamespace(choices=[choice], usage=None, model="gemini-3-pro")


@pytest.mark.parametrize("raw,expected", [("STOP", "stop"), ("MAX_TOKENS", "length"), (24, "24"), (None, "stop")])
def test_transport_normalize_response_folds_finish_reason(raw, expected):
    assert ChatCompletionsTransport().normalize_response(_fake_response(raw)).finish_reason == expected


def _make_stream_chunk(content=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
                           model=None, usage=None)


@pytest.mark.parametrize("raw,expected", [("STOP", "stop"), ("MAX_TOKENS", "length")])
@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_streaming_capture_folds_uppercase_finish_reason(_mock_close, mock_create, monkeypatch, raw, expected):
    from run_agent import AIAgent

    def _stream():
        yield _make_stream_chunk(content="partial answer")
        yield _make_stream_chunk(finish_reason=raw)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stream()
    mock_create.return_value = mock_client
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

    agent = AIAgent(api_key="test-key", base_url="https://example.com/v1", model="test/model",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    response = agent._interruptible_streaming_api_call({})

    assert response.id != PARTIAL_STREAM_STUB_ID
    assert response.choices[0].finish_reason == expected
