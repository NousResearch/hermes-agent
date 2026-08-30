"""finish_reason wire normalization (port of oh-my-pi#9566).

Some OpenAI-compatible gateways fronting Gemini backends emit the native
uppercase finish reasons (``STOP``, ``MAX_TOKENS``) instead of the lowercase
OpenAI contract values. Every downstream comparison in Hermes uses lowercase
literals, so without normalization a clean STOP completion misses the stop
handling and a MAX_TOKENS truncation never enters the length-recovery path.

Covers the single owner (``normalize_finish_reason``) plus both wire-intake
choke points: the chat_completions transport ``normalize_response`` and the
streaming chunk-capture loop's alias import.
"""

from types import SimpleNamespace

import pytest

from agent.message_sanitization import normalize_finish_reason
from agent.transports.chat_completions import ChatCompletionsTransport


# ── single owner ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Gemini-fronting gateways: native uppercase reasons
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        # mixed case defensive fold
        ("Stop", "stop"),
        ("Tool_Calls", "tool_calls"),
        # aliases
        ("end", "stop"),
        ("END", "stop"),
        ("max_tokens", "length"),
        ("function_call", "tool_calls"),
        # contract values pass through byte-identical
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_calls"),
        ("content_filter", "content_filter"),
        ("incomplete", "incomplete"),
    ],
)
def test_normalize_finish_reason_folds_to_contract(raw, expected):
    assert normalize_finish_reason(raw) == expected


@pytest.mark.parametrize("raw", [None, "", 24, {"x": 1}])
def test_normalize_finish_reason_passes_non_string_unchanged(raw):
    # Callers keep their existing ``or "stop"`` defaults for falsy values;
    # non-string values (Poolside int reasons pre-str()) are untouched.
    assert normalize_finish_reason(raw) is raw


# ── transport intake choke point ─────────────────────────────────────


def _fake_response(finish_reason):
    msg = SimpleNamespace(content="hello", tool_calls=None, refusal=None)
    choice = SimpleNamespace(finish_reason=finish_reason, message=msg)
    return SimpleNamespace(choices=[choice], usage=None, model="gemini-3-pro")


def test_transport_normalizes_uppercase_stop():
    transport = ChatCompletionsTransport()
    normalized = transport.normalize_response(_fake_response("STOP"))
    assert normalized.finish_reason == "stop"


def test_transport_normalizes_uppercase_max_tokens_to_length():
    transport = ChatCompletionsTransport()
    normalized = transport.normalize_response(_fake_response("MAX_TOKENS"))
    assert normalized.finish_reason == "length"


def test_transport_keeps_lowercase_contract_values():
    transport = ChatCompletionsTransport()
    for reason in ("stop", "length", "tool_calls", "content_filter"):
        normalized = transport.normalize_response(_fake_response(reason))
        assert normalized.finish_reason == reason


def test_transport_poolside_integer_reason_still_stringified():
    # Pre-existing Poolside behavior: int finish_reason → str, not folded.
    transport = ChatCompletionsTransport()
    normalized = transport.normalize_response(_fake_response(24))
    assert normalized.finish_reason == "24"


def test_transport_missing_reason_defaults_to_stop():
    transport = ChatCompletionsTransport()
    normalized = transport.normalize_response(_fake_response(None))
    assert normalized.finish_reason == "stop"


# ── streaming intake choke point ─────────────────────────────────────


def test_streaming_capture_uses_shared_normalizer():
    # The streaming loop imports the same single owner under a private
    # alias — verify the alias is the shared function, not a fork.
    from agent import chat_completion_helpers as cch

    assert cch._normalize_finish_reason is normalize_finish_reason
