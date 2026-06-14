"""Regression tests for issue #45908 — partial-stream recovery stub shape in
``api_mode: anthropic_messages``.

When the Anthropic SDK's ``get_final_message()`` raises (e.g. an ``IndexError``
from a malformed upstream stream with non-contiguous ``content_block`` indices)
*after* deltas were already streamed, ``interruptible_streaming_api_call``
returns a recovery stub. That stub used to be OpenAI-shaped
(``choices=[...]``) regardless of ``api_mode``. In anthropic_messages mode the
transport's ``validate_response`` checks ``response.content``, so the
OpenAI-shaped stub failed validation and the conversation loop classified it as
a retryable invalid response — retrying the unrecoverable stream up to 10x.

These tests pin the fix: in anthropic_messages mode the recovery stub is
Anthropic-shaped (``content`` list of blocks + ``stop_reason``), so it passes
``AnthropicTransport.validate_response`` and normalizes back to the recovered
text, letting the loop continue instead of retrying.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_constants import PARTIAL_STREAM_STUB_ID
from agent.transports.anthropic import AnthropicTransport


# ── Fake Anthropic streaming primitives ────────────────────────────────────

def _text_delta_event(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


class _FakeAnthropicStream:
    """Mimics the SDK streaming context manager: iterable of events, with a
    ``get_final_message()`` that raises to simulate a malformed stream."""

    def __init__(self, events, final_exc):
        self._events = events
        self._final_exc = final_exc
        self.response = SimpleNamespace(headers={}, status_code=200)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        raise self._final_exc


def _make_anthropic_agent():
    from run_agent import AIAgent
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com",
        model="claude-opus-test",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "anthropic_messages"
    agent._interrupt_requested = False
    # Credential refresh is a no-op in tests.
    agent._try_refresh_anthropic_client_credentials = lambda: False
    return agent


def _install_malformed_stream(agent, events, final_exc):
    client = MagicMock()
    client.messages.stream.return_value = _FakeAnthropicStream(events, final_exc)
    agent._anthropic_client = client


# ── Tests ──────────────────────────────────────────────────────────────────

class TestAnthropicPartialStreamStubShape:
    def test_indexerror_after_text_returns_anthropic_shaped_stub(self, monkeypatch):
        """#45908: a get_final_message() IndexError after streamed text must
        yield an Anthropic-shaped stub (``content`` list, no ``choices``) that
        recovers the partial text."""
        agent = _make_anthropic_agent()
        agent._current_streamed_assistant_text = "partial answer so far"
        _install_malformed_stream(
            agent,
            [_text_delta_event("partial answer so far")],
            IndexError("list index out of range"),
        )
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

        response = agent._interruptible_streaming_api_call({})

        assert response.id == PARTIAL_STREAM_STUB_ID
        # Anthropic-shaped: a content list, not OpenAI choices.
        assert hasattr(response, "content")
        assert not hasattr(response, "choices")
        assert isinstance(response.content, list) and response.content
        assert response.content[0].type == "text"
        assert response.content[0].text == "partial answer so far"
        assert response.stop_reason == "max_tokens"

    def test_stub_passes_anthropic_validation_and_normalizes(self, monkeypatch):
        """The recovered stub must pass AnthropicTransport.validate_response
        (the check that previously failed → 10x retry) and normalize back to
        the recovered text with a length finish_reason so the loop continues."""
        agent = _make_anthropic_agent()
        agent._current_streamed_assistant_text = "recovered text"
        _install_malformed_stream(
            agent,
            [_text_delta_event("recovered text")],
            IndexError("list index out of range"),
        )
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

        response = agent._interruptible_streaming_api_call({})

        transport = AnthropicTransport()
        assert transport.validate_response(response) is True
        assert transport.map_finish_reason(response.stop_reason) == "length"
        normalized = transport.normalize_response(response)
        assert normalized.content == "recovered text"
        assert normalized.finish_reason == "length"

    def test_no_recovered_text_uses_end_turn_so_validation_passes(self, monkeypatch):
        """With nothing recovered, the stub is an empty content list with
        stop_reason end_turn — the only empty shape that validates — so the
        loop ends cleanly instead of retrying the dead stream."""
        agent = _make_anthropic_agent()
        # A delta fired (so the recovery path runs), but no text was retained
        # (e.g. only thinking, or scrubbed output). Keep the accumulator empty
        # and the delta a no-op so _partial_text resolves to None.
        agent._fire_stream_delta = lambda text: None
        agent._current_streamed_assistant_text = ""
        _install_malformed_stream(
            agent,
            [_text_delta_event("x")],
            IndexError("list index out of range"),
        )
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

        response = agent._interruptible_streaming_api_call({})

        assert response.id == PARTIAL_STREAM_STUB_ID
        assert response.content == []
        assert response.stop_reason == "end_turn"
        transport = AnthropicTransport()
        assert transport.validate_response(response) is True

    def test_old_openai_shaped_stub_would_fail_validation(self):
        """Guard documenting the bug: an OpenAI-shaped stub fails the
        Anthropic validator, which is what drove the 10x retry storm."""
        transport = AnthropicTransport()
        openai_stub = SimpleNamespace(
            id=PARTIAL_STREAM_STUB_ID,
            model="claude-opus-test",
            usage=None,
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content="x", tool_calls=None),
                finish_reason="length",
            )],
        )
        assert transport.validate_response(openai_stub) is False
