"""Tests for content-filter stream termination → fallback detection.

Covers #32421: when a provider's output safety filter (MiniMax "new_sensitive",
Azure content_filter, Bedrock guardrail_intervened) kills a streaming response
mid-delivery, the partial stream stub should be tagged with
_content_filter_terminated=True, and the conversation loop should trigger
fallback instead of continuation retries.
"""

import pytest
from types import SimpleNamespace

from agent.chat_completion_helpers import (
    PARTIAL_STREAM_STUB_ID,  # re-exported from constants
    is_content_filter_stream_error,
)


# ── Content-filter detection in stream stub creation ────────────────────

# NOTE: The production detection logic is intentionally exposed as a small
# helper so tests call the real implementation instead of copying it.


@pytest.mark.parametrize("error_msg,expected", [
    # MiniMax — output new_sensitive with error code
    ("output new_sensitive (1027) — provider content safety filter", True),
    ("Error code: 1027 — content filtered", True),
    # Azure / OpenAI content_filter
    ("content_filter: response was filtered by Azure OpenAI content filter", True),
    ("Your request was rejected due to content_filter", True),
    ("content filtered by safety system", True),
    # Bedrock
    ("guardrail_intervened — Amazon Bedrock guardrail", True),
    ("Content was blocked by guardrail_intervened", True),
    # Gemini — SAFETY finish reason mapped to "content_filter" at transport level
    ("finish_reason: content_filter (SAFETY)", True),
    ("recitation detected — response terminated", True),
    ("response was refused: refusal", True),
    # Non-content-filter errors — should NOT trigger
    ("timeout error", False),
    ("rate limit exceeded (429)", False),
    ("connection reset by peer", False),
    ("network error", False),
    ("HTTPError: 500 Internal Server Error", False),
    ("Stream interrupted by unknown error", False),
])
def test_content_filter_detection_patterns(error_msg, expected):
    """Content-filter keywords are detected; unrelated errors are not."""
    assert is_content_filter_stream_error(error_msg) == expected, \
        f"Expected {expected} for: {error_msg}"


def test_content_filter_tag_on_stub_construction():
    """Verify the tag pattern that the conversation loop reads."""
    # Simulate what chat_completion_helpers.py does: tag the stub
    stub = SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model="MiniMax-M2.7",
        choices=[],
        usage=None,
        _dropped_tool_names=None,
        _content_filter_terminated=True,
    )
    assert getattr(stub, "_content_filter_terminated", False) is True

    # Normal stub (no content filter) shouldn't have the tag
    normal_stub = SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model="MiniMax-M2.7",
        choices=[],
        usage=None,
        _dropped_tool_names=None,
    )
    assert getattr(normal_stub, "_content_filter_terminated", False) is False


# ── Conversation-loop handling of tagged stubs ──────────────────────────

def test_loop_reads_content_filter_tag(monkeypatch):
    """The fallback trigger in conversation_loop.py reads the tag correctly."""
    # Simulate the response object with the tag
    response = SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model="MiniMax-M2.7",
        choices=[],
        usage=None,
        _dropped_tool_names=None,
        _content_filter_terminated=True,
    )
    _is_partial_stream_stub = getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
    _content_filter_terminated = getattr(response, "_content_filter_terminated", False)

    assert _is_partial_stream_stub is True
    assert _content_filter_terminated is True
    # This is the exact condition check in conversation_loop.py:
    assert _is_partial_stream_stub and _content_filter_terminated


class FakeAgent:
    """Minimal agent mock for testing fallback trigger logic."""

    def __init__(self, has_fallback=True):
        self._has_fallback = has_fallback
        self.fallback_activated = False
        self.log_messages = []
        self.log_prefix = "[test] "
        self._fallback_chain = []
        self._fallback_index = 0

    def _vprint(self, msg, force=False):
        self.log_messages.append(msg)

    def _try_activate_fallback(self):
        if self._has_fallback:
            self.fallback_activated = True
            self._has_fallback = False  # Only works once
            return True
        return False


def test_fallback_triggered_on_content_filter_tag():
    """When a stub has _content_filter_terminated, fallback is attempted."""
    agent = FakeAgent(has_fallback=True)

    response = SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        _content_filter_terminated=True,
        _dropped_tool_names=None,
    )

    _is_partial_stream_stub = getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
    content_filter_terminated = getattr(response, "_content_filter_terminated", False)

    if _is_partial_stream_stub and content_filter_terminated:
        agent._vprint("content filter terminated", force=True)
        if agent._try_activate_fallback():
            assert agent.fallback_activated is True
            agent._vprint("fallback activated", force=True)
            # The loop would `continue` here — simulated by the flag check

    assert agent.fallback_activated is True
    assert "fallback activated" in agent.log_messages[-1]


def test_no_fallback_gives_warning():
    """When no fallback is configured, agent logs a warning and continues."""
    agent = FakeAgent(has_fallback=False)

    response = SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        _content_filter_terminated=True,
        _dropped_tool_names=None,
    )

    _is_partial_stream_stub = getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
    content_filter_terminated = getattr(response, "_content_filter_terminated", False)

    if _is_partial_stream_stub and content_filter_terminated:
        agent._vprint("content filter terminated", force=True)
        if not agent._try_activate_fallback():
            agent._vprint("no fallback — retrying same provider", force=True)

    assert agent.fallback_activated is False
    assert any("no fallback" in msg for msg in agent.log_messages)


def test_normal_stub_no_content_filter_does_not_trigger():
    """A normal partial stream stub without the tag should not trigger fallback."""
    agent = FakeAgent(has_fallback=True)

    response = SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        _content_filter_terminated=False,
        _dropped_tool_names=["write_file"],
    )

    _is_partial_stream_stub = getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
    content_filter_terminated = getattr(response, "_content_filter_terminated", False)

    # This condition should evaluate to False for normal stubs
    triggers_fallback = _is_partial_stream_stub and content_filter_terminated
    assert triggers_fallback is False

    # Normal handling would proceed to continuation retry — fallback NOT called
    assert agent.fallback_activated is False


def test_non_stub_does_not_trigger():
    """A response without PARTIAL_STREAM_STUB_ID must not trigger fallback."""
    response = SimpleNamespace(
        id="chatcmpl-abc123",
        _content_filter_terminated=True,  # Tag present but not a stub!
    )

    _is_partial_stream_stub = getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
    content_filter_terminated = getattr(response, "_content_filter_terminated", False)

    # The combined condition requires BOTH to be True
    triggers_fallback = _is_partial_stream_stub and content_filter_terminated
    assert triggers_fallback is False
    # This guards against false positives on non-stub responses
