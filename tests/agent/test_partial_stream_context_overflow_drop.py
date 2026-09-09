"""Regression tests for #106260: a partial-stream that dies on a context-overflow
error must NOT append the recovered fragment to the transcript.

Before the fix, ``_partial_stream_stub`` unconditionally returned the already-streamed
text as the stub content. When the stream died because the prompt had already filled the
window (``Context length exceeded: max compression attempts (3) reached.``), appending
that fragment made the next turn's prompt strictly larger than the one that just failed
— each retry failed earlier, a monotonic death spiral that compression could not save
(short conversations sit entirely inside ``protect_last_n``).

The fix classifies the stream error: when it is ``FailoverReason.context_overflow``,
the recovered fragment is dropped (stub content ``None``) so only the continuation
nudge is added. The loop's empty-stub guard then skips appending the stub, and retries
run against a same-size prompt instead of a growing one.
"""

from unittest.mock import MagicMock

import pytest

from agent.chat_completion_helpers import _StreamingCall
from hermes_constants import PARTIAL_STREAM_STUB_ID


def _make_call(agent, error, partial_text, partial_tool_names=None):
    """Construct a ``_StreamingCall`` without running ``__init__`` (avoids the
    request-client/threading setup) — only ``_partial_stream_stub`` is exercised."""
    call = _StreamingCall.__new__(_StreamingCall)
    call.agent = agent
    call.result = {"error": error, "partial_tool_names": partial_tool_names or []}
    return call


def _make_agent(*, partial_text, provider="anthropic", model="claude-sonnet-4"):
    agent = MagicMock()
    agent._current_streamed_assistant_text = partial_text
    agent.provider = provider
    agent.model = model
    agent._fire_stream_delta = lambda text: None
    return agent


class TestPartialStreamContextOverflowDrop:
    """The death-spiral fix: context-overflow errors drop the recovered fragment."""

    @pytest.mark.parametrize("error", [
        "Context length exceeded: max compression attempts (3) reached.",
        "Context length exceeded: 51,329 tokens. Cannot compress further.",
        "This model's maximum context length is 200000 tokens. However, your messages resulted in 201234 tokens.",
    ])
    def test_context_overflow_drops_partial_content(self, error):
        agent = _make_agent(partial_text="x" * 71239)
        call = _make_call(agent, error=error, partial_text="x" * 71239)
        stub = call._partial_stream_stub()
        assert stub.id == PARTIAL_STREAM_STUB_ID
        # The 71K-char fragment must NOT be in the stub — it would grow the next prompt.
        assert stub.choices[0].message.content is None, (
            "context-overflow stub must drop the recovered fragment so the next turn's "
            "prompt is not larger than the one that just failed (death spiral)"
        )
        assert stub.choices[0].message.tool_calls is None

    def test_non_context_error_keeps_partial_content(self):
        """A transient connection error still keeps the partial fragment — the loop
        continues from where the stream died (the pre-fix behaviour, unchanged)."""
        agent = _make_agent(partial_text="partial answer that was stream")
        call = _make_call(agent, error="Connection reset by peer", partial_text="partial answer that was stream")
        stub = call._partial_stream_stub()
        assert stub.choices[0].message.content == "partial answer that was stream"

    def test_context_overflow_with_dropped_tool_keeps_warning_only(self):
        """When the stream dies on context overflow mid tool-call, the dropped-tool
        warning is still surfaced (the user must know the action was not executed), but
        the recovered text fragment is dropped — no partial-text growth."""
        agent = _make_agent(partial_text="let me write the file: " + "y" * 50000)
        call = _make_call(
            agent,
            error="Context length exceeded: max compression attempts (3) reached.",
            partial_text="let me write the file: " + "y" * 50000,
            partial_tool_names=["write_file"],
        )
        stub = call._partial_stream_stub()
        content = stub.choices[0].message.content
        # Warning is present (the tool call was attempted + not executed).
        assert content is not None
        assert "Stream stalled mid tool-call" in content
        assert "write_file" in content
        # The 50K-char recovered fragment must NOT be in the stub.
        assert "y" * 100 not in content, "recovered fragment leaked into the stub despite context-overflow drop"

    def test_non_context_with_dropped_tool_keeps_partial_plus_warning(self):
        """Non-context-overflow + dropped tool: the recovered fragment stays alongside
        the warning (unchanged pre-fix behaviour — only context-overflow drops)."""
        agent = _make_agent(partial_text="partial preamble text")
        call = _make_call(
            agent,
            error="Connection reset by peer",
            partial_text="partial preamble text",
            partial_tool_names=["write_file"],
        )
        stub = call._partial_stream_stub()
        content = stub.choices[0].message.content
        assert content is not None
        assert "partial preamble text" in content  # fragment kept
        assert "Stream stalled mid tool-call" in content  # warning kept

    def test_context_overflow_empty_partial_is_noop(self):
        """If nothing was streamed before the context overflow, dropping is a no-op
        (the stub is already empty — the loop guard skips it)."""
        agent = _make_agent(partial_text="")
        call = _make_call(
            agent,
            error="Context length exceeded: max compression attempts (3) reached.",
            partial_text="",
        )
        stub = call._partial_stream_stub()
        assert stub.choices[0].message.content is None
