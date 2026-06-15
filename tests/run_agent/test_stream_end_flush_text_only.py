"""Regression tests for the text-only stream-end flush fix (PR #27463).

When the assistant returns a text-only response (no tool calls),
``conversation_loop.run_conversation`` used to never finalise the streaming
display. PR #27463 added a ``stream_delta_callback(None)`` call in the
no-tool-calls branch and set ``agent._response_was_previewed = True`` so
``cli.py`` would skip the second Rich Panel render.

The hermes-sweeper review (teknium1, 2026-06-13) flagged two issues:

1. ``_response_was_previewed`` was set unconditionally once a callback was
   configured, even when no visible text had actually been streamed.  If
   the response produced only buffered content the partial-tag detector
   swallowed and never recovered, or only a <think> block, the CLI would
   skip its final render and silently hide the assistant's reply.
2. There were no regression tests pinning this behaviour.

These tests pin the corrected behaviour.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_stream_chunk(content=None, finish_reason=None, model="test/model"):
    """Build a mock streaming chunk matching OpenAI's ChatCompletionChunk shape."""
    delta = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=None)


def _make_empty_chunk():
    """Build a usage-only final chunk (no choices)."""
    return SimpleNamespace(
        choices=[],
        model="test/model",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
    )


class _FakeChatCompletions:
    """Stand-in for ``client.chat.completions.create``.

    Returns a fresh iterator every call.  ``conversation_loop.py`` short-
    circuits the streaming code path when the request client is a
    ``unittest.mock.Mock`` instance, so we use a plain class.
    """

    def __init__(self, side_effects):
        self._side_effects = list(side_effects)
        self.call_count = 0
        self.call_args_list = []

    def create(self, **kwargs):
        self.call_count += 1
        self.call_args_list.append(kwargs)
        idx = min(self.call_count - 1, len(self._side_effects) - 1)
        chunks = self._side_effects[idx]
        return iter(chunks)


class _FakeRequestClient:
    """Bare-bones OpenAI request client.  NOT a ``Mock`` subclass so
    ``conversation_loop`` keeps the streaming code path active."""

    def __init__(self, side_effects):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(side_effects))


class _FakeStreamDisplay:
    """Mimics just enough of ``HermesCLI`` for the gate to read.

    The production gate in ``conversation_loop.py`` reads
    ``stream_delta_callback.__self__._stream_flushed_chars``.  The
    callback must therefore be a bound method on an object that exposes
    that counter — exactly what a real ``HermesCLI`` instance does.
    """

    def __init__(self):
        self._stream_flushed_chars = 0
        self.deltas = []
        self.flush_none_calls = 0

    def _stream_delta(self, text):
        if text is None:
            # Mirrors the production flush — caller fires this with
            # ``None`` at end-of-stream to close the response box.
            self.flush_none_calls += 1
            return
        self._stream_flushed_chars += len(text)
        self.deltas.append(text)


@pytest.fixture()
def loop_agent():
    """AIAgent wired through ``_create_request_openai_client`` patches,
    mirroring the pattern used in ``test_streaming.py``.
    """
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


# ── Tests ────────────────────────────────────────────────────────────────


class TestTextOnlyStreamEndFlush:
    """Pin the no-tool-calls stream-end flush behaviour."""

    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_visible_text_streamed_sets_response_was_previewed(
        self, mock_create, mock_close, loop_agent,
    ):
        """A text-only response that streams visible text must mark
        ``result["response_previewed"] = True`` so the CLI skips the
        second Rich Panel render.  Regression guard for the original
        PR #27463 fix.

        Note: ``agent._response_was_previewed`` is reset to ``False``
        by ``turn_finalizer.finalize_turn`` (line 363) right after
        capturing the result dict, so it is NOT observable on the
        agent post-call.  We assert on the result dict instead — that
        is what ``cli.py:10483`` actually reads.
        """
        display = _FakeStreamDisplay()
        loop_agent.stream_delta_callback = display._stream_delta

        mock_create.return_value = _FakeRequestClient(side_effects=[
            [
                _make_stream_chunk(content="Hello"),
                _make_stream_chunk(content=" world"),
                _make_stream_chunk(content="!", finish_reason="stop"),
                _make_empty_chunk(),
            ],
        ])

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hi")

        assert display._stream_flushed_chars == len("Hello world!"), (
            f"Expected 12 visible chars to flow through _stream_delta, "
            f"got {display._stream_flushed_chars}.  Deltas: {display.deltas!r}"
        )
        assert display.flush_none_calls >= 1, (
            "stream_delta_callback(None) must be fired at end-of-stream so "
            "the CLI closes the response box and flushes any buffered text."
        )
        assert result.get("response_previewed") is True, (
            "When visible text was streamed, result['response_previewed'] "
            "must be True so the CLI doesn't render the response a second "
            f"time.  Got: {result.get('response_previewed')!r}"
        )

    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_no_visible_text_keeps_response_was_previewed_false(
        self, mock_create, mock_close, loop_agent,
    ):
        """Regression for the sweeper's #1 finding: if the callback was
        configured but no visible text was ever emitted (here: a stream
        that produces only a <think> block, stripped by the filter),
        ``result["response_previewed"]`` must be False.  The CLI's final
        Rich Panel render is what makes the response visible in that
        case — suppressing it would silently hide the reply.
        """
        display = _FakeStreamDisplay()
        loop_agent.stream_delta_callback = display._stream_delta

        # Model returns only a <think> block with no content after it.
        # The streaming filter strips the <think>...</think> pair, so
        # the user never sees any visible characters live.
        mock_create.return_value = _FakeRequestClient(side_effects=[
            [
                _make_stream_chunk(content="<think>"),
                _make_stream_chunk(content="internal monologue only"),
                _make_stream_chunk(content="</think>", finish_reason="stop"),
                _make_empty_chunk(),
            ],
        ])

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("think about it")

        assert display._stream_flushed_chars == 0, (
            "No visible text was streamed; the counter must be zero."
        )
        assert display.flush_none_calls >= 1, (
            "stream_delta_callback(None) must still fire to close the "
            "response box (with no footer, since the box was never opened)."
        )
        assert result.get("response_previewed") is False, (
            "When nothing visible was streamed, result['response_previewed'] "
            "must be False — otherwise the CLI would skip its final "
            "render and the assistant's reply would be hidden.  "
            f"Got: {result.get('response_previewed')!r}"
        )

    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_partial_tag_recovered_text_counts_as_visible(
        self, mock_create, mock_close, loop_agent,
    ):
        """Regression for the sweeper's #2 finding: text held back by
        partial-tag detection in ``_stream_prefilt`` (e.g. a trailing
        ``<`` at a chunk boundary that the streaming filter suspected
        might be the start of ``<think>``) gets recovered by
        ``_flush_stream()`` and emitted as regular text.  That recovered
        text must count toward ``_stream_flushed_chars`` so the gate
        recognises the response as already previewed.
        """
        display = _FakeStreamDisplay()
        loop_agent.stream_delta_callback = display._stream_delta

        # Stream where a chunk ends with a bare "<" that the
        # partial-tag detector will hold, then a follow-up that
        # resolves the tag without it being a real <think> opener, so
        # the held "<" must be recovered as regular text in
        # _flush_stream().
        mock_create.return_value = _FakeRequestClient(side_effects=[
            [
                _make_stream_chunk(content="Here is the answer: 4"),
                _make_stream_chunk(content="2 <"),  # trailing '<' held
                _make_stream_chunk(content="3", finish_reason="stop"),
                _make_empty_chunk(),
            ],
        ])

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("what is 6*7?")

        # The visible-streamed portion plus the trailing "<" that
        # _flush_stream() recovers must produce a non-zero counter.
        # We don't pin the exact split between streamed-vs-recovered
        # here — that's the partial-tag detector's contract — only that
        # *some* visible text was emitted and the gate flipped.
        assert display._stream_flushed_chars > 0, (
            "Partial-tag recovery must contribute to _stream_flushed_chars "
            "so the gate recognises recovered text as visible content.  "
            f"Got deltas: {display.deltas!r}"
        )
        assert result.get("response_previewed") is True, (
            "When the partial-tag recovery emits visible text, the gate "
            "must mark the response as already previewed so the CLI "
            "doesn't double-render it.  "
            f"Got: {result.get('response_previewed')!r}"
        )
