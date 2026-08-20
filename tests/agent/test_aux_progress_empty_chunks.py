"""Tests that content-free frames do not claim the aux progress hook.

A chunk arriving is *not* a token arriving.  Providers keep a stalled stream
open with content-free frames (keepalive pings, role-only first deltas, and
usage-only trailers), and ``CompressionCommitFence`` reads the progress hook
as "the summary model produced a token" to distinguish a slow model from a
hung one.  Ticking per-chunk makes a hung stream indistinguishable from a live
one and renders the hang detector dead code for exactly the failure it exists
to catch (#78981).
"""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.auxiliary_client import (
    _ChatStreamAccumulator,
    _aggregate_chat_stream,
    aux_progress_hook,
)
from agent.conversation_compression import CompressionCommitFence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(content=None, reasoning=None, tool_calls=None):
    """Minimal chunk with a choice but no real payload by default."""
    delta = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=None)
    return SimpleNamespace(id=None, model=None, choices=[choice], usage=None)


def _keepalive_no_choices():
    """Keepalive / usage-only trailer: ``choices`` is empty, usage populated."""
    return SimpleNamespace(
        id=None, model=None,
        choices=[],
        usage=SimpleNamespace(total_tokens=10),
    )


def _keepalive_empty_delta():
    """Role-only ping: a choice is present but its delta carries nothing."""
    return _chunk()


# ---------------------------------------------------------------------------
# Content-free frames must NOT tick the hook
# ---------------------------------------------------------------------------

class TestContentFreeFramesAreNotProgress:
    """A chunk arriving is not a token arriving.

    Waiters read the hook as "the summary model produced a token" to tell a
    slow model from a hung one (``CompressionCommitFence``).  Providers keep a
    stalled stream open with content-free frames, so ticking per-chunk makes a
    hung stream indistinguishable from a live one.
    """

    @pytest.mark.parametrize(
        "factory",
        [_keepalive_no_choices, _keepalive_empty_delta],
        ids=["usage-only-trailer", "empty-delta"],
    )
    def test_keepalive_frames_do_not_tick(self, factory):
        """50 content-free frames in a row must produce zero progress ticks."""
        ticks: list = []
        with aux_progress_hook(lambda: ticks.append(1)):
            _aggregate_chat_stream(iter([factory() for _ in range(50)]))
        assert ticks == [], (
            f"Expected 0 ticks, got {len(ticks)}; "
            "content-free frames must not advance the progress clock"
        )

    @pytest.mark.parametrize(
        "chunk, label",
        [
            (_chunk(content="tok"), "content"),
            (_chunk(reasoning="thinking"), "reasoning"),
            (
                _chunk(tool_calls=[SimpleNamespace(
                    index=0, id="call_1",
                    function=SimpleNamespace(name="f", arguments="{}"),
                )]),
                "tool_calls",
            ),
        ],
        ids=["content", "reasoning", "tool_calls"],
    )
    def test_real_deltas_still_tick(self, chunk, label):
        """Control: a chunk that actually carries a delta must tick exactly once."""
        del label
        ticks: list = []
        with aux_progress_hook(lambda: ticks.append(1)):
            _aggregate_chat_stream(iter([chunk]))
        assert len(ticks) == 1, (
            f"Expected 1 tick for a real delta, got {len(ticks)}"
        )

    def test_stalled_stream_lets_the_fence_idle_clock_run(self):
        """End-to-end against the real fence.

        A stream that delivers keepalive frames and zero tokens must not hold
        ``seconds_since_progress()`` at zero, or the compression waiter extends
        its wait to the ceiling instead of failing fast.
        """
        fence = CompressionCommitFence()

        def _stalled():
            for _ in range(10):
                time.sleep(0.02)
                yield _keepalive_no_choices()

        with aux_progress_hook(fence.touch_progress):
            _aggregate_chat_stream(_stalled())

        elapsed = fence.seconds_since_progress()
        assert elapsed >= 0.15, (
            f"Expected fence idle clock >= 0.15s after stalled stream, "
            f"got {elapsed:.3f}s — keepalive frames must not reset progress"
        )


# ---------------------------------------------------------------------------
# _ChatStreamAccumulator unit-level checks
# ---------------------------------------------------------------------------

def test_usage_only_chunk_does_not_notify_progress():
    """Usage trailer (choices=[]) must not fire the progress hook."""
    aggregator = _ChatStreamAccumulator()
    with patch("agent.auxiliary_client._notify_aux_progress") as mock_notify:
        aggregator.feed(
            SimpleNamespace(
                usage=SimpleNamespace(total_tokens=10),
                choices=[],
            )
        )
        mock_notify.assert_not_called()


def test_empty_delta_does_not_notify_progress():
    """Role-only first delta (content/reasoning/tool_calls all None) must not tick."""
    aggregator = _ChatStreamAccumulator()
    with patch("agent.auxiliary_client._notify_aux_progress") as mock_notify:
        aggregator.feed(
            SimpleNamespace(choices=[
                SimpleNamespace(delta=SimpleNamespace(
                    content=None, reasoning=None, tool_calls=None,
                ))
            ])
        )
        mock_notify.assert_not_called()


def test_content_delta_notifies_progress():
    aggregator = _ChatStreamAccumulator()
    with patch("agent.auxiliary_client._notify_aux_progress") as mock_notify:
        aggregator.feed(_chunk(content="hello"))
        mock_notify.assert_called_once()


def test_reasoning_delta_notifies_progress():
    aggregator = _ChatStreamAccumulator()
    with patch("agent.auxiliary_client._notify_aux_progress") as mock_notify:
        aggregator.feed(_chunk(reasoning="let me think"))
        mock_notify.assert_called_once()


def test_tool_call_delta_notifies_progress():
    aggregator = _ChatStreamAccumulator()
    with patch("agent.auxiliary_client._notify_aux_progress") as mock_notify:
        aggregator.feed(_chunk(tool_calls=[SimpleNamespace(
            index=0, id="c1",
            function=SimpleNamespace(name="fn", arguments="{}"),
        )]))
        mock_notify.assert_called_once()


def test_anthropic_ping_event_does_not_tick_progress():
    from agent.auxiliary_client import _anthropic_event_has_content
    ping_event = SimpleNamespace(type="ping")
    assert _anthropic_event_has_content(ping_event) is False

    text_event = SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(text="hello", thinking=None, partial_json=None),
    )
    assert _anthropic_event_has_content(text_event) is True


def test_codex_keepalive_event_does_not_tick_progress():
    from agent.auxiliary_client import _codex_event_has_content
    created_event = SimpleNamespace(type="response.created")
    assert _codex_event_has_content(created_event) is False


@pytest.mark.parametrize(
    "event_type",
    [
        "response.output_text.delta",
        "response.reasoning_summary_text.delta",
    ],
)
def test_codex_runtime_text_deltas_count_as_progress(event_type):
    """Wire names from agent/codex_runtime.py must tick the fence.

    A healthy Codex aux stream emits response.output_text.delta (and
    response.reasoning_summary_text.delta) for the actual token flow.
    Matching only the guessed response.text.delta name leaves minutes of
    live text looking idle, so the 120s abort kills a healthy stream.
    """
    from agent.auxiliary_client import _codex_event_has_content

    assert _codex_event_has_content(SimpleNamespace(type=event_type)) is True
    assert _codex_event_has_content({"type": event_type}) is True

