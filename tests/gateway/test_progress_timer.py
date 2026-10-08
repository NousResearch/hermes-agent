"""Elapsed-timer line on the edited tool-progress bubble (#4885).

The invariant is about the RELATION between the bubble's start and the rendered value: the
timer carries whole intervals of wall clock since that bubble's first progress event, holds
still while tool lines land between boundaries, and restarts from a fresh baseline after the
bubble is reset. Clocks are injected (``now=``) — no sleeps, so nothing here depends on how
busy the runner is. The exact glyphs/spelling are deliberately not pinned.
"""

import re
from types import SimpleNamespace

from gateway.session import SessionSource
from gateway.config import Platform
from gateway.turn_context import TurnContext

_TICK = 5.0
# Digits anywhere in the line: the rendered elapsed seconds, whatever marker or unit wraps it.
_ELAPSED = re.compile(r"(\d+)")


def _runner(**ctx_fields):
    from gateway.run_turn_runner import TurnRunner

    class _StubGatewayRunner:
        def _delivery_adapter_for(self, source):
            return None

    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="c1"),
        progress_grouping="accumulate",
        _run_still_current=lambda: True,
        **ctx_fields,
    )
    return TurnRunner(_StubGatewayRunner(), ctx)


def _open_bubble(runner, *, timer, interval=_TICK):
    """A progress-edit state with one tool line already rendered into an existing bubble."""
    runner._ctx.progress_timer = timer
    runner._ctx.progress_timer_interval = interval
    st = runner._progress_edit_state(SimpleNamespace())
    st.progress_msg_id = "bubble-1"
    runner._progress_absorb(st, "tool one", now=100.0)
    return st


def _elapsed_seconds(st, body):
    """Whole seconds the rendered timer carries, or None when the body carries no timer.

    The timer only ever appears as lines BEYOND the accumulated tool lines, so the count of body
    lines against the buffer is what says whether a timer is present — no marker is matched, which
    keeps this from freezing the rendering.
    """
    lines = body.splitlines()
    if len(lines) <= len(st.progress_lines):
        return None
    match = _ELAPSED.search(lines[-1])
    return int(match.group(1)) if match else None


def test_timer_value_tracks_whole_intervals_since_the_bubbles_first_event():
    runner = _runner()
    st = _open_bubble(runner, timer=True)

    # Before the first boundary there is no value at all — not a premature "0s".
    assert _elapsed_seconds(st, runner._progress_body(st, now=100.0)) is None

    # A tool line inside the interval must not invent or move the number.
    runner._progress_absorb(st, "tool two", now=103.0)
    assert _elapsed_seconds(st, runner._progress_body(st, now=104.9)) is None

    # One whole interval of wall clock since the bubble opened.
    assert _elapsed_seconds(st, runner._progress_body(st, now=105.0)) == 5
    runner._progress_absorb(st, "tool three", now=107.0)
    assert _elapsed_seconds(st, runner._progress_body(st, now=107.5)) == 5

    # Two whole intervals, still measured from the bubble's first event and not the tool's.
    assert _elapsed_seconds(st, runner._progress_body(st, now=110.0)) == 10
    assert _elapsed_seconds(st, runner._progress_body(st, now=110.0)) == 10

    # A content message closes the bubble; the next one restarts from its own first event.
    runner._reset_progress_bubble(st)
    runner._progress_absorb(st, "tool four", now=200.0)
    assert _elapsed_seconds(st, runner._progress_body(st, now=200.0)) is None
    assert _elapsed_seconds(st, runner._progress_body(st, now=205.0)) == 5


def test_no_timer_line_without_the_operator_opt_in():
    runner = _runner()
    st = _open_bubble(runner, timer=False)
    runner._progress_absorb(st, "tool two", now=104.0)
    assert runner._progress_body(st, now=1_000.0) == "tool one\ntool two"
    assert _elapsed_seconds(st, runner._progress_body(st, now=1_000.0)) is None