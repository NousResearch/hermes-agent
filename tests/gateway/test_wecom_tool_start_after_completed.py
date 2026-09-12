"""Regression: a tool starting after a PRIOR tool completed must arm an
immediate first frame — not fall into the same arm gap the continuation-
thinking fix (ccd42de904) closed for on_llm_thinking.

User-reported intermittent symptom: after body text finishes and the model
calls a tool, the tool-progress / timer line takes SEVERAL SECONDS to appear.

Mechanism (sibling of the on_llm_thinking gap): ``on_tool_completed`` pops the
finished tool's ``_tool_start_times`` entry but never cancels
``_tool_timer_handle`` — so a periodic handle armed for the just-finished tool
survives as a ZOMBIE.  When the next tool starts, ``_start_tool_timer`` sees::

    need_arm = self._tool_timer_handle is None and self._tool_timer_loop is not None
    #          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  False — zombie handle set
    if need_arm:                # skipped
        _arm_tool_timer()       # the ONLY synchronous first-tick path

So the new tool's progress/timer frame is NOT pushed synchronously; it waits
for the zombie handle's next ``call_later(1.0)`` tick — the visible
seconds-long gap.  Intermittent because it depends on the exact residual
handle state at the body→tool boundary (whether a zombie handle survives, and
how far into its 1s cadence it is).

Drives the real ToolTimerMixin path via on_tool_started / on_tool_completed.
Proven RED on the current ``need_arm`` gate; a fix (rearm when a stale handle
survives a tool boundary) turns it green.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.gateway.test_wecom_continuation_thinking_live_entry import _make_consumer, _drain_timer_ticks


class TestToolStartAfterPriorToolCompleted:
    @pytest.mark.asyncio
    async def test_immediate_first_frame_when_prior_tool_left_a_zombie_handle(self):
        """A tool starting while a prior tool's periodic handle is still set
        (zombie, not cancelled by on_tool_completed) must STILL push its first
        frame within the sub-second window — not wait for the next 1s tick."""
        sc = _make_consumer()
        sc._use_native_streaming = True

        task = asyncio.create_task(sc.run())
        try:
            await asyncio.sleep(0.12)
            assert sc._native_stream_opened is True

            # Tool 1 starts → timer arms (handle set), then completes.
            sc.on_tool_started("terminal", tool_call_id="call-1")
            await asyncio.sleep(0.05)
            assert sc._tool_timer_handle is not None       # tick loop armed
            assert "call-1" in sc._tool_start_times

            sc.on_tool_completed("terminal", 1.0, tool_call_id="call-1")
            await asyncio.sleep(0.02)
            # on_tool_completed popped the entry but did NOT cancel the handle:
            # the handle is now a ZOMBIE pointing at no live tool entry.
            assert "call-1" not in sc._tool_start_times
            assert sc._tool_timer_handle is not None        # zombie survives

            # Body text streams (not modelled here — the point is the handle is
            # still set). A continuation tool now starts.
            _drain_timer_ticks(sc)
            tick_before = sc._tool_timer_tick_count

            sc.on_tool_started("read_file", tool_call_id="call-2")
            await asyncio.sleep(0.05)  # < 1s tick cadence

            assert "call-2" in sc._tool_start_times
            _drain_timer_ticks(sc)
            assert sc._tool_timer_tick_count > tick_before, (
                "a tool starting after a prior tool completed (zombie handle "
                "still set) must push its first progress/timer frame within the "
                "sub-second window — _start_tool_timer's need_arm gate "
                "(handle is None) must not leave the frame waiting for the "
                "zombie handle's next call_later(1.0) tick"
            )
        finally:
            sc.finish()
