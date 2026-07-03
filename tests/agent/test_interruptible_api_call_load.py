"""Regression load test: does interruptible_api_call starve the asyncio
event loop under realistic streaming-shape load?

Issue: https://github.com/NousResearch/hermes-agent/issues/57903

The companion ``test_interruptible_api_call_poll_window_is_short``
test only checks the poll timeout literal. This test exercises the
*realistic* failure mode observed in production: a long LLM streaming
call where the worker thread spends most of its time on CPU work
(JSON parsing each streaming chunk, prompt-cache reconstruction,
tool-call argument assembly) and releases the GIL only briefly
between bursts. The pre-fix 300ms ``t.join`` busy-poll let the main
thread block for up to 300ms between checks; even the 50ms post-fix
poll can be starved if the worker's CPU burst exceeds 50ms.

The test launches an ``interruptible_api_call`` against a stub agent
whose SDK call mimics this streaming behavior — periodic CPU bursts
holding the GIL for 80-150ms separated by short GIL-yield sleeps.
While that runs, a parallel thread drives a mock asyncio event-loop
heartbeat every 50ms and records the largest gap between
consecutive heartbeats. If the gap exceeds 500ms, the heartbeat
service would have stalled long enough to trip the web_server's 5s
watchdog in production.

Why this matters: the production stalls are 15-25 seconds. The
worker-thread CPU-burst pattern is the most plausible mechanism we
haven't ruled out yet. This test gives us a deterministic
reproduction we can iterate on without burning tokens on a real LLM.

Pre-fix (300ms poll, no ``time.sleep(0)``): max gap ~ 300ms or more
when worker holds GIL, accumulating to several seconds over a 30s
call. Likely FAIL with budget 500ms.

Post-fix (50ms poll + ``time.sleep(0)``): max gap should be ~80-150ms
(worker CPU burst size). Should PASS with budget 500ms.

Sub-event-loop bridge (follow-up): worker thread owns its own event
loop and yields GIL naturally on each chunk await. Max gap should
drop to <50ms. Should PASS with a tighter 100ms budget.
"""
from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from agent import chat_completion_helpers as cch


# Test parameters chosen to fit in a CI budget (~5s) while still
# exercising the stall mechanism (a single 100ms CPU burst inside
# a 2s call is enough to expose 50ms-poll starvation).
TOTAL_CALL_SECONDS = 2.0
CPU_BURST_INTERVAL_SECONDS = 0.05  # 20Hz bursts (mimics stream chunks)
CPU_BURST_DURATION_SECONDS = 0.08  # 80ms CPU = exceeds the 50ms poll window
MAX_HEARTBEAT_GAP_BUDGET_SECONDS = 0.5


def _make_streaming_shape_agent() -> MagicMock:
    """Build an agent whose SDK call mimics real LLM streaming:
    periodic CPU bursts separated by short GIL-yield sleeps. The
    bursts hold the GIL for longer than the main thread's poll
    window so the busy-poll can be starved. The yield-sleeps
    release the GIL between bursts so the event loop *can*
    progress — the question is whether it has enough wall-clock
    between bursts to do so.
    """
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    # 60s stale timeout — well above the 2s test call so it never fires.
    agent._compute_non_stream_stale_timeout.return_value = 60.0

    fake_client = MagicMock()

    def _streaming_shape_create(**_kwargs):
        t_end = time.monotonic() + TOTAL_CALL_SECONDS
        while time.monotonic() < t_end and not agent._interrupt_requested:
            # CPU burst: busy work holding the GIL.
            burst_end = time.monotonic() + CPU_BURST_DURATION_SECONDS
            while time.monotonic() < burst_end:
                _ = sum(range(1000))
            # Brief GIL yield between bursts.
            time.sleep(CPU_BURST_INTERVAL_SECONDS - CPU_BURST_DURATION_SECONDS
                       if CPU_BURST_INTERVAL_SECONDS > CPU_BURST_DURATION_SECONDS
                       else 0.0)
        # Return a minimal valid chat-completions response.
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = "ok"
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = "stop"
        return response

    fake_client.chat.completions.create.side_effect = _streaming_shape_create
    agent._create_request_openai_client.return_value = fake_client
    agent._close_request_openai_client = MagicMock()
    agent._abort_request_openai_client = MagicMock()
    return agent


def _heartbeat_worker(gaps: list, stop: threading.Event) -> None:
    """Record wall-clock gaps between consecutive heartbeat ticks.

    The mock event loop "heartbeats" every 20ms. If ``interruptible_api_call``
    blocks the main thread (or any thread the heartbeat is running on)
    past 500ms, the next tick is delayed by the block time. Recording the
    max gap captures "how long was the main thread unresponsive" — exactly
    the symptom the web_server watchdog measures.
    """
    HEARTBEAT_INTERVAL = 0.02  # 20ms — 25x finer than the watchdog's 500ms threshold
    deadline = time.monotonic() + 30.0
    last = time.monotonic()
    while not stop.is_set() and time.monotonic() < deadline:
        time.sleep(HEARTBEAT_INTERVAL)
        now = time.monotonic()
        gaps.append(now - last)
        last = now


def test_interruptible_api_call_does_not_starve_event_loop_under_cpu_load():
    """The fix from commit 29f55d4 keeps the busy-poll window at 50ms;
    that should be short enough that the main thread stays responsive
    even when the worker thread is doing CPU work between stream chunks.
    If this test fails, the 50ms window is still too long for the
    realistic streaming-shape load, and we need the sub-event-loop
    bridge (issue #57903 follow-up).
    """
    gaps: list = []
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_worker, args=(gaps, stop), daemon=True, name="load-heartbeat"
    )
    heartbeat.start()
    # Let the heartbeat establish a stable baseline.
    time.sleep(0.1)

    agent = _make_streaming_shape_agent()
    t0 = time.monotonic()
    response = cch.interruptible_api_call(agent, {"model": "x", "messages": []})
    elapsed = time.monotonic() - t0
    stop.set()
    heartbeat.join(timeout=2.0)

    assert getattr(response, "choices", None) is not None

    # Filter the gaps: the first ~5 are warmup, and the last few may
    # be the heartbeat thread's own wakeup delay at shutdown. Look at
    # the body of the run only.
    body = gaps[3:-3] if len(gaps) > 8 else gaps
    max_gap = max(body) if body else 0.0

    assert max_gap < MAX_HEARTBEAT_GAP_BUDGET_SECONDS, (
        f"interruptible_api_call starved the GIL: heartbeat gap {max_gap * 1000:.0f}ms "
        f"during a {TOTAL_CALL_SECONDS}s streaming-shape call (budget "
        f"{MAX_HEARTBEAT_GAP_BUDGET_SECONDS * 1000:.0f}ms). The 50ms poll window "
        f"is not short enough for this CPU-burst pattern — see issue "
        f"#57903 and the sub-event-loop bridge design."
    )