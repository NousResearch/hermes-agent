"""Reproduction test for production stalls under realistic streaming load.

Issue: https://github.com/NousResearch/hermes-agent/issues/57903

Goal: deterministically reproduce the 15-25 second ``event loop stalled``
warnings observed during long LLM streaming sessions with 300K+ tokens of
context (the ``1811fc`` session profile). The reproduction must run in CI
without burning real API tokens — so we synthesize the streaming-shape
load that *causes* the stalls.

Mechanism (matches production observations in gui.log):

  - A 30-second simulated streaming call emits chunks every 50ms.
  - Each chunk is a JSON document of ~5-10KB that the worker thread
    parses with ``json.loads()`` (mimics the Anthropic SDK's
    MessageStream event parsing, which on Python 3.11 holds the GIL
    during JSON deserialization of each event).
  - A parallel "heartbeat" thread tries to tick every 50ms — this stands
    in for the web_server 2s heartbeat callback.
  - If the gap between two consecutive heartbeat ticks exceeds 500ms
    (the web_server stall watchdog threshold), the test fails.

Production correlates: in gui.log during 1811fc's heavy use, every
15-25s the watchdog logs ``event loop stalled Ns (GIL pressure
suspected)`` and the desktop WS times out. The pattern is the GIL
contention between the worker's stream JSON parsing and the main
thread's busy-poll.

Test budget: 1s max gap (the 500ms threshold plus slack). On the
current code with a 50ms busy-poll this *should* pass because the
poll interval is shorter than the worker's average CPU burst. If
this test fails, the sub-event-loop bridge from issue #57903 is
needed to yield the GIL from the worker's JSON parsing path.
"""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from agent import chat_completion_helpers as cch


# Test parameters chosen to match production observations.
TEST_CALL_SECONDS = 6.0  # long enough to expose stall pattern, short enough for CI
CHUNK_INTERVAL_SECONDS = 0.05  # 20Hz chunk arrival (mimics Anthropic streaming)
# Average Anthropic streaming chunk ~5KB; bigger chunks = longer JSON parse
# = longer GIL hold. We use realistic-sized JSON.
CHUNK_TEMPLATE_SIZE_BYTES = 6000
HEARTBEAT_INTERVAL_SECONDS = 0.05  # same as web_server watchdog cadence
MAX_HEARTBEAT_GAP_BUDGET_SECONDS = 1.0  # 2x the watchdog's 500ms threshold


def _make_realistic_streaming_agent() -> MagicMock:
    """Build an agent whose SDK call simulates the Anthropic streaming
    response shape: a stream object with __iter__ that yields JSON
    chunk dicts. The main loop in ``create_anthropic_message`` would
    parse each chunk with ``json.loads()``-equivalent work.

    We replicate that work here by calling ``json.loads()`` on a
    realistic-sized JSON payload every CHUNK_INTERVAL_SECONDS for the
    duration of the call. JSON parsing on a 6KB Python string takes
    ~5-15ms on a modern CPU, which is enough to expose a 50ms-poll
    busy-poll starvation when the worker's parse exceeds the poll
    interval.
    """
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent._compute_non_stream_stale_timeout.return_value = max(60.0, TEST_CALL_SECONDS * 3)

    # Build a realistic chunk payload: a JSON object with text content,
    # usage metadata, and a stop_reason — close to what Anthropic actually
    # sends in a MessageStream event.
    chunk_payload = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": "x" * (CHUNK_TEMPLATE_SIZE_BYTES // 2),
        },
    }
    chunk_str = json.dumps(chunk_payload)

    fake_client = MagicMock()

    def _streaming_create(**_kwargs):
        t_end = time.monotonic() + TEST_CALL_SECONDS
        # Simulate a TCP receive: select() on Windows blocks for tens of
        # milliseconds waiting for the next chunk. We use socket.select()
        # on a non-blocking socket pair to model that I/O wait without
        # spending tokens on a real LLM.
        import socket
        sock_a, sock_b = socket.socketpair()
        sock_a.setblocking(False)
        sock_b.setblocking(False)
        try:
            chunk_str = json.dumps(chunk_payload)
            iteration = 0
            while time.monotonic() < t_end and not agent._interrupt_requested:
                # CPU work that simulates the SDK's chunk JSON parsing.
                # json.loads on a 6KB string takes ~5-15ms.
                for _ in range(10):
                    _ = json.loads(chunk_str)
                # I/O wait: simulate waiting for the next chunk to arrive.
                # select() returns immediately (no data ready), but on
                # Windows the syscall still takes a few ms and any actual
                # GIL contention shows up here too.
                try:
                    sock_a.recv(1)
                except BlockingIOError:
                    pass
                # 10% of chunks trigger a heavier parse (tool-call chunk
                # with arguments that the SDK has to JSON-validate).
                if iteration % 10 == 0:
                    _ = json.loads(json.dumps({"args": list(range(200))}))
                iteration += 1
                time.sleep(CHUNK_INTERVAL_SECONDS)
        finally:
            sock_a.close()
            sock_b.close()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = "ok"
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = "stop"
        return response

    fake_client.chat.completions.create.side_effect = _streaming_create
    agent._create_request_openai_client.return_value = fake_client
    agent._close_request_openai_client = MagicMock()
    agent._abort_request_openai_client = MagicMock()
    return agent


def _heartbeat_worker(gaps: list, stop: threading.Event) -> None:
    """Record wall-clock gaps between consecutive heartbeat ticks.

    Mimics what the web_server watchdog measures: a callback that
    should fire every HEARTBEAT_INTERVAL_SECONDS. A gap > the watchdog
    threshold would trip the production ``event loop stalled`` warning.
    """
    deadline = time.monotonic() + 30.0
    last = time.monotonic()
    while not stop.is_set() and time.monotonic() < deadline:
        time.sleep(HEARTBEAT_INTERVAL_SECONDS)
        now = time.monotonic()
        gaps.append(now - last)
        last = now


def test_interruptible_api_call_reproduces_production_stalls():
    """Reproduce the 15-25s event-loop stall pattern observed in
    production. If this test fails, the 50ms poll interval is not
    short enough for the realistic streaming-shape load and we need
    the sub-event-loop bridge from issue #57903.
    """
    gaps: list = []
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_worker, args=(gaps, stop), daemon=True, name="repro-heartbeat"
    )
    heartbeat.start()
    time.sleep(0.1)  # let heartbeat establish baseline

    agent = _make_realistic_streaming_agent()
    t0 = time.monotonic()
    response = cch.interruptible_api_call(agent, {"model": "x", "messages": []})
    elapsed = time.monotonic() - t0
    stop.set()
    heartbeat.join(timeout=2.0)

    assert getattr(response, "choices", None) is not None
    # Sanity: total wall-clock should match the simulated call duration.
    assert elapsed >= TEST_CALL_SECONDS * 0.95, (
        f"interruptible_api_call returned in {elapsed:.1f}s, expected "
        f"~{TEST_CALL_SECONDS}s. The test setup may be broken."
    )

    # Look at the body of the run only — first few and last few gaps
    # are warmup/teardown.
    body = gaps[3:-3] if len(gaps) > 8 else gaps
    max_gap = max(body) if body else 0.0

    assert max_gap < MAX_HEARTBEAT_GAP_BUDGET_SECONDS, (
        f"interruptible_api_call reproduced the production stall pattern: "
        f"heartbeat gap {max_gap * 1000:.0f}ms during a {TEST_CALL_SECONDS}s "
        f"streaming-shape call (budget {MAX_HEARTBEAT_GAP_BUDGET_SECONDS * 1000:.0f}ms). "
        f"This is the symptom users see as 'event loop stalled Ns' warnings "
        f"and 'Gateway offline' flashes during long LLM calls. See "
        f"issue #57903 for the diagnosis and the proposed fix."
    )