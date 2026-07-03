"""Benchmark + diagnostic for the interruptible_api_call main-thread busy-poll.

Issue: https://github.com/NousResearch/hermes-agent/issues/57903

The non-streaming ``interruptible_api_call`` in
``agent/chat_completion_helpers.py`` blocks the main thread in a busy-poll
loop (``t.join(timeout=0.3)``) while waiting for the LLM API call to
return. Between joins the asyncio event loop in the same process (uvicorn
+ tui_gateway WebSocket server) cannot service its 2s heartbeat, the
5s-stall watchdog in ``hermes_cli/web_server.py`` fires, and the desktop
WebSocket trips its 10s timeout — manifesting as "Gateway offline" /
"не отвечает" flashes during long-running LLM calls.

This script reproduces the symptom locally and quantifies the
regression by running a simulated ``_call`` that sleeps for N seconds and
measuring how many times a parallel "heartbeat" thread gets to tick
while the busy-poll is in progress.

Usage::

    # Run with the current (broken) code:
    python scripts/bench_interruptible_api_call.py --seconds 30
    # → expect ~100 heartbeat ticks (GIL is held for the full 300ms poll)

    # After a fix that replaces t.join(timeout=0.3) with
    # future.result(timeout=0.05) (or equivalent):
    # → expect ~600 heartbeat ticks (GIL released every 50ms)

A 5x increase in heartbeat ticks per 30s window is the success criterion.
If your machine shows less than 3x, the fix is not aggressive enough;
if it shows ~100 ticks, the busy-poll is still starving the event loop.

The benchmark only uses sync primitives available in the stdlib — no
asyncio, no fixtures, no model provider, no API keys.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

# Bootstrap: when invoked directly (not via pytest), add repo root to path.
import pathlib
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent import chat_completion_helpers as cch  # noqa: E402


def _make_agent(duration_seconds: float) -> MagicMock:
    """An agent mock whose SDK call sleeps for ``duration_seconds``.

    The agent's surface is the minimum needed to drive
    ``interruptible_api_call`` through the non-streaming
    ``chat_completions`` path. All Codex / Anthropic / Bedrock /
    MoA branches short-circuit to the default chat_completions path
    so we can use the simplest mock.
    """
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent._compute_non_stream_stale_timeout.return_value = max(60.0, duration_seconds * 3)

    fake_client = MagicMock()

    def _slow_create(**_kwargs):
        # Simulate a long-running LLM call that returns a minimal
        # chat.completions response.
        time.sleep(duration_seconds)
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = "ok"
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = "stop"
        return response

    fake_client.chat.completions.create.side_effect = _slow_create
    agent._create_request_openai_client.return_value = fake_client
    agent._close_request_openai_client = MagicMock()
    agent._abort_request_openai_client = MagicMock()
    return agent


def _heartbeat_worker(counter: list, interval: float = 0.1, stop: threading.Event | None = None) -> None:
    """Tick ``counter[0] += 1`` every ``interval`` seconds.

    Stops when ``stop`` is set, or runs forever if ``stop`` is None.
    Uses ``time.sleep(interval)`` which releases the GIL on every
    iteration — this is a *contender* for the GIL, not a privileged
    scheduler. The number of ticks we accumulate while
    ``interruptible_api_call`` runs is the proxy for "how much does the
    main thread's busy-poll block other threads from doing work".
    """
    deadline = time.monotonic() + 60 * 60  # upper bound, safety
    while True:
        if stop is not None and stop.is_set():
            return
        if time.monotonic() > deadline:
            return
        counter[0] += 1
        time.sleep(interval)


def run_benchmark(duration_seconds: float) -> dict:
    """Run the benchmark and return a dict of measurements."""
    agent = _make_agent(duration_seconds)
    ticks: list = [0]
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_worker,
        args=(ticks, 0.1, stop),
        daemon=True,
        name="bench-heartbeat",
    )

    # Warm up the heartbeat thread so the very first tick doesn't
    # under-count the contention window.
    heartbeat.start()
    time.sleep(0.2)

    # Reset the counter and start the timed window.
    ticks[0] = 0
    t0 = time.monotonic()
    response = cch.interruptible_api_call(agent, {"model": "x", "messages": []})
    elapsed = time.monotonic() - t0
    stop.set()
    heartbeat.join(timeout=2.0)

    # ``response`` is a MagicMock from _slow_create; just check it has
    # the expected attribute, the value is not interesting.
    assert getattr(response, "choices", None) is not None, (
        "interruptible_api_call returned no response after sleeping "
        f"{duration_seconds:.1f}s; the helper is broken or the mock is wrong"
    )

    expected_ticks_at_50ms = duration_seconds * 20  # 1000 / 50
    expected_ticks_at_100ms = duration_seconds * 10  # 1000 / 100
    expected_ticks_at_300ms = duration_seconds / 0.3  # current code baseline

    return {
        "duration_seconds": round(duration_seconds, 2),
        "elapsed_seconds": round(elapsed, 3),
        "heartbeat_ticks": ticks[0],
        "expected_baseline_300ms_poll": round(expected_ticks_at_300ms, 1),
        "expected_fixed_50ms_poll": round(expected_ticks_at_50ms, 1),
        "ratio_to_baseline": round(ticks[0] / expected_ticks_at_300ms, 2)
            if expected_ticks_at_300ms > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="How long the simulated LLM call should sleep (default: 30s).",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    args = parser.parse_args()

    if args.seconds < 0.5:
        parser.error("--seconds must be at least 0.5 (need a long-enough call to measure)")

    result = run_benchmark(args.seconds)

    if args.output == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"Simulated LLM call duration:  {result['duration_seconds']}s")
        print(f"Wall-clock duration:           {result['elapsed_seconds']}s")
        print(f"Heartbeat ticks during call:   {result['heartbeat_ticks']}")
        print(f"  Baseline (300ms poll, current code): {result['expected_baseline_300ms_poll']}")
        print(f"  Target   (50ms poll, fixed code):     {result['expected_fixed_50ms_poll']}")
        ratio = result["ratio_to_baseline"]
        if ratio >= 4.0:
            print(f"Ratio to baseline: {ratio}x  → FIXED (>= 4x is the success criterion)")
            return 0
        if ratio >= 2.0:
            print(f"Ratio to baseline: {ratio}x  → PARTIAL (better but not the target)")
            return 1
        print(f"Ratio to baseline: {ratio}x  → BROKEN (< 2x means the busy-poll still wins)")
        return 2


if __name__ == "__main__":
    sys.exit(main())
