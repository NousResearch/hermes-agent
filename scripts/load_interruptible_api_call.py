"""Load test for interruptible_api_call under concurrent LLM-call pressure.

Issue: https://github.com/NousResearch/hermes-agent/issues/57903

The companion ``bench_interruptible_api_call.py`` runs a *single*
simulated LLM call and measures GIL yield + interrupt latency. This
script runs **N concurrent simulated calls** to exercise the
busy-poll under contention — the same shape as a desktop session
with multiple open tabs each driving their own conversation turn.

What it measures:
  1. **Wall-clock duration per call** — should be close to the
     simulated duration, with no extra latency from contention.
  2. **Heartbeat tick count** in a parallel sentinel thread that
     mimics the web_server event loop's 2s heartbeat. With the
     50ms poll interval (post-fix) the heartbeat should keep
     ticking; with the 300ms poll (pre-fix) it would also tick
     (Python's GIL switch handles it), but the *event loop in
     the same thread* would not get to advance.

This script does **not** measure event-loop starvation directly —
that requires running the real Hermes serve and inspecting gui.log.
What it does measure is whether concurrent ``interruptible_api_call``
calls on the same Python process starve the GIL or the event loop.

Usage::

    # Single call, 8 seconds — sanity check
    python scripts/load_interruptible_api_call.py --seconds 8 --concurrency 1

    # 4 concurrent calls, 8 seconds each
    python scripts/load_interruptible_api_call.py --seconds 8 --concurrency 4

    # Default: 3 concurrent calls, 8 seconds each
    python scripts/load_interruptible_api_call.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time
from unittest.mock import MagicMock

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent import chat_completion_helpers as cch  # noqa: E402


HEARTBEAT_INTERVAL_SECONDS = 0.1  # mimic web_server's 2s heartbeat at 20x rate for finer resolution


def _make_agent(duration_seconds: float) -> MagicMock:
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent._compute_non_stream_stale_timeout.return_value = max(
        60.0, duration_seconds * 3
    )
    fake_client = MagicMock()

    def _slow_create(**_kwargs):
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


def _heartbeat_worker(tick_holder: list, stop: threading.Event) -> None:
    """Tick every HEARTBEAT_INTERVAL_SECONDS. Counts voluntary context
    switches the OS observes on this thread — the same kind of signal
    the web_server watchdog looks for."""
    deadline = time.monotonic() + 600.0
    while not stop.is_set() and time.monotonic() < deadline:
        tick_holder[0] += 1
        time.sleep(HEARTBEAT_INTERVAL_SECONDS)


def run_load(duration_seconds: float, concurrency: int) -> dict:
    """Launch ``concurrency`` parallel ``interruptible_api_call`` instances
    against simulated agents. Measure heartbeat ticks and per-call latency."""
    agents = [_make_agent(duration_seconds) for _ in range(concurrency)]
    ticks: list = [0]
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_worker, args=(ticks, stop), daemon=True, name="load-heartbeat"
    )
    heartbeat.start()
    time.sleep(0.2)

    ticks[0] = 0
    t0 = time.monotonic()
    results: list = []

    def _worker(idx: int, agent: MagicMock) -> None:
        ct0 = time.monotonic()
        response = cch.interruptible_api_call(agent, {"model": "x", "messages": []})
        ct1 = time.monotonic()
        results.append({
            "call_index": idx,
            "duration_seconds": round(ct1 - ct0, 3),
            "expected_seconds": duration_seconds,
            "overhead_seconds": round((ct1 - ct0) - duration_seconds, 3),
        })

    threads = [
        threading.Thread(target=_worker, args=(i, a), daemon=True, name=f"load-call-{i}")
        for i, a in enumerate(agents)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0
    stop.set()
    heartbeat.join(timeout=2.0)

    expected_total_ticks = elapsed / HEARTBEAT_INTERVAL_SECONDS
    return {
        "concurrency": concurrency,
        "call_duration_seconds": duration_seconds,
        "total_wall_clock_seconds": round(elapsed, 3),
        "heartbeat_ticks": ticks[0],
        "heartbeat_expected_max": round(expected_total_ticks, 1),
        "heartbeat_ratio": round(ticks[0] / expected_total_ticks, 3) if expected_total_ticks > 0 else 0.0,
        "per_call_results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--output", choices=("text", "json"), default="text")
    args = parser.parse_args()

    if args.seconds < 0.5:
        parser.error("--seconds must be >= 0.5")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")

    result = run_load(args.seconds, args.concurrency)

    if args.output == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"=== Concurrent load: {args.concurrency} calls × {args.seconds}s ===")
        print(f"  Wall-clock total: {result['total_wall_clock_seconds']}s")
        print(f"  Heartbeat ticks:  {result['heartbeat_ticks']} "
              f"(max expected {result['heartbeat_expected_max']}, "
              f"ratio {result['heartbeat_ratio']})")
        for r in result["per_call_results"]:
            print(f"  Call {r['call_index']}: {r['duration_seconds']}s "
                  f"(expected {r['expected_seconds']}s, "
                  f"overhead {r['overhead_seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())