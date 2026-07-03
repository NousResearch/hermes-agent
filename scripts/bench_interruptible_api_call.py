"""Diagnostic for issue #57903: interruptible_api_call main-thread blocking.

The non-streaming ``interruptible_api_call`` in
``agent/chat_completion_helpers.py`` runs the SDK call in a worker
thread and busy-polls ``t.join(timeout=0.3)`` on the main thread. The
busy-poll does release the GIL while the join is blocked (Python's
``_Condition.wait`` releases the GIL on Windows), so other Python
threads continue to make progress. The actual harm is to the asyncio
event loop running on the same thread (uvicorn + tui_gateway): the
sync ``t.join(timeout=0.3)`` blocks the event loop from servicing
its 2s heartbeat and the desktop's WebSocket heartbeats, which trips
the 5s-stall watchdog and the desktop's 10s WS timeout.

This script measures two things:
  1. **GIL yield**: how many times a parallel heartbeat thread gets to
     tick while ``interruptible_api_call`` is blocked. Confirms the
     current code is GIL-friendly (Python releases the GIL inside
     ``t.join``).
  2. **Interrupt latency**: how quickly ``interruptible_api_call``
     returns after ``_interrupt_requested`` is set during a long SDK
     call. The current 300ms poll caps this at ~300ms plus socket
     teardown time.

Together these two measurements characterize the *real* symptom
(event-loop starvation) by ruling out the *plausible* symptoms
(GIL starvation, slow interrupt detection). The fix — replacing
the sync ``t.join`` busy-poll with an async-aware bridge that lets the
event loop run between waits — needs a different kind of measurement
(see the tests in ``tests/agent/test_interruptible_api_call_yields_gil.py``
for an end-to-end regression guard).

Usage::

    python scripts/bench_interruptible_api_call.py --seconds 10

Exit code 0 = healthy on both metrics; 1 = interrupt latency > 500ms
(probably the 300ms poll is the bottleneck); 2 = GIL yield below
floor (probably a GIL regression introduced by future changes).
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


HEARTBEAT_INTERVAL_SECONDS = 0.1
INTERRUPT_LATENCY_BUDGET_SECONDS = 0.5  # 300ms poll + ~100ms socket teardown + CI headroom
GIL_YIELD_FLOOR_TICKS_PER_SECOND = 8.0  # current code yields ~10/s; threshold = 8/s


def _make_slow_agent(duration_seconds: float) -> tuple[MagicMock, list]:
    """Agent whose SDK call sleeps for ``duration_seconds`` and returns a
    minimal chat-completions response. The second list element is a
    1-element scratchpad the caller can use to record timestamps.
    """
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
    return agent, [None]


def _make_interrupting_agent(duration_seconds: float) -> MagicMock:
    """Agent whose SDK call flips ``_interrupt_requested`` mid-flight and
    raises a transport error to simulate the force-close.
    """
    import httpx

    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent._compute_non_stream_stale_timeout.return_value = 60.0

    fake_client = MagicMock()

    def _slow_create(**_kwargs):
        time.sleep(duration_seconds)
        agent._interrupt_requested = True
        time.sleep(0.05)
        raise httpx.RemoteProtocolError("forced close (bench)")

    fake_client.chat.completions.create.side_effect = _slow_create
    agent._create_request_openai_client.return_value = fake_client
    agent._close_request_openai_client = MagicMock()
    agent._abort_request_openai_client = MagicMock()
    return agent


def _heartbeat_worker(counter: list, interval: float, stop: threading.Event) -> None:
    deadline = time.monotonic() + 60.0
    while not stop.is_set() and time.monotonic() < deadline:
        counter[0] += 1
        time.sleep(interval)


def measure_gil_yield(duration_seconds: float) -> dict:
    """Run a long SDK call and count parallel heartbeat ticks."""
    agent, _ = _make_slow_agent(duration_seconds)
    ticks: list = [0]
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_worker,
        args=(ticks, HEARTBEAT_INTERVAL_SECONDS, stop),
        daemon=True,
        name="bench-heartbeat",
    )
    heartbeat.start()
    time.sleep(0.2)
    ticks[0] = 0
    t0 = time.monotonic()
    response = cch.interruptible_api_call(agent, {"model": "x", "messages": []})
    elapsed = time.monotonic() - t0
    stop.set()
    heartbeat.join(timeout=2.0)
    assert getattr(response, "choices", None) is not None
    return {
        "call_duration_seconds": round(duration_seconds, 2),
        "wall_clock_seconds": round(elapsed, 3),
        "heartbeat_ticks": ticks[0],
        "ticks_per_second": round(ticks[0] / elapsed, 2) if elapsed > 0 else 0.0,
        "expected_perfect_ticks": round(duration_seconds / HEARTBEAT_INTERVAL_SECONDS, 1),
    }


def measure_interrupt_latency(call_setup_seconds: float) -> dict:
    """Run an SDK call that flips ``_interrupt_requested`` mid-flight and
    measure wall-clock time from flip to ``interruptible_api_call``
    return.
    """
    agent = _make_interrupting_agent(call_setup_seconds)
    flag_holder: list = [None]

    real_slow_create = agent._create_request_openai_client.return_value.chat.completions.create.side_effect

    def _timed_create(**kwargs):
        time.sleep(call_setup_seconds)
        flag_holder[0] = time.monotonic()
        agent._interrupt_requested = True
        time.sleep(0.05)
        import httpx
        raise httpx.RemoteProtocolError("forced close (bench)")

    agent._create_request_openai_client.return_value.chat.completions.create.side_effect = _timed_create

    t0 = time.monotonic()
    try:
        cch.interruptible_api_call(agent, {"model": "x", "messages": []})
    except Exception:
        pass
    elapsed = time.monotonic() - t0

    assert flag_holder[0] is not None, "interrupt flag was never set"
    interrupt_to_return = (t0 + elapsed) - flag_holder[0]
    return {
        "call_setup_seconds": round(call_setup_seconds, 2),
        "wall_clock_seconds": round(elapsed, 3),
        "interrupt_to_return_seconds": round(interrupt_to_return, 3),
        "budget_seconds": INTERRUPT_LATENCY_BUDGET_SECONDS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=8.0,
                        help="Duration of the simulated LLM call (default: 8s).")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    args = parser.parse_args()
    if args.seconds < 0.5:
        parser.error("--seconds must be >= 0.5")

    gil = measure_gil_yield(args.seconds)
    intr = measure_interrupt_latency(args.seconds)

    result = {"gil_yield": gil, "interrupt_latency": intr}

    if args.output == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"=== GIL yield during {args.seconds}s simulated call ===")
        print(f"  Heartbeat ticks: {gil['heartbeat_ticks']} "
              f"({gil['ticks_per_second']}/s, theoretical max {gil['expected_perfect_ticks']})")
        print(f"=== Interrupt latency ===")
        print(f"  Wall-clock from flag flip to return: {intr['interrupt_to_return_seconds']}s "
              f"(budget {intr['budget_seconds']}s)")
        print()

    # Pass/fail: interrupt latency is the actionable metric.
    intr_ok = intr["interrupt_to_return_seconds"] <= INTERRUPT_LATENCY_BUDGET_SECONDS
    gil_ok = gil["ticks_per_second"] >= GIL_YIELD_FLOOR_TICKS_PER_SECOND
    if intr_ok and gil_ok:
        return 0
    if not intr_ok:
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
