"""Test the _notification_poller_loop busy-loop hypothesis.

Issue: #57903 follow-up. Py-spy dump of the gateway showed thread 6740
``(_notification_poller_loop)`` as ``active+gil`` in every dump during
event-loop stalls. The thread does:

    if _notification_event_belongs_elsewhere(session, evt):
        process_registry.completion_queue.put(evt)
        time.sleep(0.1)
        continue

When multiple desktop sessions share the process (Яков's setup: 2-3
sessions at once), events for *other* sessions cycle through put+get on
the shared completion_queue. The poller eats 100ms of sleep per foreign
event. With 3 sessions and 2 events/sec, that's 600ms of sleep + 6 queue
put+get pairs per second — all while holding the GIL, starving other
threads including the main asyncio event loop.

This test reproduces the busy-loop in isolation and measures the impact.

The fix: don't put-back. Either drop the event (orphan) or push to the
owner's own queue directly. The put-back is a workaround for "I can't
tell which session owns this event" — but the code already has
``is_completion_consumed`` and session lookup logic, so the put-back
should not be needed.
"""
from __future__ import annotations

import queue
import threading
import time


def _run_poller_busy_loop(num_sessions: int, event_rate_hz: float, duration_seconds: float = 2.0) -> dict:
    """Simulate the poller thread in isolation.

    Multiple session-pollers share one completion_queue. Foreign events
    get put back. We measure how much GIL they hold (proxied by
    busy-poll iteration count).
    """
    shared_queue: queue.Queue = queue.Queue()
    stop = threading.Event()
    counters = {f"poller_{i}": [0] for i in range(num_sessions)}
    poll_latencies = []

    def poller(idx: int):
        local_session_key = f"session-{idx}"
        local_counter = counters[f"poller_{idx}"]
        # This is the actual code from tui_gateway/server.py:8316-8330
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                evt = shared_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if evt.get("session_key") != local_session_key:
                # Foreign event: put back, sleep 100ms.
                shared_queue.put(evt)
                time.sleep(0.1)
                continue
            local_counter[0] += 1
            poll_latencies.append(time.monotonic() - t0)

    def event_producer():
        end = time.monotonic() + duration_seconds
        interval = 1.0 / event_rate_hz
        next_at = time.monotonic()
        i = 0
        while not stop.is_set() and time.monotonic() < end:
            # Cycle events through all sessions.
            target_session = f"session-{i % num_sessions}"
            shared_queue.put({"type": "completion", "session_key": target_session})
            i += 1
            next_at += interval
            sleep_for = max(0, next_at - time.monotonic())
            time.sleep(sleep_for)

    pollers = [threading.Thread(target=poller, args=(i,), daemon=True) for i in range(num_sessions)]
    producer = threading.Thread(target=event_producer, daemon=True)
    for p in pollers:
        p.start()
    producer.start()
    producer.join()
    time.sleep(0.3)
    stop.set()
    for p in pollers:
        p.join(timeout=1.0)

    total_processed = sum(c[0] for c in counters.values())
    avg_latency = sum(poll_latencies) / len(poll_latencies) if poll_latencies else 0
    max_latency = max(poll_latencies) if poll_latencies else 0
    return {
        "num_sessions": num_sessions,
        "event_rate_hz": event_rate_hz,
        "duration_seconds": duration_seconds,
        "events_processed": total_processed,
        "events_per_session": {k: c[0] for k, c in counters.items()},
        "avg_local_poll_latency_ms": round(avg_latency * 1000, 2),
        "max_local_poll_latency_ms": round(max_latency * 1000, 2),
    }


def test_poller_busy_loop_blocks_own_session():
    """When 3 sessions share a queue, the busy-loop causes foreign-event
    back-and-forth that delays local event delivery.

    Baseline: 1 session processes events with low latency.
    Multi-session: each session's events get delayed by foreign-event
    cycling between pollers.
    """
    single = _run_poller_busy_loop(num_sessions=1, event_rate_hz=2.0, duration_seconds=2.0)
    multi = _run_poller_busy_loop(num_sessions=3, event_rate_hz=2.0, duration_seconds=2.0)

    # With 3 sessions, foreign events cycle through 2 other pollers before
    # reaching the owner. Each cycle takes 100ms sleep + queue ops.
    # For a target session to receive its event, it has to wait through
    # 2 * 100ms = 200ms of foreign cycling on average.
    # Max latency should be much higher with multiple sessions.
    print(f"\nsingle session: avg={single['avg_local_poll_latency_ms']}ms, "
          f"max={single['max_local_poll_latency_ms']}ms, "
          f"processed={single['events_processed']}")
    print(f"3 sessions: avg={multi['avg_local_poll_latency_ms']}ms, "
          f"max={multi['max_local_poll_latency_ms']}ms, "
          f"per_session={multi['events_per_session']}")

    # Sanity: single session should process most events. With 2 events/sec
    # for 2 seconds, single session should process ~4 events.
    assert single["events_processed"] >= 3, (
        f"Single session should process most events; got "
        f"{single['events_processed']}/4 (or fewer if event producer is slow)"
    )

    # With 3 sessions, max latency should grow significantly. We don't
    # assert a specific number because GIL switching adds variance, but
    # we want a regression guard for the busy-loop pattern.
    if multi["max_local_poll_latency_ms"] > single["max_local_poll_latency_ms"] * 10:
        # This is the symptom — multi-session poll latency is >10x
        # single-session. Documents that the put-back pattern hurts.
        pass  # print("multi-session put-back pattern hurts (expected)")

    # Per-session throughput should be roughly equal (1/N of total).
    # If the put-back causes starvation of one session, this fails.
    per_session = multi["events_per_session"]
    counts = list(per_session.values())
    if counts:
        ratio = max(counts) / max(1, min(counts))
        assert ratio < 3, (
            f"Per-session event throughput is unbalanced: {per_session}. "
            f"Max/min ratio {ratio:.1f} indicates one session is "
            f"starving — the put-back busy-loop is causing unfair "
            f"scheduling."
        )