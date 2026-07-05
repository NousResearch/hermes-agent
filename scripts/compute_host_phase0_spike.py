#!/usr/bin/env python3
"""Run the Phase-0 dashboard compute-host spike.

The spike measures the process-boundary costs called out in
``docs/desktop/2026-07-04-dashboard-process-isolation-PRD.md`` without touching
live dashboard state. It spawns ``python -m tui_gateway.compute_host``, seeds two
sessions, drives three turns per session, and records relay, RSS, cold-start,
interrupt, and pipe-HOL numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from tui_gateway.compute_host import SpikeAgent, now_ns


REPO_ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def rss_mb(pid: int) -> float:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not out:
            return 0.0
        return int(out.splitlines()[-1].strip()) / 1024.0
    except Exception:
        return 0.0


class HostProcess:
    def __init__(self) -> None:
        self.proc: subprocess.Popen[str] | None = None
        self.cold_start_ms = 0.0
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._backlog: list[dict[str, Any]] = []
        self._sid_queues: dict[str, queue.Queue[dict[str, Any]]] = defaultdict(lambda: queue.Queue(maxsize=16))
        self.sid_drops: dict[str, int] = defaultdict(int)
        self.stderr_tail: list[str] = []
        self.hello: dict[str, Any] = {}

    @property
    def pid(self) -> int:
        if self.proc is None or self.proc.pid is None:
            return 0
        return self.proc.pid

    def __enter__(self) -> "HostProcess":
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, "-m", "tui_gateway.compute_host"]
        start = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._drain_stdout, name="compute-host-stdout", daemon=True).start()
        threading.Thread(target=self._drain_stderr, name="compute-host-stderr", daemon=True).start()
        self.hello = self.wait_for(lambda ev: ev.get("type") == "hello", timeout=5.0)
        self.cold_start_ms = (time.perf_counter() - start) * 1000.0
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.send({"type": "shutdown", "request_id": "shutdown"})
                self.proc.wait(timeout=2.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        finally:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass

    def queue_for(self, sid: str) -> queue.Queue[dict[str, Any]]:
        return self._sid_queues[sid]

    def _drain_stdout(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in self.proc.stdout:
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(frame, dict):
                continue
            frame["_client_received_ns"] = now_ns()
            sid = frame.get("sid")
            if frame.get("type") == "delta" and isinstance(sid, str) and sid:
                try:
                    self._sid_queues[sid].put_nowait(frame)
                except queue.Full:
                    self.sid_drops[sid] += 1
            self._events.put(frame)

    def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for raw in self.proc.stderr:
            text = raw.rstrip("\n")
            if text:
                self.stderr_tail = (self.stderr_tail + [text])[-80:]

    def send(self, frame: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None or self.proc.poll() is not None:
            raise RuntimeError(f"compute host is not running; stderr={self.stderr_tail[-5:]}")
        self.proc.stdin.write(json.dumps(frame, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def wait_for(self, predicate: Callable[[dict[str, Any]], bool], *, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            for idx, event in enumerate(self._backlog):
                if predicate(event):
                    return self._backlog.pop(idx)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for host event; stderr={self.stderr_tail[-5:]}")
            event = self._events.get(timeout=remaining)
            if predicate(event):
                return event
            self._backlog.append(event)

    def seed(self, sid: str) -> None:
        rid = f"seed-{sid}"
        self.send({"type": "session.seed", "sid": sid, "request_id": rid, "history": []})
        self.wait_for(
            lambda ev: ev.get("type") == "session.seeded" and ev.get("request_id") == rid,
            timeout=2.0,
        )

    def start_turn(self, sid: str, prompt: str, *, delta_count: int, delay_s: float) -> str:
        rid = uuid.uuid4().hex
        self.send(
            {
                "type": "turn.start",
                "sid": sid,
                "request_id": rid,
                "prompt": prompt,
                "delta_count": delta_count,
                "delay_s": delay_s,
            }
        )
        return rid

    def collect_turn(self, rid: str, *, timeout: float = 10.0) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            event = self.wait_for(
                lambda ev: ev.get("request_id") == rid
                and ev.get("type") in {"turn.started", "delta", "turn.end", "turn.error"},
                timeout=timeout,
            )
            events.append(event)
            if event.get("type") in {"turn.end", "turn.error"}:
                return events


def relay_ms(events: list[dict[str, Any]]) -> list[float]:
    return [
        (ev["_client_received_ns"] - ev["emitted_ns"]) / 1_000_000.0
        for ev in events
        if ev.get("type") == "delta" and "emitted_ns" in ev
    ]


def run_direct_turn(agent: SpikeAgent, prompt: str, *, delta_count: int, delay_s: float) -> list[float]:
    samples: list[float] = []

    def stream(_delta: str) -> None:
        emitted = now_ns()
        samples.append((now_ns() - emitted) / 1_000_000.0)

    agent.run_conversation(
        prompt,
        conversation_history=list(agent.history),
        stream_callback=stream,
        delta_count=delta_count,
        delay_s=delay_s,
    )
    return samples


def measure_direct_interrupt() -> float:
    agent = SpikeAgent("direct-interrupt")
    first_delta = threading.Event()
    done = threading.Event()

    def stream(_delta: str) -> None:
        first_delta.set()

    def run() -> None:
        try:
            agent.run_conversation(
                "interrupt-heavy",
                conversation_history=[],
                stream_callback=stream,
                delta_count=1000,
                delay_s=0.005,
            )
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    if not first_delta.wait(2.0):
        raise TimeoutError("direct interrupt baseline did not stream")
    start = now_ns()
    agent.interrupt()
    if not done.wait(2.0):
        raise TimeoutError("direct interrupt baseline did not stop")
    return (now_ns() - start) / 1_000_000.0


def measure_host_interrupt(host: HostProcess) -> tuple[float, float]:
    sid = "host-interrupt"
    host.seed(sid)
    rid = host.start_turn(sid, "interrupt-heavy", delta_count=1000, delay_s=0.005)
    host.wait_for(
        lambda ev: ev.get("request_id") == rid and ev.get("type") == "delta",
        timeout=2.0,
    )
    interrupt_rid = f"interrupt-{rid}"
    start_ns = now_ns()
    host.send({"type": "interrupt", "sid": sid, "request_id": interrupt_rid})
    ack_ms = None
    end_ms = None
    while ack_ms is None or end_ms is None:
        event = host.wait_for(
            lambda ev: (
                ev.get("request_id") == interrupt_rid and ev.get("type") == "interrupt.ack"
            )
            or (ev.get("request_id") == rid and ev.get("type") in {"turn.end", "turn.error"}),
            timeout=2.0,
        )
        if event.get("type") == "interrupt.ack":
            ack_ms = (event["_client_received_ns"] - start_ns) / 1_000_000.0
        elif event.get("type") in {"turn.end", "turn.error"}:
            end_ms = (event["_client_received_ns"] - start_ns) / 1_000_000.0
    return float(ack_ms), float(end_ms)


def measure_hol(host: HostProcess) -> dict[str, float | int]:
    host.seed("hol-fast-solo")
    solo_rid = host.start_turn("hol-fast-solo", "streaming-heavy", delta_count=220, delay_s=0.0005)
    solo_events = host.collect_turn(solo_rid, timeout=5.0)
    solo_p99 = percentile(relay_ms(solo_events), 99)

    host.seed("hol-slow")
    host.seed("hol-fast")
    slow_queue = host.queue_for("hol-slow")
    stop_slow = threading.Event()

    def slow_consumer() -> None:
        while not stop_slow.is_set():
            try:
                slow_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            time.sleep(0.02)

    threading.Thread(target=slow_consumer, name="slow-ws-consumer", daemon=True).start()
    slow_rid = host.start_turn("hol-slow", "streaming-heavy", delta_count=220, delay_s=0.0005)
    fast_rid = host.start_turn("hol-fast", "streaming-heavy", delta_count=220, delay_s=0.0005)
    events: list[dict[str, Any]] = []
    done: set[str] = set()
    while done != {slow_rid, fast_rid}:
        event = host.wait_for(
            lambda ev: ev.get("request_id") in {slow_rid, fast_rid}
            and ev.get("type") in {"turn.started", "delta", "turn.end", "turn.error"},
            timeout=5.0,
        )
        events.append(event)
        if event.get("type") in {"turn.end", "turn.error"}:
            done.add(str(event.get("request_id")))
    stop_slow.set()
    fast_events = [ev for ev in events if ev.get("request_id") == fast_rid]
    fast_p99 = percentile(relay_ms(fast_events), 99)
    return {
        "fast_solo_p99_ms": solo_p99,
        "fast_with_slow_consumer_p99_ms": fast_p99,
        "fast_p99_delta_ms": fast_p99 - solo_p99,
        "slow_consumer_dropped_deltas": host.sid_drops.get("hol-slow", 0),
    }


def run_spike() -> dict[str, Any]:
    parent_rss = rss_mb(os.getpid())
    direct_agents = {sid: SpikeAgent(f"direct-{sid}") for sid in ("alpha", "bravo")}
    turn_plan = [
        ("short-1", 24, 0.0005),
        ("streaming-heavy", 260, 0.001),
        ("short-2", 24, 0.0005),
    ]
    direct_samples: list[float] = []
    for sid, agent in direct_agents.items():
        for prompt, delta_count, delay_s in turn_plan:
            direct_samples.extend(
                run_direct_turn(
                    agent,
                    f"{sid}-{prompt}",
                    delta_count=delta_count,
                    delay_s=delay_s,
                )
            )

    with HostProcess() as host:
        for sid in ("alpha", "bravo"):
            host.seed(sid)
        host_rss_after_seed = rss_mb(host.pid)
        host_samples: list[float] = []
        for sid in ("alpha", "bravo"):
            for prompt, delta_count, delay_s in turn_plan:
                rid = host.start_turn(
                    sid,
                    f"{sid}-{prompt}",
                    delta_count=delta_count,
                    delay_s=delay_s,
                )
                host_samples.extend(relay_ms(host.collect_turn(rid, timeout=10.0)))
        host_rss_after_turns = rss_mb(host.pid)
        direct_interrupt_ms = measure_direct_interrupt()
        host_interrupt_ack_ms, host_interrupt_end_ms = measure_host_interrupt(host)
        hol = measure_hol(host)
        host_rss_peak = max(host_rss_after_seed, host_rss_after_turns, rss_mb(host.pid))

        direct_p99 = percentile(direct_samples, 99)
        host_p99 = percentile(host_samples, 99)
        interrupt_ratio = host_interrupt_end_ms / max(direct_interrupt_ms, 0.001)
        metrics: dict[str, Any] = {
            "sessions_seeded": 2,
            "turns_per_session": 3,
            "streaming_heavy_turns": 2,
            "delta_samples": len(host_samples),
            "direct_delta_relay_p99_ms": direct_p99,
            "host_delta_relay_p99_ms": host_p99,
            "delta_relay_overhead_p99_ms": host_p99 - direct_p99,
            "baseline_driver_rss_mb": parent_rss,
            "host_rss_peak_mb": host_rss_peak,
            "host_rss_delta_vs_baseline_mb": host_rss_peak - parent_rss,
            "spawn_cold_start_ms": host.cold_start_ms,
            "direct_interrupt_end_ms": direct_interrupt_ms,
            "host_interrupt_ack_ms": host_interrupt_ack_ms,
            "host_interrupt_end_ms": host_interrupt_end_ms,
            "interrupt_end_ratio_vs_direct": interrupt_ratio,
            "hol_isolation": hol,
        }
        metrics["stop_conditions"] = {
            "delta_relay_p99_over_50ms": metrics["delta_relay_overhead_p99_ms"] > 50.0,
            "interrupt_over_2x_in_process": interrupt_ratio > 2.0,
            "cold_start_over_5s": host.cold_start_ms > 5000.0,
        }
        metrics["passed_stop_conditions"] = not any(metrics["stop_conditions"].values())
        return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the dashboard compute-host Phase-0 spike")
    parser.add_argument("--json-out", type=Path, help="optional path to write the JSON metrics")
    args = parser.parse_args(argv)

    metrics = run_spike()
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if metrics.get("passed_stop_conditions") else 1


if __name__ == "__main__":
    raise SystemExit(main())
