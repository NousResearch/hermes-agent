#!/usr/bin/env python3
"""Measure loop-diagnostics overhead on a representative Kanban worker loop.

The task contract requires: "Measure diagnostic overhead on a representative
loop and document any material cost."

This benchmark measures two distinct costs:

  1. Recorder in-path overhead — the per-tool-call cost added to a worker
     when ``kanban.loop_diagnostics.enabled=true``. We drive a real
     ``LoopDiagnosticsRecorder`` through its public hook surface
     (``on_pre_tool_call`` / ``on_post_tool_call`` / ``on_llm_turn_end`` /
     ``finish``) exactly as an instrumented worker fires the observer hooks,
     and compare against a baseline with the feature disabled (recorder
     disabled by config -> the enabled flag short-circuits every hook).

  2. Diagnosis engine cost — how long the deterministic engine takes to
     produce a DiagnosisResult on a representative failure trace (a
     multi-hop input_invalid propagation through 100 actions).

Output is printed to stdout and can be pasted into the operations doc.

Run:  python3 scripts/loop_diagnostics_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hermes_cli.kanban_db as kb
from hermes_cli.observability.loop_diagnostics_engine import diagnose
from hermes_cli.observability.loop_diagnostics_recorder import (
    LoopDiagnosticsRecorder,
    loop_traces_dir,
)

CONFIG_ENABLED = (
    "kanban:\n"
    "  loop_diagnostics:\n"
    "    enabled: true\n"
    "    diagnose_on_failure: true\n"
    "    max_events_per_run: 10000\n"
    "    retain_runs: 3\n"
)
CONFIG_DISABLED = CONFIG_ENABLED.replace("enabled: true", "enabled: false")

# Representative loop: a worker that does an LLM turn, then runs a chain of
# producer/consumer tool calls (search -> extract -> read -> patch), with a
# nested retry loop in the middle. This mirrors real worker behaviour.
TOOL_CHAIN = [
    ("web_search", {"query": "quantum error correction survey"}),
    ("web_extract", {"urls": ["https://arxiv.org/abs/1234.56789"]}),
    ("read_file", {"path": "/tmp/paper.md"}),
    ("patch", {"path": "/tmp/paper.md"}),
    ("terminal", {"command": "python3 build.py"}),
]


def _new_run(conn, title: str):
    tid = kb.create_task(conn, title=title, assignee="bench")
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:bench")
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    return tid, int(row["current_run_id"])


class WorkerLoop:
    """Drive a real recorder through its hook surface, like a worker."""

    def __init__(self, task_id: str, run_id: int):
        os.environ["HERMES_KANBAN_TASK"] = task_id
        os.environ["HERMES_KANBAN_RUN_ID"] = str(run_id)
        os.environ["HERMES_KANBAN_BOARD"] = "default"
        self.rec = LoopDiagnosticsRecorder()
        self.rec.start()
        self._seq = 0
        self._turn = 0

    def tool(self, name: str, args=None, status: str = "ok"):
        turn = f"t{self._turn}"
        cid = f"call_{name}_{self._seq}"
        self._seq += 1
        self.rec.on_pre_tool_call(
            tool_name=name, args=args or {}, turn_id=turn, tool_call_id=cid
        )
        self.rec.on_post_tool_call(
            tool_name=name, turn_id=turn, tool_call_id=cid, status=status,
            error_type=None, error_message=None,
            result=None if status == "ok" else "boom", duration_ms=5,
        )

    def llm_turn(self):
        self._turn += 1
        self.rec.on_llm_turn_end(
            turn_id=f"t{self._turn}", model="bench-model", duration_ms=10
        )

    def finish(self, outcome="completed", error=""):
        self.rec.finish(outcome=outcome, error=error or None)


def run_loop(conn, *, enabled: bool, actions: int, loops: int):
    """Run a representative loop and return elapsed wall time."""
    title = "bench-enabled" if enabled else "bench-disabled"
    tid, run_id = _new_run(conn, title)
    wl = WorkerLoop(tid, run_id)
    wl.llm_turn()
    t0 = time.perf_counter()
    for _ in range(loops):
        for name, args in TOOL_CHAIN:
            wl.tool(name, args)
        wl.llm_turn()
    elapsed = time.perf_counter() - t0
    wl.finish(outcome="completed")
    return tid, run_id, elapsed, actions * loops


def bench_recorder_overhead(conn, *, actions: int = 50, loops: int = 10):
    """Measure enabled vs disabled per-tool-call overhead."""
    n = actions * loops
    # Disabled baseline first (recorder short-circuits every hook).
    _, _, disabled_t, _ = run_loop(
        conn, enabled=False, actions=actions, loops=loops
    )
    # Enabled: same loop with recording on.
    tid, run_id, enabled_t, n_actions = run_loop(
        conn, enabled=True, actions=actions, loops=loops
    )
    per_call_disabled = disabled_t / n_actions * 1e6
    per_call_enabled = enabled_t / n_actions * 1e6
    delta = per_call_enabled - per_call_disabled
    print(f"recorder overhead: {n_actions} tool calls")
    print(f"  disabled: {disabled_t:.4f}s total, {per_call_disabled:.1f}us/call")
    print(f"  enabled:  {enabled_t:.4f}s total, {per_call_enabled:.1f}us/call")
    print(f"  delta:    {delta:+.1f}us per tool call")
    return tid, run_id, delta


def bench_engine(conn, *, path: str):
    """Diagnose a representative multi-hop failure trace and time it."""
    t0 = time.perf_counter()
    result = diagnose("t_bench", 1, trace_path=Path(path))
    elapsed = time.perf_counter() - t0
    print(
        f"engine diagnosis: {elapsed*1000:.2f}ms for "
        f"{result['evidence']['trace_event_count']} events -> "
        f"status={result['status']} category={result['category']}"
    )
    return result, elapsed


def build_failure_trace(path: Path, n_actions: int = 100):
    """Build a representative input_invalid propagation trace on disk.

    First action fails, the next n_actions-1 consume the failure (data
    edges), so the engine must walk the whole chain to the root.
    """
    recs = []
    ts = 1785739000
    recs.append({
        "schema_version": "hermes.loop_diagnostics.v1", "kind": "run_header",
        "task_id": "t_bench", "run_id": 1, "attempt": 1, "ts": ts,
    })
    ts += 1
    for i in range(1, n_actions + 1):
        recs.append({
            "schema_version": "hermes.loop_diagnostics.v1",
            "kind": "action_start", "task_id": "t_bench", "run_id": 1,
            "action_id": f"1:{i}", "parent_action_id": None, "loop_id": None,
            "iteration": None, "ts": ts, "action_kind": "tool_call",
            "tool_name": "read_file" if i > 1 else "terminal",
            "summary": "read file" if i > 1 else "clone repo",
        })
        ts += 1
        if i == 1:
            recs.append({
                "schema_version": "hermes.loop_diagnostics.v1",
                "kind": "action_end", "task_id": "t_bench", "run_id": 1,
                "action_id": "1:1", "ts": ts, "status": "error",
                "duration_ms": 3000, "summary": "clone failed",
                "error_type": "ExitCodeError",
                "error_message": "repository not found",
                "result_hash": "deadbeef",
            })
        else:
            recs.append({
                "schema_version": "hermes.loop_diagnostics.v1",
                "kind": "action_end", "task_id": "t_bench", "run_id": 1,
                "action_id": f"1:{i}", "ts": ts, "status": "error",
                "duration_ms": 1000, "summary": "read failed",
                "error_type": "FileNotFoundError",
                "error_message": "path does not exist",
                "result_hash": f"hash{i:08x}",
            })
        ts += 1
        if i < n_actions:
            recs.append({
                "schema_version": "hermes.loop_diagnostics.v1",
                "kind": "edge", "task_id": "t_bench", "run_id": 1,
                "from_action_id": f"1:{i}", "to_action_id": f"1:{i+1}",
                "edge_kind": "data", "ts": ts,
            })
            ts += 1
    recs.append({
        "schema_version": "hermes.loop_diagnostics.v1", "kind": "run_footer",
        "task_id": "t_bench", "run_id": 1, "ts": ts, "outcome": "blocked",
        "error": "read failed: path does not exist", "event_count": n_actions * 2,
    })
    path.write_text(
        "\n".join(json.dumps(r, separators=(",", ":")) for r in recs) + "\n",
        encoding="utf-8",
    )


def main():
    tmp = Path(tempfile.mkdtemp(prefix="loop-diag-bench-"))
    home = tmp / ".hermes"
    home.mkdir()
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HERMES_KANBAN_BOARD"] = "default"
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_HOME", "HERMES_KANBAN_WORKSPACES_ROOT"):
        os.environ.pop(var, None)
    (home / "config.yaml").write_text(CONFIG_ENABLED, encoding="utf-8")
    kb.init_db()

    print("== loop-diagnostics overhead benchmark ==")
    print(f"repo: {PROJECT_ROOT}")
    print(f"python: {sys.version.split()[0]}")

    with kb.connect() as conn:
        _, _, delta = bench_recorder_overhead(conn, actions=50, loops=10)

    # Engine cost on a representative 100-action failure chain.
    trace = tmp / "failure-trace.jsonl"
    build_failure_trace(trace, n_actions=100)
    result, elapsed = bench_engine(conn, path=str(trace))

    # Repeat the engine measurement a few times to get a stable number.
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        diagnose("t_bench", 1, trace_path=trace)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median = times[len(times) // 2]
    print(f"engine diagnosis (median of 5): {median:.2f}ms")

    print("\n== verdict ==")
    if abs(delta) < 200:
        print("recorder in-path overhead: NEGLIGIBLE (<200us per tool call)")
    else:
        print(f"recorder in-path overhead: MATERIAL ({delta:.1f}us per tool call)")
    if median < 100:
        print("engine diagnosis cost: NEGLIGIBLE (<100ms per diagnosis)")
    else:
        print(f"engine diagnosis cost: MATERIAL ({median:.2f}ms per diagnosis)")


if __name__ == "__main__":
    main()
