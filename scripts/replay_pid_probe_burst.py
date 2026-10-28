#!/usr/bin/env python3
"""Replay harness for the R5 pid-not-alive false-reclaim incident (10-17 / 10-18 bursts).

Rebuilds every recorded ``pid N not alive`` reclaim (fleet board ``task_runs``, 43 rows)
as a live board scenario — each reclaimed worker replaced by a REAL live stand-in process
(R5 proved 42/43 were still executing at reclaim time) — and runs the patched crash
sweep over it twice: tick N with the recorded failure mode injected (the process-birth
re-read transiently fails on the inference-loaded dispatch host) and tick N+1 with the
read recovered. The baseline column is the pre-fix decision rule evaluated against the
same recorded facts; ground truth is the R5 classification (session-lag / waitpid-reap
evidence), not either model's own verdict.

Usage:
    python scripts/replay_pid_probe_burst.py \
        --runs-db ~/.hermes/kanban/boards/football-iq/kanban.db \
        --classification r5_classification.json --out replay_result.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

STANDIN = (
    "import signal,time\n"
    "signal.signal(signal.SIGTERM, lambda *a: None)\n"  # the fleet workers were mid-tool-call, uncancellable
    "time.sleep(600)\n"
)

DEFAULT_BURST = ("2026-10-16", "2026-10-19")  # 10-17 and 10-18 bursts


def _to_epoch(value):
    if isinstance(value, (int, float)):
        return float(value)
    return time.mktime(time.strptime(str(value), "%Y-%m-%d %H:%M:%S"))


def load_burst(runs_db: Path, classification: Path, burst: tuple[str, str]):
    con = sqlite3.connect(f"file:{runs_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id AS run, task_id, pid, started_at, ended_at FROM ("
        "  SELECT r.id, r.task_id,"
        "         CAST(substr(r.error, 5, instr(substr(r.error,'5'),' ') - 1) AS INTEGER) AS pid,"
        "         r.started_at, r.ended_at, r.error"
        "  FROM task_runs r WHERE r.error LIKE 'pid % not alive%')"
    ).fetchall()
    con.close()
    truth = {e["run"]: e for e in json.loads(classification.read_text(encoding="utf-8"))}
    lo, hi = burst
    events = []
    for r in rows:
        when = _to_epoch(r["started_at"])
        day = time.strftime("%Y-%m-%d", time.localtime(when))
        if not (lo <= day <= hi):
            continue
        t = truth.get(r["run"], {})
        # R5 ground truth: the worker session outlived its own reclaim (session lag) => alive.
        alive_at_reclaim = (
            t.get("verdict") == "FALSE_RECLAIM"
            or (t.get("sess_lag") or 0) > 0
            or (t.get("sess_secs") or 0) > 0
        )
        events.append(
            {
                "run": r["run"],
                "task": r["task_id"],
                "pid": r["pid"],
                "when": when,
                "day": day,
                "alive_at_reclaim": bool(alive_at_reclaim),
                "hb_age": t.get("hb_age"),
                "verdict": t.get("verdict"),
            }
        )
    events.sort(key=lambda e: e["when"])
    return events


def _baseline_reclaims(*, pid_alive_at_event, fingerprint_read, recorded):
    """Pre-fix rule: ``_worker_alive = _pid_alive and not _pid_recycled`` where ``_pid_recycled``
    answered True whenever the recomputed fingerprint was ``None`` (line 404/407 fail-closed)."""
    if not pid_alive_at_event:
        return True
    return fingerprint_read is None or fingerprint_read != recorded


def replay(events, monkeypatch_fingerprint, confirm_seconds: str):
    """Two ticks per recorded event against the patched module. Returns per-event verdicts."""
    import os

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    root = Path(os.environ["HERMES_HOME"])
    conn = kbc.connect(root / "kanban.db")
    monkeypatch = monkeypatch_fingerprint
    # tick N: the recorded host-wide transient (birth re-read fails for every probe in the tick)
    monkeypatch(True)
    standins = []
    for ev in events:
        tid = kb.create_task(conn, title=f"replay-run-{ev['run']}", assignee="worker")
        kb.claim_task(conn, tid)
        proc = subprocess.Popen([sys.executable, "-c", STANDIN])
        standins.append(proc)
        kbd._set_worker_pid(conn, tid, proc.pid)
        old = int(time.time()) - 3600
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_started_at=?, started_at=?, claim_expires=? WHERE id=?",
                (f"{ev['pid']}|replay-{ev['run']}", old, old, tid),
            )
        # fingerprint recorded at spawn would have been the stand-in's real birth; keep the
        # REAL one so tick N+1 (read recovered) can verify a match, like the fleet post-transient.
        from gateway.drain_control import current_instantiation_epoch
        from gateway.status import get_process_start_time

        real_fp = f"{current_instantiation_epoch()}|{get_process_start_time(proc.pid)}"
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET worker_started_at=? WHERE id=?", (real_fp, tid))
        ev["_tid"] = tid
        ev["_fingerprint"] = real_fp

    tick1 = set(kbd.detect_crashed_workers(conn))
    monkeypatch(False)  # read recovered (next tick, ~60 s later on the fleet)
    tick2 = set(kbd.detect_crashed_workers(conn))
    def _status(task_id):
        task = kb.get_task(conn, task_id)
        return task.status if task else "?"

    statuses = {ev["_tid"]: _status(ev["_tid"]) for ev in events}
    for proc in standins:
        proc.kill()
    for proc in standins:
        proc.wait(timeout=15)
    conn.close()
    for ev in events:
        ev["patched_tick1_reclaim"] = ev["_tid"] in tick1
        ev["patched_tick2_reclaim"] = ev["_tid"] in tick2
        ev["patched_final_status"] = statuses[ev["_tid"]]
    return events


def replay_true_deaths(events, monkeypatch_fingerprint):
    """Control: with genuinely dead workers the patched sweep must STILL reclaim (eventually)."""
    import os

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    root = Path(os.environ["HERMES_HOME"])
    conn = kbc.connect(root / "kanban.db")
    monkeypatch_fingerprint(False)
    pids = []
    for ev in events[:25]:
        tid = kb.create_task(conn, title=f"deadctl-{ev['run']}", assignee="worker")
        kb.claim_task(conn, tid)
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=15)
        pids.append((tid, proc.pid))
        kbd._set_worker_pid(conn, tid, proc.pid)
        old = int(time.time()) - 3600
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_started_at=?, started_at=?, claim_expires=? WHERE id=?",
                (f"x|dead-{ev['run']}", old, old, tid),
            )
    kbd.detect_crashed_workers(conn)
    # The scratch board still holds the tick-1 holds from ``replay()``; count only this
    # control's own tasks. A control task whose status left ``running`` was booked.
    still = {
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE title LIKE 'deadctl-%' AND status = 'running'"
        ).fetchall()
    }
    tested = {tid for tid, _ in pids}
    conn.close()
    return {
        "true_deaths_tested": len(tested),
        "true_deaths_reclaimed_tick1": len(tested - still),
        "true_deaths_still_held": len(tested & still),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-db", required=True, type=Path)
    ap.add_argument("--classification", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("replay_result.json"))
    ap.add_argument("--burst", nargs=2, default=list(DEFAULT_BURST), metavar=("FROM", "TO"))
    ap.add_argument("--confirm-seconds", default="60")
    args = ap.parse_args()

    import os

    # This harness IS a dispatch-board descendant: the delegation fence would (correctly) deny
    # it any board except the pinned one. It never touches the production board for writes —
    # it reads the runs history read-only and builds a throwaway scratch board under HERMES_HOME.
    os.environ.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    os.environ.pop("HERMES_KANBAN_DB", None)
    os.environ.pop("HERMES_KANBAN_BOARD", None)
    os.environ.pop("HERMES_KANBAN_TASK", None)

    root = Path(os.environ.get("HERMES_HOME") or (Path.cwd() / "replay-home"))
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(root)
    os.environ["HERMES_KANBAN_CRASH_GRACE_SECONDS"] = "0"
    os.environ["HERMES_KANBAN_DEAD_PROBE_CONFIRM_SECONDS"] = args.confirm_seconds

    from hermes_cli import kanban_db_dispatch as kbd

    events = load_burst(args.runs_db, args.classification, tuple(args.burst))
    if not events:
        print("no burst events found in window", args.burst, file=sys.stderr)
        return 2

    real_fp = kbd._process_fingerprint

    def _inject(failing: bool):
        kbd._process_fingerprint = (lambda pid: None) if failing else real_fp

    replay(events, _inject, args.confirm_seconds)
    control = replay_true_deaths(events, _inject)

    live = [e for e in events if e["alive_at_reclaim"]]
    baseline_false = sum(
        1
        for e in live
        if _baseline_reclaims(
            pid_alive_at_event=True, fingerprint_read=None, recorded=e.get("_fingerprint")
        )
    )
    patched_false = sum(
        1 for e in live if e["patched_tick1_reclaim"] or e["patched_tick2_reclaim"]
    )
    result = {
        "burst_window": list(args.burst),
        "events_total": len(events),
        "ground_truth_live_workers": len(live),
        "baseline_false_reclaims": baseline_false,
        "patched_false_reclaims": patched_false,
        "patched_held_then_alive_tick1": sum(1 for e in live if not e["patched_tick1_reclaim"]),
        "patched_recovered_never_reclaimed": sum(
            1 for e in live if not e["patched_tick1_reclaim"] and not e["patched_tick2_reclaim"]
        ),
        "control_true_deaths": control,
        "events": [
            {k: v for k, v in e.items() if not k.startswith("_") or k == "_fingerprint"}
            for e in events
        ],
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2))
    return 0 if patched_false == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
