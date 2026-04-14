#!/usr/bin/env python3
"""Deterministic integration harness for autopilot deferred wake scheduling behavior."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from fixed_now import now_ts

ROOT = Path(__file__).resolve().parents[3]
TICK_SCRIPT = ROOT / "ops" / "autopilot" / "bin" / "hl_autopilot_tick.sh"
DEFAULT_FIXED_NOW_TS = 1773100800


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, env=env, cwd=str(cwd or ROOT), check=False)


def _scenario_now_ts() -> int:
    return now_ts(fallback_ts=DEFAULT_FIXED_NOW_TS)


def _spawn_sleep(seconds: int) -> int:
    cp = _run(
        [
            "bash",
            "-lc",
            f"nohup bash -lc 'sleep {max(1, int(seconds))}' > /dev/null 2>&1 < /dev/null & echo $!",
        ]
    )
    _assert(cp.returncode == 0, f"failed to spawn helper sleep process: {cp.stderr}")
    try:
        return int((cp.stdout or "").strip().splitlines()[-1])
    except Exception as exc:  # pragma: no cover - defensive guard
        raise AssertionError(f"failed to parse helper pid from `{cp.stdout}`: {exc}")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _kill(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return


def _prepare_fake_autopilot(tmp: Path) -> tuple[Path, Path, Path, Path]:
    fake_root = tmp / "sandbox" / "ops" / "autopilot"
    tick_copy = fake_root / "bin" / "hl_autopilot_tick.sh"
    state_path = fake_root / "state" / "hl_terminal_v1.json"
    runs_dir = fake_root / "runs"

    tick_copy.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(TICK_SCRIPT, tick_copy)
    return fake_root, tick_copy, state_path, runs_dir


def scenario_no_eligible_step_arms_deferred_wake(tmp: Path) -> None:
    _, tick_copy, state_path, _ = _prepare_fake_autopilot(tmp)
    now = _scenario_now_ts()
    target_ts = now + 120

    state = {
        "paused": False,
        "cycle": 1,
        "max_cycles": 3,
        "repo": {"path": str(ROOT)},
        "steps": [
            {
                "id": "audit_alignment",
                "title": "Audit alignment",
                "kind": "shell",
                "status": "queued",
                "attempts": 0,
                "max_attempts": 3,
                "cmd": "echo noop",
                "next_after_ts": target_ts,
                "last_error": None,
            }
        ],
        "active": None,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lock_path = tmp / "autopilot_deferred_wake.lock"
    deferred_log = tmp / "deferred_wake.log"
    env = {
        **os.environ,
        "OPENCLAW_ROOT": str(ROOT),
        "OPENCLAW_AUTOPILOT_USE_QUEUE_ARBITRATOR": "0",
        "OPENCLAW_AUTOPILOT_DEFERRED_WAKE_MAX_SEC": "180",
        "OPENCLAW_AUTOPILOT_DEFERRED_WAKE_MIN_SEC": "2",
        "OPENCLAW_AUTOPILOT_DEFERRED_WAKE_LOG": str(deferred_log),
        "OPENCLAW_AUTOPILOT_LOCK_FILE": str(lock_path),
        "OPENCLAW_AUTOPILOT_FIXED_NOW_TS": str(now),
    }

    cp = _run(["bash", str(tick_copy)], env=env, cwd=tmp)
    _assert(cp.returncode == 0, f"tick script failed: {cp.stderr}\n{cp.stdout}")

    updated = json.loads(state_path.read_text(encoding="utf-8"))
    wake = updated.get("deferred_wake")
    _assert(isinstance(wake, dict), f"expected deferred_wake payload in state: {updated}")

    wake_pid = int(wake.get("pid") or 0)
    _assert(wake_pid > 0, f"expected deferred wake pid > 0, got: {wake}")
    _assert(_pid_alive(wake_pid), f"expected deferred wake process to be alive: {wake}")

    _assert(int(wake.get("target_next_after_ts") or 0) == target_ts, f"unexpected target_next_after_ts: {wake}")
    delay_sec = int(wake.get("delay_sec") or 0)
    _assert(2 <= delay_sec <= 180, f"unexpected delay_sec bounds: {wake}")

    due_ts = int(wake.get("due_ts") or 0)
    scheduled_ts = int(wake.get("scheduled_ts") or 0)
    _assert(due_ts >= scheduled_ts + 2, f"expected due_ts >= scheduled_ts + 2: {wake}")

    _kill(wake_pid)


def scenario_launch_clears_existing_deferred_wake(tmp: Path) -> None:
    _, tick_copy, state_path, _ = _prepare_fake_autopilot(tmp)
    stale_wake_pid = _spawn_sleep(120)
    _assert(_pid_alive(stale_wake_pid), "failed to start stale wake helper process")
    now = _scenario_now_ts()

    state = {
        "paused": False,
        "cycle": 1,
        "max_cycles": 3,
        "repo": {"path": str(ROOT)},
        "deferred_wake": {
            "pid": stale_wake_pid,
            "scheduled_ts": now - 10,
            "due_ts": now + 90,
            "delay_sec": 100,
            "target_next_after_ts": now + 90,
            "reason": "stale",
        },
        "steps": [
            {
                "id": "sync_spec_context",
                "title": "Sync spec",
                "kind": "shell",
                "status": "queued",
                "attempts": 0,
                "max_attempts": 3,
                "cmd": "sleep 20",
                "next_after_ts": None,
                "last_error": None,
            }
        ],
        "active": None,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lock_path = tmp / "autopilot_deferred_wake.lock"
    env = {
        **os.environ,
        "OPENCLAW_ROOT": str(ROOT),
        "OPENCLAW_AUTOPILOT_USE_QUEUE_ARBITRATOR": "0",
        "OPENCLAW_AUTOPILOT_LOCK_FILE": str(lock_path),
        "OPENCLAW_AUTOPILOT_FIXED_NOW_TS": str(now),
    }

    active_pid = 0
    try:
        cp = _run(["bash", str(tick_copy)], env=env, cwd=tmp)
        _assert(cp.returncode == 0, f"tick script failed: {cp.stderr}\n{cp.stdout}")

        updated = json.loads(state_path.read_text(encoding="utf-8"))
        _assert("deferred_wake" not in updated, f"expected deferred wake to be cleared after launch: {updated}")

        active = updated.get("active")
        _assert(isinstance(active, dict), f"expected active step after launch: {updated}")
        active_pid = int(active.get("pid") or 0)
        _assert(active_pid > 0, f"expected active pid > 0 after launch: {active}")

        # Best-effort check that stale helper was terminated.
        time.sleep(0.2)
        _assert(not _pid_alive(stale_wake_pid), "expected stale deferred wake helper to be terminated")
    finally:
        _kill(stale_wake_pid)
        _kill(active_pid)


def scenario_dead_wake_rearmed_before_due(tmp: Path) -> None:
    _, tick_copy, state_path, _ = _prepare_fake_autopilot(tmp)

    stale_wake_pid = _spawn_sleep(1)
    time.sleep(1.2)
    _assert(not _pid_alive(stale_wake_pid), "expected stale wake helper process to have exited")

    now = _scenario_now_ts()
    target_ts = now + 90
    state = {
        "paused": False,
        "cycle": 1,
        "max_cycles": 3,
        "repo": {"path": str(ROOT)},
        "deferred_wake": {
            "pid": stale_wake_pid,
            "scheduled_ts": now - 5,
            "due_ts": target_ts,
            "delay_sec": 95,
            "target_next_after_ts": target_ts,
            "reason": "stale_early_exit",
        },
        "steps": [
            {
                "id": "audit_alignment",
                "title": "Audit alignment",
                "kind": "shell",
                "status": "queued",
                "attempts": 0,
                "max_attempts": 3,
                "cmd": "echo noop",
                "next_after_ts": target_ts,
                "last_error": None,
            }
        ],
        "active": None,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lock_path = tmp / "autopilot_deferred_wake.lock"
    env = {
        **os.environ,
        "OPENCLAW_ROOT": str(ROOT),
        "OPENCLAW_AUTOPILOT_USE_QUEUE_ARBITRATOR": "0",
        "OPENCLAW_AUTOPILOT_DEFERRED_WAKE_DRIFT_GRACE_SEC": "5",
        "OPENCLAW_AUTOPILOT_LOCK_FILE": str(lock_path),
        "OPENCLAW_AUTOPILOT_FIXED_NOW_TS": str(now),
    }

    cp = _run(["bash", str(tick_copy)], env=env, cwd=tmp)
    _assert(cp.returncode == 0, f"tick script failed: {cp.stderr}\n{cp.stdout}")

    updated = json.loads(state_path.read_text(encoding="utf-8"))
    wake = updated.get("deferred_wake")
    _assert(isinstance(wake, dict), f"expected deferred wake to be re-armed after early exit: {updated}")
    wake_pid = int(wake.get("pid") or 0)
    _assert(wake_pid > 0 and wake_pid != stale_wake_pid, f"expected newly armed wake pid, got: {wake}")

    meta = updated.get("deferred_wake_meta")
    _assert(isinstance(meta, dict), f"expected deferred_wake_meta in state: {updated}")
    last_event = meta.get("last_event")
    _assert(isinstance(last_event, dict), f"expected deferred_wake_meta.last_event: {meta}")
    _assert(last_event.get("event") == "wake_process_exited_early", f"unexpected early-exit event payload: {last_event}")
    _assert(int(meta.get("early_exit_total") or 0) >= 1, f"expected early_exit_total >= 1: {meta}")

    _kill(wake_pid)


def scenario_overdue_wake_recovers_and_launches(tmp: Path) -> None:
    _, tick_copy, state_path, _ = _prepare_fake_autopilot(tmp)

    stale_wake_pid = _spawn_sleep(120)
    _assert(_pid_alive(stale_wake_pid), "failed to start stale wake helper process")

    now = _scenario_now_ts()
    state = {
        "paused": False,
        "cycle": 1,
        "max_cycles": 3,
        "repo": {"path": str(ROOT)},
        "deferred_wake": {
            "pid": stale_wake_pid,
            "scheduled_ts": now - 120,
            "due_ts": now - 45,
            "delay_sec": 75,
            "target_next_after_ts": now - 45,
            "reason": "overdue_test",
        },
        "steps": [
            {
                "id": "sync_spec_context",
                "title": "Sync spec",
                "kind": "shell",
                "status": "queued",
                "attempts": 0,
                "max_attempts": 3,
                "cmd": "sleep 20",
                "next_after_ts": now - 5,
                "last_error": None,
            }
        ],
        "active": None,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lock_path = tmp / "autopilot_deferred_wake.lock"
    env = {
        **os.environ,
        "OPENCLAW_ROOT": str(ROOT),
        "OPENCLAW_AUTOPILOT_USE_QUEUE_ARBITRATOR": "0",
        "OPENCLAW_AUTOPILOT_DEFERRED_WAKE_DRIFT_GRACE_SEC": "5",
        "OPENCLAW_AUTOPILOT_LOCK_FILE": str(lock_path),
        "OPENCLAW_AUTOPILOT_FIXED_NOW_TS": str(now),
    }

    active_pid = 0
    try:
        cp = _run(["bash", str(tick_copy)], env=env, cwd=tmp)
        _assert(cp.returncode == 0, f"tick script failed: {cp.stderr}\n{cp.stdout}")

        updated = json.loads(state_path.read_text(encoding="utf-8"))
        _assert("deferred_wake" not in updated, f"expected overdue deferred wake to be cleared: {updated}")

        active = updated.get("active")
        _assert(isinstance(active, dict), f"expected active step after overdue wake recovery: {updated}")
        active_pid = int(active.get("pid") or 0)
        _assert(active_pid > 0, f"expected active pid > 0 after overdue wake recovery: {active}")

        meta = updated.get("deferred_wake_meta")
        _assert(isinstance(meta, dict), f"expected deferred_wake_meta in state: {updated}")
        _assert(int(meta.get("missed_recovery_total") or 0) >= 1, f"expected missed recovery counter increment: {meta}")

        last_event = meta.get("last_event")
        _assert(isinstance(last_event, dict), f"expected deferred_wake_meta.last_event: {meta}")
        _assert(last_event.get("event") == "missed_wake_recovered", f"unexpected missed wake event: {last_event}")
        _assert(int(last_event.get("drift_sec") or 0) >= 5, f"expected drift_sec >= 5 for recovered overdue wake: {last_event}")

        time.sleep(0.2)
        _assert(not _pid_alive(stale_wake_pid), "expected overdue deferred wake helper to be terminated")
    finally:
        _kill(stale_wake_pid)
        _kill(active_pid)


SCENARIOS = [
    ("no_eligible_step_arms_deferred_wake", scenario_no_eligible_step_arms_deferred_wake),
    ("launch_clears_existing_deferred_wake", scenario_launch_clears_existing_deferred_wake),
    ("dead_wake_rearmed_before_due", scenario_dead_wake_rearmed_before_due),
    ("overdue_wake_recovers_and_launches", scenario_overdue_wake_recovers_and_launches),
]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="autopilot_tick_deferred_wake_") as td:
        tmp = Path(td)
        passed = 0
        for name, fn in SCENARIOS:
            fn(tmp)
            passed += 1
            print(f"PASS: {name}")
    print(f"SUMMARY: {passed}/{len(SCENARIOS)} scenarios passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
