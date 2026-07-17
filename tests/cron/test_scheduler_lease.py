"""Real-process contracts for the HERMES_HOME scheduler ownership lease."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cron.scheduler_lease import SchedulerOwnershipLease

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = r"""
import os, sys, time
from pathlib import Path
from cron.scheduler_lease import SchedulerOwnershipLease
if len(sys.argv) > 3:
    ready, go = Path(sys.argv[3]), Path(sys.argv[4])
    ready.touch()
    while not go.exists():
        time.sleep(0.005)
lease = SchedulerOwnershipLease.try_acquire(
    hermes_home=Path(sys.argv[1]), owner="gateway", provider="builtin"
)
print("acquired" if lease else "blocked", flush=True)
if lease:
    time.sleep(float(sys.argv[2]))
    lease.release()
"""


def _spawn(
    home: Path,
    hold: float = 0.0,
    *,
    ready: Path | None = None,
    go: Path | None = None,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-c", _PROBE, str(home), str(hold)]
        + ([str(ready), str(go)] if ready is not None and go is not None else []),
        cwd=_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _line(proc: subprocess.Popen[str]) -> str:
    assert proc.stdout is not None
    return proc.stdout.readline().strip()


def test_same_home_cross_process_contention_and_graceful_release(tmp_path):
    first = _spawn(tmp_path, 1.0)
    assert _line(first) == "acquired"
    second = _spawn(tmp_path)
    assert second.communicate(timeout=5)[0].strip() == "blocked"
    assert first.wait(timeout=5) == 0
    third = _spawn(tmp_path)
    assert third.communicate(timeout=5)[0].strip() == "acquired"
    assert (tmp_path / "cron" / ".scheduler-owner.lock").exists()


def test_different_homes_can_both_acquire(tmp_path):
    first = _spawn(tmp_path / "a", 0.5)
    second = _spawn(tmp_path / "b", 0.5)
    assert _line(first) == "acquired"
    assert _line(second) == "acquired"
    assert first.wait(timeout=5) == second.wait(timeout=5) == 0


def test_same_process_contention(tmp_path):
    first = SchedulerOwnershipLease.try_acquire(
        hermes_home=tmp_path, owner="gateway", provider="builtin"
    )
    assert first is not None
    try:
        assert SchedulerOwnershipLease.try_acquire(
            hermes_home=tmp_path, owner="desktop", provider="builtin"
        ) is None
    finally:
        first.release()


def test_forced_process_death_recovers_without_unlink(tmp_path):
    child = _spawn(tmp_path, 30.0)
    assert _line(child) == "acquired"
    lock_path = tmp_path / "cron" / ".scheduler-owner.lock"
    inode = lock_path.stat().st_ino
    if os.name == "nt":
        child.kill()
    else:
        child.send_signal(signal.SIGKILL)
    child.wait(timeout=5)

    recovered = SchedulerOwnershipLease.try_acquire(
        hermes_home=tmp_path, owner="desktop", provider="builtin"
    )
    assert recovered is not None
    try:
        assert lock_path.stat().st_ino == inode
    finally:
        recovered.release()


def test_corrupt_stale_metadata_is_diagnostic_only(tmp_path):
    path = tmp_path / "cron" / ".scheduler-owner.lock"
    path.parent.mkdir(parents=True)
    path.write_text("not-json secret-looking-stale-garbage", encoding="utf-8")
    lease = SchedulerOwnershipLease.try_acquire(
        hermes_home=tmp_path, owner="gateway", provider="builtin"
    )
    assert lease is not None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["owner"] == "gateway"
        assert set(payload) == {
            "version", "pid", "process_start_time", "owner", "provider", "acquired_at"
        }
    finally:
        lease.release()


def test_rapid_overlapping_start_has_one_holder(tmp_path):
    go = tmp_path / "go"
    ready_paths = [tmp_path / f"ready-{index}" for index in range(8)]
    children = [
        _spawn(tmp_path, 0.5, ready=ready, go=go) for ready in ready_paths
    ]
    deadline = time.monotonic() + 10
    while not all(path.exists() for path in ready_paths):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    go.touch()
    outputs = [child.communicate(timeout=10)[0].strip() for child in children]
    assert outputs.count("acquired") == 1
    assert outputs.count("blocked") == 7
