"""Progress-lease renewal during the unclean-exit state.db integrity check (#115542).

After an unclean gateway exit, ``check_state_db_integrity`` runs one monolithic
``PRAGMA quick_check(1)``. On a 37GB store a healthy check takes ~4100s while
per-call watchdog leases clamp at 900s, so without renewal every supervised
startup is killed with exit 75 and the gateway never reaches the cron ticker.
The checker must renew a phase-owned lease for as long as SQLite makes progress,
while preserving the exact ok/absent/first-complaint verdicts fail-closed.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import hermes_startup_watchdog as sw
from hermes_startup_watchdog import (
    SERVICE_RESTART_EXIT_CODE,
    StartupWatchdogHandle,
    arm_startup_watchdog,
    disarm_startup_watchdog,
)

import gateway.lifecycle_ledger as ledger
from gateway.lifecycle_ledger import (
    check_state_db_integrity,
    get_lifecycle_sentinel_path,
    record_startup,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Every test gets a fresh watchdog singleton and its own HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG, raising=False)
    monkeypatch.delenv(sw.ENV_STARTUP_WATCHDOG_TIMEOUT_S, raising=False)
    sw._reset_for_tests()
    yield
    sw._reset_for_tests()


@pytest.fixture(autouse=True)
def _no_cpu_progress(monkeypatch):
    """Freeze process CPU time so only a progress lease can save a slow check."""
    monkeypatch.setattr(
        StartupWatchdogHandle, "_process_cpu_seconds", staticmethod(lambda: 0.0)
    )


class _ExitCapture:
    """Replaces StartupWatchdogHandle._exit so _fire() cannot kill pytest."""

    def __init__(self):
        self.codes: list[int] = []
        self.fired = threading.Event()

    def __call__(self, code: int) -> None:
        self.codes.append(code)
        self.fired.set()


@pytest.fixture
def exit_capture(monkeypatch):
    capture = _ExitCapture()
    monkeypatch.setattr(StartupWatchdogHandle, "_exit", staticmethod(capture))
    return capture


_DEAD_PID = 2 ** 22 + 12345  # beyond default pid_max; never alive


class _SlowConn:
    """sqlite3.Connection stand-in: a slow quick_check honoring the progress handler."""

    def __init__(self, *, ticks=20, tick_s=0.05, verdict="ok", drive_handler=True):
        self._handler = None
        self.ticks = ticks
        self.tick_s = tick_s
        self.verdict = verdict
        self.drive_handler = drive_handler
        self.handler_calls = 0

    def set_progress_handler(self, callback, n):
        self._handler = callback

    def execute(self, sql):
        assert "quick_check" in sql
        for _ in range(self.ticks):
            time.sleep(self.tick_s)
            if self.drive_handler and self._handler is not None:
                self.handler_calls += 1
                assert self._handler() == 0, "progress handler must never abort the check"
        return self

    def fetchone(self):
        return (self.verdict,)

    def close(self):
        pass


def _make_state_db(home: Path, *, corrupt: bool = False) -> Path:
    path = home / "state.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany(
        "INSERT INTO sessions (v) VALUES (?)", [(f"row-{i}" * 40,) for i in range(200)]
    )
    conn.commit()
    conn.close()
    if corrupt:
        with open(path, "r+b") as handle:
            handle.seek(4096 * 6)
            handle.write(b"\xEF" * 4096)
    return path


def _write_sentinel(home: Path, phase: str = "running") -> None:
    path = get_lifecycle_sentinel_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "phase": phase,
            "pid": _DEAD_PID,
            "start_time": 1000.0,
            "started_at": "2026-08-26T23:56:45+00:00",
        }),
        encoding="utf-8",
    )


# ── lease renewal keeps a slow healthy check alive ──────────────────────────


def test_slow_check_renews_lease_and_survives_watchdog(tmp_path, monkeypatch, exit_capture):
    """A slow-but-progressing check repeatedly renews past the deadline (no exit 75)."""
    monkeypatch.setattr(ledger, "_INTEGRITY_CHECK_LEASE_RENEW_S", 0.0)
    (tmp_path / "state.db").touch()
    handle = arm_startup_watchdog(timeout_s=0.2)
    assert handle is not None
    monkeypatch.setattr(
        sqlite3, "connect", lambda *args, **kwargs: _SlowConn(verdict="ok")
    )

    assert check_state_db_integrity(home=tmp_path) == "ok"

    assert handle._lease_count >= 2, "entry lease plus at least one renewal"
    assert handle._lease_phase == ledger._INTEGRITY_CHECK_LEASE_PHASE
    time.sleep(0.4)  # past the original 0.2s deadline with no further work
    assert not exit_capture.fired.is_set()
    disarm_startup_watchdog()


def test_slow_check_without_renewal_fires_watchdog(tmp_path, monkeypatch, exit_capture):
    """Control/mutation: with the lease install disabled (pre-fix behavior), the slow check is killed."""
    (tmp_path / "state.db").touch()
    monkeypatch.setattr(ledger, "_install_integrity_check_lease", lambda conn: None)
    handle = arm_startup_watchdog(timeout_s=0.2)
    assert handle is not None
    monkeypatch.setattr(
        sqlite3, "connect", lambda *args, **kwargs: _SlowConn(verdict="ok")
    )

    assert check_state_db_integrity(home=tmp_path) == "ok"

    assert exit_capture.fired.wait(timeout=10)
    assert exit_capture.codes == [SERVICE_RESTART_EXIT_CODE]


# ── verdicts are preserved exactly ──────────────────────────────────────────


def test_real_check_with_handler_traffic_stays_ok(tmp_path, monkeypatch):
    """The handler fires during a real PRAGMA, returns 0, verdict stays ok."""
    _make_state_db(tmp_path)
    monkeypatch.setattr(ledger, "_INTEGRITY_CHECK_PROGRESS_OPS", 1)
    monkeypatch.setattr(ledger, "_INTEGRITY_CHECK_LEASE_RENEW_S", 0.0)
    assert check_state_db_integrity(home=tmp_path) == "ok"


def test_real_check_still_reports_corruption(tmp_path):
    _make_state_db(tmp_path, corrupt=True)
    verdict = check_state_db_integrity(home=tmp_path)
    assert verdict != "ok"
    assert "btreeInitPage" in verdict or "malformed" in verdict.lower()


def test_missing_store_stays_absent(tmp_path):
    assert check_state_db_integrity(home=tmp_path) == "absent"


def test_connect_failure_stays_failed_closed(tmp_path, monkeypatch):
    (tmp_path / "state.db").touch()

    def _boom(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(sqlite3, "connect", _boom)
    verdict = check_state_db_integrity(home=tmp_path)
    assert verdict.startswith("check-failed: ") and "disk gone" in verdict


# ── wiring: only after an unclean prior life, no worker threads ─────────────


def test_unclean_startup_holds_lease_while_checking(tmp_path, monkeypatch):
    _make_state_db(tmp_path)
    _write_sentinel(tmp_path)
    monkeypatch.setattr(ledger, "_INTEGRITY_CHECK_LEASE_RENEW_S", 0.0)
    handle = arm_startup_watchdog(timeout_s=60)
    assert handle is not None

    evidence = record_startup(home=tmp_path)

    assert evidence is not None
    assert evidence["state_db_integrity"] == "ok"
    assert handle._lease_phase == ledger._INTEGRITY_CHECK_LEASE_PHASE
    assert handle._lease_count >= 1
    disarm_startup_watchdog()


def test_clean_startup_takes_no_lease(tmp_path, monkeypatch):
    _make_state_db(tmp_path, corrupt=True)
    _write_sentinel(tmp_path, phase="exited")
    handle = arm_startup_watchdog(timeout_s=60)
    assert handle is not None

    record_startup(home=tmp_path)

    assert handle._lease_count == 0
    disarm_startup_watchdog()


def test_check_spawns_no_worker_threads(tmp_path):
    """Renewal rides the synchronous PRAGMA — no checker worker may outlive it."""
    _make_state_db(tmp_path)
    before = set(threading.enumerate())
    assert check_state_db_integrity(home=tmp_path) == "ok"
    assert set(threading.enumerate()) <= before
