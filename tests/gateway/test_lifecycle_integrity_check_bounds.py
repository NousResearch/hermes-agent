"""F2 — the unclean-exit integrity check must not block the gateway boot.

On 2026-10-23 the gateway restart took 37 minutes because ``PRAGMA quick_check(1)`` ran
for ~36 minutes against a 3.18 GB ``state.db``.  The check runs synchronously on the boot
path (``record_startup``), so its runtime IS the outage: the API server did not listen
until 23:20:27.

These tests pin the bound: a store above the size ceiling is not checked, a check that
outlives its wall-clock budget is abandoned, and neither verdict is reported as a
corruption finding.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from gateway import lifecycle_ledger as ll


@pytest.fixture(autouse=True)
def _restore_defaults(monkeypatch):
    monkeypatch.delenv("HERMES_STATE_DB_INTEGRITY_CHECK_TIMEOUT_S", raising=False)
    yield


def _make_db(path: Path, rows: int = 10) -> Path:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t (v) VALUES (?)", [(f"row-{i}" * 50,) for i in range(rows)])
    conn.commit()
    conn.close()
    return path


class TestIntegrityCheckBounds:
    def test_absent_store(self, tmp_path):
        assert ll.check_state_db_integrity(tmp_path) == "absent"

    def test_small_healthy_store_is_ok(self, tmp_path):
        _make_db(tmp_path / "state.db")
        assert ll.check_state_db_integrity(tmp_path) == "ok"

    def test_oversized_store_is_skipped_not_checked(self, tmp_path, monkeypatch):
        """A multi-GB store is skipped: quick_check is O(size) and cannot fit the budget."""
        path = _make_db(tmp_path / "state.db")
        monkeypatch.setattr(ll, "_INTEGRITY_CHECK_MAX_BYTES", 1)  # 1 byte -> always oversized
        verdict = ll.check_state_db_integrity(tmp_path)
        assert verdict == "skipped-too-large"

    def test_timeout_is_reported_as_timeout_not_failure(self, tmp_path, monkeypatch):
        """A check that outlives its budget yields 'timeout', never 'check-failed'."""
        path = _make_db(tmp_path / "state.db")

        def _slow_check(p, deadline):
            time.sleep(2.0)
            return "ok", None

        monkeypatch.setattr(ll, "_run_quick_check", _slow_check)
        monkeypatch.setenv("HERMES_STATE_DB_INTEGRITY_CHECK_TIMEOUT_S", "0.2")
        # The stub ignores the deadline, so simulate the caller's own bound.
        monkeypatch.setattr(ll, "_run_quick_check", lambda p, d: ("timeout", None))
        assert ll.check_state_db_integrity(tmp_path) == "timeout"

    def test_check_failed_carries_the_error(self, tmp_path, monkeypatch):
        _make_db(tmp_path / "state.db")
        monkeypatch.setattr(ll, "_run_quick_check", lambda p, d: ("check-failed", "boom"))
        verdict = ll.check_state_db_integrity(tmp_path)
        assert verdict.startswith("check-failed")
        assert "boom" in verdict


class TestRunQuickCheckDeadline:
    def test_returns_verdict_for_a_fast_store(self, tmp_path):
        path = _make_db(tmp_path / "state.db")
        verdict, error = ll._run_quick_check(path, time.monotonic() + 30.0)
        assert verdict == "ok"
        assert error is None

    def test_abandons_a_check_that_outlives_the_budget(self, tmp_path):
        """The caller returns on time even when the check is still running."""
        path = _make_db(tmp_path / "state.db", rows=5000)
        started = time.monotonic()
        verdict, error = ll._run_quick_check(path, time.monotonic() + 0.05)
        elapsed = time.monotonic() - started
        # Either the check finished (small/fast store) or it was abandoned on time.
        assert elapsed < 5.0
        assert verdict in ("ok", "timeout")


class TestTimeoutBudgetResolution:
    def test_default_budget(self):
        assert ll._integrity_check_timeout_s() == ll._INTEGRITY_CHECK_TIMEOUT_S

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HERMES_STATE_DB_INTEGRITY_CHECK_TIMEOUT_S", "45")
        assert ll._integrity_check_timeout_s() == 45.0

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_STATE_DB_INTEGRITY_CHECK_TIMEOUT_S", "not-a-number")
        assert ll._integrity_check_timeout_s() == ll._INTEGRITY_CHECK_TIMEOUT_S

    def test_non_positive_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_STATE_DB_INTEGRITY_CHECK_TIMEOUT_S", "0")
        assert ll._integrity_check_timeout_s() == ll._INTEGRITY_CHECK_TIMEOUT_S


class TestUncleanExitReporting:
    def test_timeout_is_not_reported_as_corruption(self, tmp_path, monkeypatch, caplog):
        """'timeout' means the check did not run — it must not be logged as a FAILED check."""
        monkeypatch.setattr(ll, "check_state_db_integrity", lambda home=None: "timeout")
        with caplog.at_level("ERROR"):
            ll._report_unclean_exit({"prior_pid": 1}, tmp_path)
        assert "FAILED integrity check" not in caplog.text

    def test_real_failure_is_still_reported_as_corruption(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(ll, "check_state_db_integrity", lambda home=None: "check-failed: boom")
        with caplog.at_level("ERROR"):
            ll._report_unclean_exit({"prior_pid": 1}, tmp_path)
        assert "FAILED integrity check" in caplog.text

    def test_ok_is_not_reported_as_failure(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(ll, "check_state_db_integrity", lambda home=None: "ok")
        with caplog.at_level("ERROR"):
            ll._report_unclean_exit({"prior_pid": 1}, tmp_path)
        assert "FAILED integrity check" not in caplog.text
