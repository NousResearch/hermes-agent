"""Regression tests for #125184: state-layer robustness.

1. ``hermes_state_lockguard.hold()`` used to leave a partially armed guard
   unrecorded: a refused OFD lock (EAGAIN under a foreign EXCLUSIVE holder)
   skipped the range with no log at any level, and an OSError in the arming
   block was DEBUG-only. Both must now surface at WARNING.
2. ``SessionDB._execute_write`` used to swallow a failing ``rollback()`` with
   ``except Exception: pass`` and re-enter the callback inside a possibly-open
   transaction. A failed rollback must now be logged and every retry path must
   refuse to continue — the original error propagates.
"""

import logging
import sqlite3

import pytest

from hermes_state import SessionDB
import hermes_state_lockguard as lg


# ---------------------------------------------------------------------------
# 1. Lockguard arming failures are no longer silent
# ---------------------------------------------------------------------------

class TestLockguardArmingFailureVisibility:
    def test_refused_ofd_lock_logs_warning(self, tmp_path, caplog, monkeypatch):
        """EAGAIN under a foreign EXCLUSIVE holder: the skipped range must be
        reported at WARNING by the completeness check, not silently dropped."""
        lg_mod = lg
        if not lg_mod.supported():
            pytest.skip("OFD locks unavailable on this platform")

        db = tmp_path / "state.db"
        db.touch()
        (tmp_path / "state.db-shm").touch()

        real_ofd_lock = lg_mod._ofd_lock
        refusing_ranges = set()

        def refusing_ofd_lock(fd, lock_type, start, length, *, cmd=None):
            # Refuse the FIRST range armed on this fd, take later ones (releases).
            if cmd is None and (fd, start) not in refusing_ranges:
                refusing_ranges.add((fd, start))
                raise BlockingIOError()  # what fcntl.fcntl raises on EAGAIN
            return real_ofd_lock(fd, lock_type, start, length, cmd=cmd)

        monkeypatch.setattr(lg_mod, "_ofd_lock", refusing_ofd_lock)

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            held = lg_mod.hold(db)
        try:
            assert "incomplete" in caplog.text and "WAL lock guard" in caplog.text
            # The refused range is provably not recorded as held.
            ranges = lg_mod._guard_ranges(db)
            assert len(held) < len(ranges)
        finally:
            lg_mod.release(held)

    def test_arming_oserror_logs_warning_not_debug(self, tmp_path, caplog, monkeypatch):
        """An OSError while arming (e.g. EACCES) must be WARNING, not DEBUG."""
        if not lg.supported():
            pytest.skip("OFD locks unavailable on this platform")

        db = tmp_path / "state.db"
        db.touch()

        def exploding_own_fds(_identities):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(lg, "_own_fds_for", exploding_own_fds)

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            held = lg.hold(db)
        try:
            assert "failed to arm" in caplog.text
            records = [r for r in caplog.records if "failed to arm" in r.getMessage()]
            assert records and records[0].levelno == logging.WARNING
        finally:
            lg.release(held)

    def test_happy_path_hold_stays_silent(self, tmp_path, caplog):
        """No warning when every range arms cleanly (no log-spam regression)."""
        if not lg.supported():
            pytest.skip("OFD locks unavailable on this platform")

        db = tmp_path / "state.db"
        db.touch()
        fd = __import__("os").open(db, __import__("os").O_RDWR)
        try:
            with caplog.at_level(logging.WARNING, logger="hermes_state"):
                held = lg.hold(db)
            try:
                assert held, "expected the opened fd to be armed"
                assert "WAL lock guard" not in caplog.text
            finally:
                lg.release(held)
        finally:
            __import__("os").close(fd)

    def test_strict_raises_when_a_range_is_missing(self, tmp_path, caplog, monkeypatch):
        """strict=True turns the same refusal into WalGuardArmedIncompleteError — the write
        path's hard-fail contract from #125184 — while the WARNING still fires."""
        if not lg.supported():
            pytest.skip("OFD locks unavailable on this platform")

        db = tmp_path / "state.db"
        db.touch()
        (tmp_path / "state.db-shm").touch()

        real_ofd_lock = lg._ofd_lock
        refusing_ranges = set()

        def refusing_ofd_lock(fd, lock_type, start, length, *, cmd=None):
            if cmd is None and (fd, start) not in refusing_ranges:
                refusing_ranges.add((fd, start))
                raise BlockingIOError()
            return real_ofd_lock(fd, lock_type, start, length, cmd=cmd)

        monkeypatch.setattr(lg, "_ofd_lock", refusing_ofd_lock)

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            with pytest.raises(lg.WalGuardArmedIncompleteError, match="refusing to run"):
                lg.hold(db, strict=True)
        assert "WAL lock guard incomplete" in caplog.text

    def test_strict_happy_path_does_not_raise(self, tmp_path):
        """strict=True with every range arming is a plain hold — the normal open path."""
        if not lg.supported():
            pytest.skip("OFD locks unavailable on this platform")

        db = tmp_path / "state.db"
        db.touch()
        import os as _os
        fd = _os.open(db, _os.O_RDWR)
        held: dict = {}
        try:
            held = lg.hold(db, strict=True)
            assert held
        finally:
            lg.release(held)
            _os.close(fd)


# ---------------------------------------------------------------------------
# 2. Failed rollback stops the retry loop instead of being swallowed
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(SessionDB, "_WRITE_PATIENCE_S", 2.0)
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MIN_S", 0.001)
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MAX_S", 0.005)
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


class _RollbackBreakingConn:
    """Proxy over the live connection whose ``rollback`` always raises and whose
    callback raises a transient lock error — the exact #125184 scenario."""

    def __init__(self, conn, lock_error):
        self._conn = conn
        self._lock_error = lock_error

    def execute(self, sql, *a, **kw):
        if str(sql).strip().upper().startswith("BEGIN"):
            return self._conn.execute(sql, *a, **kw)
        return self._conn.execute(sql, *a, **kw)

    def rollback(self):
        raise sqlite3.OperationalError("cannot rollback - no transaction is active")

    def __getattr__(self, name):
        return getattr(self._conn, name)


class TestFailedRollbackStopsRetries:
    def test_lock_error_after_failed_rollback_propagates_without_retry(self, db, monkeypatch, caplog):
        """Callback raises a retriable lock error; rollback also fails. The
        write must fail with the ORIGINAL lock error on the FIRST attempt —
        no re-entry inside a possibly-open transaction."""
        calls = {"n": 0}

        def flaky(conn):
            calls["n"] += 1
            raise sqlite3.OperationalError("database is locked")

        real_conn = db._conn

        def proxying_execute_write(fn, patience_s=None):
            return type(db)._execute_write.__get__(db)(
                lambda conn: fn(_RollbackBreakingConn(real_conn, None)), patience_s=patience_s
            )

        # Simpler: swap the live connection object for the proxy before the write.
        monkeypatch.setattr(db, "_conn", _RollbackBreakingConn(real_conn, None), raising=False)
        # _raise_if_db_replaced and friends read db._conn only via execute(); keep them whole.

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            with pytest.raises(sqlite3.OperationalError) as excinfo:
                db._execute_write(flaky)

        assert calls["n"] == 1, "must not retry after a failed rollback"
        assert "database is locked" in str(excinfo.value)
        assert "rollback failed" in caplog.text

    def test_no_more_rows_after_failed_rollback_propagates_without_retry(self, db, monkeypatch, caplog):
        """Same contract for the 'no more rows' transient class."""
        calls = {"n": 0}

        def flaky(conn):
            calls["n"] += 1
            raise sqlite3.InterfaceError("no more rows available")

        real_conn = db._conn
        monkeypatch.setattr(db, "_conn", _RollbackBreakingConn(real_conn, None), raising=False)

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            with pytest.raises(sqlite3.InterfaceError, match="no more rows"):
                db._execute_write(flaky)

        assert calls["n"] == 1
        assert "rollback failed" in caplog.text
