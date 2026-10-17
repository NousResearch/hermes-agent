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

# ---------------------------------------------------------------------------
# 3. Review follow-ups (andrexibiza, both P1): strict failure must unwind
#    partial guard ownership; a failed rollback must retire the connection
#    instead of leaving an ambiguous-transaction handle installed.
# ---------------------------------------------------------------------------

class TestStrictFailureUnwindsPartialOwnership:
    def test_strict_raise_restores_handles_baseline_and_releases(self, tmp_path, monkeypatch):
        """First identity arms, second refuses, strict raises: _HANDLES returns
        to baseline and a subsequent full hold/release leaves no residue."""
        if not lg.supported():
            pytest.skip("OFD locks unavailable on this platform")

        db = tmp_path / "state.db"
        db.touch()
        (tmp_path / "state.db-shm").touch()
        os_mod = __import__("os")
        fd = os_mod.open(db, os_mod.O_RDWR)  # descriptors the guard can see
        fd_shm = os_mod.open(tmp_path / "state.db-shm", os_mod.O_RDWR)
        try:
            real_ofd_lock = lg._ofd_lock

            def refuse_shm_range(fd_, lock_type, start, length, *, cmd=None):
                # Deterministic two-identity split: the main-file range arms,
                # the -shm DMS byte refuses (EAGAIN under a foreign EXCLUSIVE
                # holder -> _ofd_lock returns False and the range is skipped).
                if cmd is None and start == lg._SHM_DMS_BYTE:
                    raise BlockingIOError()
                return real_ofd_lock(fd_, lock_type, start, length, cmd=cmd)

            monkeypatch.setattr(lg, "_ofd_lock", refuse_shm_range)

            baseline = dict(lg._HANDLES)
            ranges = lg._guard_ranges(db)
            assert len(ranges) == 2, "fixture must present both guard identities"

            with pytest.raises(lg.WalGuardArmedIncompleteError):
                lg.hold(db, strict=True)

            assert dict(lg._HANDLES) == baseline, "strict failure leaked a handle count"

            # A later real hold/release cycle must leave no residual count either
            # (the refusal is per-range, so the main-file identity still arms).
            held = lg.hold(db, strict=False)
            try:
                assert held
            finally:
                lg.release(held)
            assert dict(lg._HANDLES) == baseline
        finally:
            os_mod.close(fd)
            os_mod.close(fd_shm)


class TestFailedRollbackRetiresConnection:
    def test_second_write_after_failed_rollback_uses_fresh_connection(self, db, monkeypatch, caplog):
        """Two-call witness: the first write hits a failed rollback and must
        fail with the ORIGINAL error; the second write on the same SessionDB
        goes through a proven-fresh connection — never a 'cannot start a
        transaction within a transaction' inherited from the ambiguous one."""
        calls = {"n": 0}

        def flaky(conn):
            calls["n"] += 1
            raise sqlite3.OperationalError("database is locked")

        real_conn = db._conn
        monkeypatch.setattr(db, "_conn", _RollbackBreakingConn(real_conn, None), raising=False)

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                db._execute_write(flaky)

        assert calls["n"] == 1
        assert "rollback failed" in caplog.text
        # The ambiguous-transaction connection was retired, not left installed:
        # the handle holds no connection until the next write reopens one.
        assert db._conn is None

        # Second write: fresh connection, clean BEGIN IMMEDIATE, real commit —
        # never an inherited "cannot start a transaction within a transaction".
        db._execute_write(
            lambda conn: conn.execute(
                "CREATE TABLE IF NOT EXISTS _witness(v TEXT)"
            ) and conn.execute("INSERT INTO _witness VALUES ('after-retire')")
        )
        assert db._conn is not None and db._conn is not real_conn
        row = db._conn.execute("SELECT v FROM _witness").fetchone()
        assert row is not None and row[0] == "after-retire"

    def test_failed_rollback_can_reopen_clean_wal(self, db, monkeypatch, tmp_path):
        """#125184 R1 witness (review-supplied): after a failed-rollback retire that
        closed the sole WAL holder, the sidecar identity must be cleared with the
        connection — the next write re-adopts the on-disk generation instead of
        misclassifying our own clean close as a deleted WAL generation."""
        if not db._wal_active:
            pytest.skip("this regression must actually exercise WAL (this dev box "
                        "forces journal_mode=DELETE; the CI lane with SQLite >=3.51.3 "
                        "covers the WAL-open path)")
        while db._evict_one_idle_read_conn():
            pass
        assert db._db_sidecar_identity
        monkeypatch.setattr(
            db, "_conn", _RollbackBreakingConn(db._conn, None)
        )

        def fail(conn):
            conn.execute(
                "INSERT INTO state_meta(key, value) VALUES (?, ?)",
                ("rollback_reopen_probe", "must-rollback"),
            )
            raise sqlite3.OperationalError("database is locked")

        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            db._execute_write(fail)
        assert db._conn is None
        # The retire cleared the recorded sidecar identity: no false generation loss.
        db.set_meta("rollback_reopen_probe", "after-retire")
        assert db.get_meta("rollback_reopen_probe") == "after-retire"
        assert not db._db_wal_generation_lost
        # Control: with the identity (incorrectly) retained, the same sequence WOULD
        # trip DeletedWalGenerationError — pinned by monkeypatching the clear away.
        db2 = SessionDB(db_path=tmp_path / "state2.db")
        try:
            if not db2._wal_active:
                pytest.skip("second instance also not WAL on this runtime")
            while db2._evict_one_idle_read_conn():
                pass
            monkeypatch.setattr(
                db2, "_conn", _RollbackBreakingConn(db2._conn, None)
            )
            retained = dict(db2._db_sidecar_identity)
            real_retire = type(db2)._retire_connection_locked

            def retain_identity_retire(self, conn=None):
                real_retire(self)
                # Simulate the pre-R1 behavior: identity survives the retire.
                self._db_sidecar_identity = retained

            monkeypatch.setattr(SessionDB, "_retire_connection_locked", retain_identity_retire)
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                db2._execute_write(
                    lambda conn: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked"))
                )
            from hermes_state import DeletedWalGenerationError
            with pytest.raises(DeletedWalGenerationError):
                db2.set_meta("rollback_reopen_probe", "never")
        finally:
            db2.close()


class TestStrictReopenFailureIsNotReusable:
    """#125184 R2 witness (review-supplied): a strict hold() that raises during the
    reopen must not leave a usable unguarded writer installed — the refusal repeats
    while the fault is installed and the handle recovers once it clears."""

    def _refusing_ofd_lock(self, monkeypatch):
        real_ofd_lock = lg._ofd_lock

        def refuse(fd, lock_type, start, length, *, cmd=None):
            if cmd is None and lock_type == lg._F_RDLCK:
                return False
            return real_ofd_lock(fd, lock_type, start, length, cmd=cmd)

        monkeypatch.setattr(lg, "_ofd_lock", refuse)

    def test_strict_reopen_failure_is_not_reusable(self, db, monkeypatch):
        if not (db._wal_active and lg.supported()):
            pytest.skip("requires WAL + OFD guard support (CI lane covers it)")
        assert not db._wal_guard_degraded
        db.close()
        real_ofd_lock = lg._ofd_lock

        def refuse(fd, lock_type, start, length, *, cmd=None):
            if cmd is None and lock_type == lg._F_RDLCK:
                return False
            return real_ofd_lock(fd, lock_type, start, length, cmd=cmd)

        monkeypatch.setattr(lg, "_ofd_lock", refuse)
        outcomes = []
        for _ in range(2):
            try:
                db.set_meta("must_not_commit", "unguarded-write")
            except lg.WalGuardArmedIncompleteError:
                outcomes.append("refused")
            else:
                outcomes.append("COMMITTED")
        assert outcomes == ["refused", "refused"]
        assert db._conn is None
        # Recovery: with the fault cleared, the next write re-establishes the
        # full open+guard pair instead of staying refused.
        monkeypatch.setattr(lg, "_ofd_lock", real_ofd_lock)
        db.set_meta("must_not_commit", "recovered-write")
        assert db.get_meta("must_not_commit") == "recovered-write"
        assert db._conn is not None

    def test_strict_reopen_failure_oserror_also_retires(self, db, monkeypatch):
        """The OSError variant of the arming fault retires the candidate too."""
        if not (db._wal_active and lg.supported()):
            pytest.skip("requires WAL + OFD guard support (CI lane covers it)")
        db.close()
        real_ofd_lock = lg._ofd_lock

        def exploding(fd, lock_type, start, length, *, cmd=None):
            if cmd is None and lock_type == lg._F_RDLCK:
                raise OSError("guard arming exploded")
            return real_ofd_lock(fd, lock_type, start, length, cmd=cmd)

        monkeypatch.setattr(lg, "_ofd_lock", exploding)
        with pytest.raises(lg.WalGuardArmedIncompleteError):
            db.set_meta("must_not_commit", "unguarded-write")
        assert db._conn is None, "a failed arming must not leave the writer installed"
