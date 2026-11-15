"""Checkpoint re-arm policy, including a DELETE-runtime dispatch control.

The dispatch control uses real guard descriptors but does not claim WAL coverage.
The native cases require the production WAL gate to admit this SQLite build.
"""
import logging
import os

import pytest

from hermes_state import SessionDB
import hermes_state_lockguard as lg


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("0", False), ("false", False),
    ("no", False), ("off", False), ("unexpected", False),
    ("1", True), ("true", True), ("yes", True), ("on", True),
    (" TRUE ", True),
])
def test_bypass_is_explicit_and_read_once(tmp_path, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("HERMES_STATE_WAL_GUARD_BYPASS", raising=False)
    else:
        monkeypatch.setenv("HERMES_STATE_WAL_GUARD_BYPASS", value)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._wal_guard_degraded is expected
        monkeypatch.setenv("HERMES_STATE_WAL_GUARD_BYPASS", "0" if expected else "1")
        assert db._wal_guard_degraded is expected
    finally:
        db.close()


@pytest.mark.parametrize("native_wal", [False, True], ids=["dispatch-control", "native-wal"])
@pytest.mark.parametrize("fault", ["refusal", "oserror"])
@pytest.mark.parametrize("bypass", ["0", "false", "off", "1"])
def test_checkpoint_missing_shm_retires_before_next_write(
    tmp_path, monkeypatch, caplog, native_wal, fault, bypass,
):
    if not lg.supported():
        pytest.skip("requires OFD locks")
    monkeypatch.setenv("HERMES_STATE_WAL_GUARD_BYPASS", bypass)
    db = SessionDB(db_path=tmp_path / "state.db")
    shm_fd = None
    original_wal = db._wal_active
    try:
        if native_wal and not original_wal:
            pytest.skip("production SQLite gate did not admit WAL; dispatch control is separate")
        while db._evict_one_idle_read_conn():
            pass
        # Simulate the documented minted-after-open SHM state: the main range
        # has ownership, but the SHM range has not yet been added to this handle.
        lg.release(db._wal_lock_guard)
        if not original_wal:
            shm_fd = os.open(str(db.db_path) + "-shm", os.O_CREAT | os.O_RDWR, 0o600)
        shm_ident = lg._identity(str(db.db_path) + "-shm")
        assert shm_ident is not None
        if not original_wal:
            # A main-file OFD read lock blocks DELETE-mode commits. Keep only
            # the real SHM descriptor in this dispatch-only lock fixture.
            real_ranges = lg._guard_ranges
            monkeypatch.setattr(lg, "_guard_ranges", lambda path: {
                ident: rng for ident, rng in real_ranges(path).items() if ident == shm_ident
            })
        db._wal_lock_guard = lg.hold(db.db_path, strict=True)
        assert shm_ident in db._wal_lock_guard
        lg.release({shm_ident: db._wal_lock_guard.pop(shm_ident)})
        # On vulnerable SQLite this only selects the production checkpoint
        # dispatch branch; the connection remains DELETE, not unsafe forced WAL.
        db._wal_active = True
        if not original_wal:
            real_open = db._open_writer_conn

            def open_dispatch_control():
                conn = real_open()
                db._wal_active = True  # dispatch seam only; journal stays DELETE
                return conn

            monkeypatch.setattr(db, "_open_writer_conn", open_dispatch_control)
        real_lock = lg._ofd_lock

        def fail_shm(fd, lock_type, start, length, *, cmd=None):
            if cmd is None and lock_type == lg._F_RDLCK and start == lg._SHM_DMS_BYTE:
                if fault == "oserror":
                    raise OSError("checkpoint guard arming exploded")
                return False
            return real_lock(fd, lock_type, start, length, cmd=cmd)

        monkeypatch.setattr(lg, "_ofd_lock", fail_shm)
        candidate = db._conn
        if bypass == "1":
            with caplog.at_level(logging.WARNING, logger="hermes_state"):
                db._try_wal_checkpoint()
            assert "WAL lock guard incomplete" in caplog.text
            assert db._conn is candidate
            db.set_meta("checkpoint_probe", "explicitly-degraded")
            assert db.get_meta("checkpoint_probe") == "explicitly-degraded"
        else:
            with caplog.at_level(logging.WARNING, logger="hermes_state"):
                with pytest.raises(lg.WalGuardArmedIncompleteError):
                    db._try_wal_checkpoint()
            assert db._conn is None
            assert not db._wal_lock_guard
            if fault == "oserror":
                assert "checkpoint guard arming exploded" in caplog.text
                assert "failed to arm" in caplog.text
            # A new write must re-establish the complete open+guard pair; it
            # cannot reuse the checkpoint's unguarded connection.
            for _ in range(2):
                with pytest.raises(lg.WalGuardArmedIncompleteError):
                    db.set_meta("checkpoint_probe", "must-not-commit")
                assert db._conn is None
            monkeypatch.setattr(lg, "_ofd_lock", real_lock)
            db.set_meta("checkpoint_probe", "recovered")
            assert db.get_meta("checkpoint_probe") == "recovered"
    finally:
        db._wal_active = original_wal
        db.close()
        if shm_fd is not None:
            os.close(shm_fd)
