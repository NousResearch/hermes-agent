"""A concurrent opener's writability preflight must not race another opener's quarantine.

Two processes (gateway + CLI) starting on a zeroed state.db: the preflight lists ``state.db``
with ``is_file()``, the other opener's quarantine renames it aside, and ``os.access`` then
fails with ENOENT. That used to be reported as "state.db is not writable ... read-only for
this user" (flaky ``test_concurrent_quarantine_no_clobber`` on CI). Event-driven, no sleeps.
"""

import os
import threading
from pathlib import Path

import hermes_state as hs
import hermes_state_repair


def test_preflight_cannot_observe_db_mid_quarantine(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    db.write_bytes(bytes(4096))  # zeroed 4 KB file

    late_reached = threading.Event()   # worker 1 hit its preflight check OR the quarantine lock
    quarantined = threading.Event()    # worker 0 finished moving the zeroed file aside
    real_access = os.access
    real_lock = hs.quarantine_cross_process_lock
    real_quarantine = hs.quarantine_invalid_state_db
    errors: list = [None, None]

    def is_late() -> bool:
        return threading.current_thread().name == "late-opener"

    def gated_access(path, mode, *args, **kwargs):
        # Worker 1 has listed state.db via is_file(); hold its access check until worker 0's
        # quarantine has renamed the file (the CI interleaving, forced).
        if is_late() and Path(path) == db and not quarantined.is_set():
            late_reached.set()
            assert quarantined.wait(10), "worker 0 never quarantined"
        return real_access(path, mode, *args, **kwargs)

    def signalling_lock(path, *args, **kwargs):
        if is_late():
            late_reached.set()  # about to block on the lock worker 0 holds
        return real_lock(path, *args, **kwargs)

    def worker(idx):
        try:
            hs.SessionDB(db_path=db).close()
        except Exception as exc:  # surfaced by the assertions below
            errors[idx] = exc

    late = threading.Thread(target=worker, args=(1,), name="late-opener")

    def quarantine_then_release(path, **kwargs):
        # Worker 0 holds the quarantine lock here; start worker 1 against the still-zeroed file.
        late.start()
        assert late_reached.wait(10), "worker 1 never reached its preflight or the lock"
        try:
            return real_quarantine(path, **kwargs)
        finally:
            quarantined.set()

    monkeypatch.setattr(hermes_state_repair.os, "access", gated_access)
    monkeypatch.setattr(hs, "quarantine_cross_process_lock", signalling_lock)
    monkeypatch.setattr(hs, "quarantine_invalid_state_db", quarantine_then_release)

    first = threading.Thread(target=worker, args=(0,), name="first-opener")
    first.start()
    first.join(20)
    late.join(20)
    assert not first.is_alive() and not late.is_alive()

    assert errors == [None, None], f"concurrent open raised: {errors}"
    baks = list(tmp_path.glob("state.db.zeroed-*.bak"))
    assert len(baks) == 1 and baks[0].read_bytes() == bytes(4096)
