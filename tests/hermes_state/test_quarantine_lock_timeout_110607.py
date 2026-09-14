"""#110607: the startup quarantine lock budget is config-tunable, not a hardcoded 5s.

``quarantine_cross_process_lock`` always accepted a ``timeout`` parameter, but the only
production call site (``SessionDB._open_writer``) never passed one — the wait budget was
pinned at 5s regardless of machine/FS speed. ``sessions.quarantine_lock_timeout`` now
flows through, and the not-acquired warning reports the real budget.
"""

from __future__ import annotations

import threading
import time

import pytest


def _write_config(home, timeout: float) -> None:
    (home / "config.yaml").write_text(
        f"sessions:\n  quarantine_lock_timeout: {timeout}\n"
    )


def _zeroed_db(home) -> "Path":
    db = home / "state.db"
    db.write_bytes(bytes(4096))
    return db


def test_helper_defaults_to_5s_without_config(tmp_path, monkeypatch):
    import hermes_state as hs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert hs._configured_quarantine_lock_timeout() == 5.0


def test_helper_reads_configured_budget(tmp_path, monkeypatch):
    import hermes_state as hs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, 0.4)
    assert hs._configured_quarantine_lock_timeout() == 0.4


def test_helper_falls_back_on_garbage(tmp_path, monkeypatch):
    import hermes_state as hs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("sessions:\n  quarantine_lock_timeout: banana\n")
    assert hs._configured_quarantine_lock_timeout() == 5.0


def test_open_writer_passes_configured_timeout(tmp_path, monkeypatch):
    """The production call site must forward the configured budget (#110607)."""
    import hermes_state as hs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, 0.4)
    db = _zeroed_db(tmp_path)

    real_lock = hs.quarantine_cross_process_lock
    captured: dict = {}

    def recorder(path, timeout=5.0):
        captured["timeout"] = timeout
        return real_lock(path, timeout=timeout)

    monkeypatch.setattr(hs, "quarantine_cross_process_lock", recorder)

    sdb = hs.SessionDB(db_path=db)
    try:
        assert captured.get("timeout") == 0.4
    finally:
        sdb.close()


def test_busy_fence_times_out_fast_under_small_budget(tmp_path, monkeypatch):
    """With a 0.2s budget, an open against a held fence fails loud and fast (~0.4s).

    Pre-fix code ignored the config: it waited out the 5s default fence (holder releases
    at 3s → acquire succeeds) and quarantined normally, so both the raise and the 2s
    ceiling discriminate old vs new (#110607).
    """
    import sqlite3

    import hermes_state as hs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, 0.2)
    db = _zeroed_db(tmp_path)

    holder_acquired = threading.Event()
    release = threading.Event()

    def holder():
        with hs.quarantine_cross_process_lock(db):
            holder_acquired.set()
            release.wait(timeout=8)

    thread = threading.Thread(name="fence-holder", target=holder)
    thread.start()
    try:
        assert holder_acquired.wait(timeout=2), "holder never acquired the fence"
        start = time.monotonic()
        with pytest.raises(sqlite3.DatabaseError):
            hs.SessionDB(db_path=db)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, (
            f"open took {elapsed:.1f}s — the configured 0.2s fence budget was not "
            "honored (pre-fix code waits out the 5s default, then succeeds)"
        )
    finally:
        release.set()
        thread.join(timeout=5)
