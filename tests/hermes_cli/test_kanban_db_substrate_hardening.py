"""Tests for the kanban SQLite substrate hardening in kanban_db.

Covers the two guards added after the 2026-06-24 ("orphan index") and
2026-07-06 ("Rowid out of order") page-corruption incidents:

* Periodic integrity re-probe: a long-lived process that connected before
  corruption happened must re-detect it instead of trusting the
  ``_INITIALIZED_PATHS`` cache forever.
* ``.corrupt.*`` backup cap: crash-looping connects against a damaged board
  must not fill the disk with hundreds of backup copies.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Helpers (corruption shape mirrors test_kanban_db.py's _write_corrupt_db)
# ---------------------------------------------------------------------------

def _write_corrupt_db(path: Path) -> bytes:
    """Valid SQLite header + malformed page content.

    Passes the cheap byte-header check but fails ``PRAGMA quick_check`` /
    ``integrity_check`` — the corruption shape both incidents produced.
    """
    header = b"SQLite format 3\x00" + b"\x10\x00\x02\x02\x00\x40\x20\x20"
    header += b"\x00\x00\x00\x0c\x00\x00\x23\x46\x00\x00\x00\x00"
    header = header.ljust(100, b"\x00")
    payload = b"definitely not a valid sqlite page \x00\x01\x02\x03" * 64
    blob = header + payload
    path.write_bytes(blob)
    return blob


def _fresh_connected_db(tmp_path: Path) -> Path:
    """Create a healthy kanban DB and leave its path trusted in the cache."""
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    assert str(db_path.resolve()) in kb._INITIALIZED_PATHS
    return db_path


def _corrupt_in_place(db_path: Path) -> None:
    """Simulate on-disk corruption happening while the path is cache-trusted."""
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.parent / (db_path.name + suffix)
        if sidecar.exists():
            sidecar.unlink()
    _write_corrupt_db(db_path)


# ---------------------------------------------------------------------------
# Periodic integrity re-probe
# ---------------------------------------------------------------------------

def test_periodic_reprobe_detects_corruption_after_first_connect(
    tmp_path, monkeypatch
):
    """The exact incident shape: connect once (healthy), corrupt on disk,
    connect again after the recheck interval → must raise and evict."""
    monkeypatch.setenv("HERMES_KANBAN_INTEGRITY_RECHECK_SECONDS", "0.000001")
    db_path = _fresh_connected_db(tmp_path)
    key = str(db_path.resolve())

    _corrupt_in_place(db_path)
    time.sleep(0.01)  # let the tiny recheck interval elapse

    with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
        kb.connect(db_path=db_path)

    err = excinfo.value
    assert "quick_check" in err.reason or "sqlite refused" in err.reason
    # Cache evicted so subsequent connects re-run the full first-connect guard.
    assert key not in kb._INITIALIZED_PATHS
    # The existing backup path still ran.
    assert err.backup_path is not None
    assert err.backup_path.exists()

    # A follow-up connect goes through the FULL guard (not the cache) and
    # still refuses — no silent schema recreation on the damaged file.
    with pytest.raises(kb.KanbanDbCorruptError):
        kb.connect(db_path=db_path)


def test_reprobe_not_run_within_default_interval(tmp_path, monkeypatch):
    """Within the (default 300s) interval a cached path opens exactly one
    sqlite connection — no probe connection on the hot path."""
    monkeypatch.delenv("HERMES_KANBAN_INTEGRITY_RECHECK_SECONDS", raising=False)
    db_path = _fresh_connected_db(tmp_path)

    calls: list[tuple] = []
    real_connect = sqlite3.connect

    def counting_connect(*args, **kwargs):
        calls.append(args)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(kb.sqlite3, "connect", counting_connect)
    conn = kb.connect(db_path=db_path)
    conn.close()
    assert len(calls) == 1, f"expected only the real connection, saw {calls!r}"


def test_reprobe_disabled_when_interval_nonpositive(tmp_path, monkeypatch):
    """<= 0 disables the periodic re-probe even when one would be due."""
    monkeypatch.setenv("HERMES_KANBAN_INTEGRITY_RECHECK_SECONDS", "0")
    db_path = _fresh_connected_db(tmp_path)
    key = str(db_path.resolve())
    # Force a probe to be "due" if the feature were enabled.
    kb._LAST_INTEGRITY_PROBE.pop(key, None)

    calls: list[tuple] = []
    real_connect = sqlite3.connect

    def counting_connect(*args, **kwargs):
        calls.append(args)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(kb.sqlite3, "connect", counting_connect)
    conn = kb.connect(db_path=db_path)
    conn.close()
    assert len(calls) == 1, "disabled re-probe must not open probe connections"


def test_reprobe_lock_contention_skips_probe_and_defers(tmp_path, monkeypatch):
    """Lock/busy during the re-probe must NOT fail the hot path, must NOT
    classify as corruption, and must defer the next probe a full interval."""
    monkeypatch.setenv("HERMES_KANBAN_INTEGRITY_RECHECK_SECONDS", "0.000001")
    db_path = _fresh_connected_db(tmp_path)
    key = str(db_path.resolve())
    time.sleep(0.01)  # make the probe due

    real_connect = sqlite3.connect
    state = {"probe_seen": False}

    def flaky_connect(*args, **kwargs):
        if not state["probe_seen"]:
            # First connection attempt is the re-probe — simulate a lock.
            state["probe_seen"] = True
            raise sqlite3.OperationalError("database is locked")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(kb.sqlite3, "connect", flaky_connect)

    conn = kb.connect(db_path=db_path)  # must NOT raise
    try:
        conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
    finally:
        conn.close()

    assert state["probe_seen"] is True
    # Path stays trusted; no spurious backup; probe deferred (timestamp set).
    assert key in kb._INITIALIZED_PATHS
    assert list(tmp_path.glob("*.corrupt.*")) == []
    assert key in kb._LAST_INTEGRITY_PROBE


def test_reprobe_connection_is_read_only(tmp_path, monkeypatch):
    """The periodic re-probe must open the DB via URI mode=ro so it can
    never checkpoint, recover, or take write locks on a WAL/hot-journal DB."""
    monkeypatch.setenv("HERMES_KANBAN_INTEGRITY_RECHECK_SECONDS", "0.000001")
    db_path = _fresh_connected_db(tmp_path)
    time.sleep(0.01)  # make the probe due

    calls: list[tuple[tuple, dict]] = []
    real_connect = sqlite3.connect

    def capturing_connect(*args, **kwargs):
        calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(kb.sqlite3, "connect", capturing_connect)
    conn = kb.connect(db_path=db_path)
    conn.close()

    # First connection is the re-probe, second is the real connection.
    assert len(calls) == 2, f"expected probe + real connection, saw {calls!r}"
    probe_args, probe_kwargs = calls[0]
    assert probe_args[0].startswith("file:")
    assert "mode=ro" in probe_args[0]
    assert probe_kwargs.get("uri") is True
    real_args, real_kwargs = calls[1]
    assert "mode=ro" not in str(real_args[0])

    # And the probe connection genuinely cannot write.
    ro = real_connect(probe_args[0], **probe_kwargs)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            ro.execute("CREATE TABLE probe_should_not_write(x)")
    finally:
        ro.close()


def test_reprobe_timestamp_set_before_probe_coalesces(tmp_path, monkeypatch):
    """The probe timestamp is recorded BEFORE the probe connection opens, so
    concurrent connects skip instead of stacking probes behind the lock."""
    monkeypatch.setenv("HERMES_KANBAN_INTEGRITY_RECHECK_SECONDS", "0.000001")
    db_path = _fresh_connected_db(tmp_path)
    key = str(db_path.resolve())
    time.sleep(0.01)  # make the probe due
    kb._LAST_INTEGRITY_PROBE.pop(key, None)

    seen: dict[str, object] = {}
    real_connect = sqlite3.connect

    def observing_connect(*args, **kwargs):
        seen.setdefault("stamp_at_probe", kb._LAST_INTEGRITY_PROBE.get(key))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(kb.sqlite3, "connect", observing_connect)
    conn = kb.connect(db_path=db_path)
    conn.close()

    assert seen["stamp_at_probe"] is not None, (
        "timestamp must be stamped before the probe connection opens"
    )


def test_reprobe_effective_interval_jitter_bounded_and_deterministic():
    """Per-process jitter: within (interval, interval*1.1], stable across
    calls in the same process, and derived from os.getpid()."""
    import os

    interval = 300.0
    first = kb._reprobe_effective_interval(interval)
    second = kb._reprobe_effective_interval(interval)
    assert first == second, "jitter must be deterministic within a process"
    assert interval <= first <= interval * 1.1
    expected = interval * (1.0 + 0.1 * ((os.getpid() % 1024) / 1024.0))
    assert first == expected


def test_integrity_recheck_seconds_env_parsing(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_INTEGRITY_RECHECK_SECONDS", raising=False)
    assert kb._integrity_recheck_seconds() == 300.0
    monkeypatch.setenv("HERMES_KANBAN_INTEGRITY_RECHECK_SECONDS", "42.5")
    assert kb._integrity_recheck_seconds() == 42.5
    monkeypatch.setenv("HERMES_KANBAN_INTEGRITY_RECHECK_SECONDS", "not-a-number")
    assert kb._integrity_recheck_seconds() == 300.0


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_integrity_recheck_seconds_nonfinite_falls_back(monkeypatch, raw):
    """float() accepts NaN/inf; both must fall back to the default instead
    of producing an interval that fails comparisons unpredictably."""
    monkeypatch.setenv("HERMES_KANBAN_INTEGRITY_RECHECK_SECONDS", raw)
    assert kb._integrity_recheck_seconds() == 300.0


# ---------------------------------------------------------------------------
# .corrupt backup swarm cap
# ---------------------------------------------------------------------------

def _make_fake_corrupt_backups(db_path: Path, count: int) -> None:
    for i in range(count):
        (db_path.parent / f"{db_path.name}.corrupt.fake{i}.bak").write_bytes(b"x")


def test_backup_cap_skips_new_backup_at_cap(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_KANBAN_CORRUPT_BACKUP_CAP", "3")
    db_path = tmp_path / "kanban.db"
    original = _write_corrupt_db(db_path)
    _make_fake_corrupt_backups(db_path, 3)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
            kb.connect(db_path=db_path)

    # Error still raised loudly, but no new backup was written.
    assert excinfo.value.backup_path is None
    siblings = sorted(p.name for p in tmp_path.glob("kanban.db.corrupt.*"))
    assert siblings == [f"kanban.db.corrupt.fake{i}.bak" for i in range(3)]
    # Original bytes untouched.
    assert db_path.read_bytes() == original
    # One warning via the module logger.
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "corrupt-backup cap" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_backup_cap_under_cap_still_creates_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_CORRUPT_BACKUP_CAP", "3")
    db_path = tmp_path / "kanban.db"
    original = _write_corrupt_db(db_path)
    _make_fake_corrupt_backups(db_path, 2)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
        kb.connect(db_path=db_path)

    backup = excinfo.value.backup_path
    assert backup is not None and backup.exists()
    assert backup.read_bytes() == original


def test_backup_cap_counts_literally_with_glob_metachar_filename(
    tmp_path, monkeypatch
):
    """A DB filename containing glob metacharacters ("kanban[1].db") must
    still count its .corrupt.* siblings — parent.glob() would treat "[1]"
    as a character class, count 0, and defeat the cap."""
    monkeypatch.setenv("HERMES_KANBAN_CORRUPT_BACKUP_CAP", "3")
    db_path = tmp_path / "kanban[1].db"
    original = _write_corrupt_db(db_path)
    for i in range(3):
        (tmp_path / f"kanban[1].db.corrupt.fake{i}.bak").write_bytes(b"x")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
        kb.connect(db_path=db_path)

    # Cap honored: no new backup despite the metachar name.
    assert excinfo.value.backup_path is None
    siblings = sorted(
        p.name for p in tmp_path.iterdir()
        if p.name.startswith("kanban[1].db.corrupt.")
    )
    assert siblings == [f"kanban[1].db.corrupt.fake{i}.bak" for i in range(3)]
    assert db_path.read_bytes() == original


def test_corrupt_backup_cap_env_parsing(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_CORRUPT_BACKUP_CAP", raising=False)
    assert kb._corrupt_backup_cap() == 16
    monkeypatch.setenv("HERMES_KANBAN_CORRUPT_BACKUP_CAP", "5")
    assert kb._corrupt_backup_cap() == 5
    monkeypatch.setenv("HERMES_KANBAN_CORRUPT_BACKUP_CAP", "garbage")
    assert kb._corrupt_backup_cap() == 16


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "1e2", "16.0"])
def test_corrupt_backup_cap_nonfinite_and_float_strings_fall_back(
    monkeypatch, raw
):
    """Non-finite and float-shaped strings must land on the default cap,
    never a NaN/inf that breaks the >= cap comparison."""
    monkeypatch.setenv("HERMES_KANBAN_CORRUPT_BACKUP_CAP", raw)
    assert kb._corrupt_backup_cap() == 16
