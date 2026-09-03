"""Tests for hermes_cli/crash_restore.py — startup crash-restore offers.

Covers the marker write/remove roundtrip under a temp HERMES_HOME, the
crashed-session scan (dead pid + open row + messages), live-pid exclusion,
cleanly-ended and deleted-row housekeeping, the current-session exclusion,
the offer limit/ordering, and the session.crash_restore config gate.
"""

import json
import os
import time

import pytest

from hermes_cli import crash_restore as cr


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def session_db(hermes_home):
    from hermes_state import SessionDB

    db = SessionDB()
    yield db
    db.close()


def _make_session(db, session_id, *, messages=1, ended=None):
    db.create_session(session_id=session_id, source="cli")
    for i in range(messages):
        db.append_message(session_id, "user", f"msg {i}")
    if ended:
        db.end_session(session_id, ended)


def _write_dead_marker(session_id, *, pid=None, ts=None):
    """Write a marker owned by a definitely-dead process."""
    directory = cr._markers_dir()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        # A pid we can guarantee is not running: fork-and-reap.
        "pid": pid if pid is not None else _dead_pid(),
        "process_start_time": None,
        "ts": ts if ts is not None else time.time(),
    }
    (directory / f"{session_id}.json").write_text(json.dumps(payload))


def _dead_pid():
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# ---------------------------------------------------------------- markers

def test_marker_write_and_remove_roundtrip(hermes_home):
    cr.write_live_marker("sess-a")
    path = cr._markers_dir() / "sess-a.json"
    assert path.is_file()
    data = json.loads(path.read_text())
    assert data["session_id"] == "sess-a"
    assert data["pid"] == os.getpid()
    cr.remove_live_marker("sess-a")
    assert not path.exists()


def test_marker_write_noop_when_disabled(hermes_home, monkeypatch):
    monkeypatch.setattr(cr, "is_enabled", lambda: False)
    cr.write_live_marker("sess-a")
    assert not (cr._markers_dir() / "sess-a.json").exists()


def test_marker_write_noop_on_empty_id(hermes_home):
    cr.write_live_marker("")
    assert not cr._markers_dir().exists() or not any(cr._markers_dir().iterdir())


# ------------------------------------------------------------------ scan

def test_scan_offers_crashed_session(hermes_home, session_db):
    _make_session(session_db, "crashed-1", messages=2)
    _write_dead_marker("crashed-1")
    offers = cr.find_crashed_sessions(session_db)
    assert [o["id"] for o in offers] == ["crashed-1"]


def test_scan_skips_live_process(hermes_home, session_db):
    _make_session(session_db, "live-1", messages=2)
    # Marker owned by THIS (very alive) process.
    cr.write_live_marker("live-1")
    assert cr.find_crashed_sessions(session_db) == []
    # Marker must survive — it belongs to a live session.
    assert (cr._markers_dir() / "live-1.json").is_file()


def test_scan_skips_current_session(hermes_home, session_db):
    _make_session(session_db, "self-1", messages=1)
    _write_dead_marker("self-1")
    assert cr.find_crashed_sessions(session_db, current_session_id="self-1") == []


def test_scan_prunes_cleanly_ended_row(hermes_home, session_db):
    _make_session(session_db, "done-1", messages=1, ended="cli_close")
    _write_dead_marker("done-1")
    assert cr.find_crashed_sessions(session_db) == []
    assert not (cr._markers_dir() / "done-1.json").exists()


def test_scan_prunes_deleted_row(hermes_home, session_db):
    _write_dead_marker("ghost-1")
    assert cr.find_crashed_sessions(session_db) == []
    assert not (cr._markers_dir() / "ghost-1.json").exists()


def test_scan_prunes_empty_session(hermes_home, session_db):
    _make_session(session_db, "empty-1", messages=0)
    _write_dead_marker("empty-1")
    assert cr.find_crashed_sessions(session_db) == []
    assert not (cr._markers_dir() / "empty-1.json").exists()


def test_scan_prunes_stale_marker(hermes_home, session_db):
    _make_session(session_db, "old-1", messages=1)
    _write_dead_marker("old-1", ts=time.time() - cr._STALE_AFTER_SECONDS - 60)
    assert cr.find_crashed_sessions(session_db) == []
    assert not (cr._markers_dir() / "old-1.json").exists()


def test_scan_prunes_corrupt_marker(hermes_home, session_db):
    directory = cr._markers_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "junk.json").write_text("{not json")
    assert cr.find_crashed_sessions(session_db) == []
    assert not (directory / "junk.json").exists()


def test_scan_limit_and_ordering(hermes_home, session_db):
    now = time.time()
    for i in range(5):
        sid = f"many-{i}"
        _make_session(session_db, sid, messages=1)
        _write_dead_marker(sid)
        # Stagger last activity so ordering is deterministic (newest first).
        session_db._execute_write(
            lambda conn, sid=sid, i=i: conn.execute(
                "UPDATE sessions SET last_activity_at = ? WHERE id = ?",
                (now - (5 - i) * 60, sid),
            )
        )
    offers = cr.find_crashed_sessions(session_db, limit=3)
    assert len(offers) == 3
    assert [o["id"] for o in offers] == ["many-4", "many-3", "many-2"]


def test_scan_disabled_by_config(hermes_home, session_db, monkeypatch):
    _make_session(session_db, "gated-1", messages=1)
    _write_dead_marker("gated-1")
    monkeypatch.setattr(cr, "is_enabled", lambda: False)
    assert cr.find_crashed_sessions(session_db) == []


def test_scan_none_db_is_safe(hermes_home):
    assert cr.find_crashed_sessions(None) == []
