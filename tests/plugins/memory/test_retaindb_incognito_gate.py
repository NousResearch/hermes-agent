"""Incognito gate for the RetainDB provider (Gate 1): protected sessions must not be ingested.

The write-behind queue must skip sessions that hold a live incognito lease (``plugin-data/
incognito/leases.json``) and sessions that no longer exist in state.db at flush time (burned
between enqueue and flush, including crash-replayed rows). The gate fails open on read errors.
"""
from __future__ import annotations

import json
import sqlite3
import time

from plugins.memory import retaindb
from plugins.memory.retaindb import _WriteQueue


def _make_home(tmp_path, *, leases=None, session_ids=(), message_sessions=()):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    if leases is not None:
        d = home / "plugin-data" / "incognito"
        d.mkdir(parents=True, exist_ok=True)
        (d / "leases.json").write_text(json.dumps(leases), encoding="utf-8")
    conn = sqlite3.connect(str(home / "state.db"))
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, hidden INTEGER DEFAULT 0)")
    conn.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT, role TEXT, content TEXT)")
    for sid in session_ids:
        conn.execute("INSERT INTO sessions (id) VALUES (?)", (sid,))
    for i, sid in enumerate(message_sessions):
        conn.execute("INSERT INTO messages (id, session_id, role, content) VALUES (?, ?, 'user', 'x')", (str(i), sid))
    conn.commit()
    conn.close()
    return home


class _RecordingClient:
    def __init__(self):
        self.calls = []

    def ingest_session(self, user_id, session_id, messages, timeout=15.0):
        self.calls.append(session_id)
        return {"ok": True}


def _pending_count(db_path):
    conn = sqlite3.connect(str(db_path))
    n = conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
    conn.close()
    return n


# --- predicate ---------------------------------------------------------------

def test_guard_skips_live_incognito_session(tmp_path):
    home = _make_home(tmp_path, leases={"sid-live": {"mode": "standard"}}, session_ids=["sid-live"])
    assert retaindb._should_skip_ingest(home, "sid-live") is True


def test_guard_skips_burned_session(tmp_path):
    home = _make_home(tmp_path, session_ids=["sid-someone-else"])
    assert retaindb._should_skip_ingest(home, "sid-burned") is True


def test_guard_passes_normal_session(tmp_path):
    home = _make_home(tmp_path, session_ids=["sid-normal"])
    assert retaindb._should_skip_ingest(home, "sid-normal") is False


def test_guard_messages_only_session_passes(tmp_path):
    # No sessions row yet, but messages exist — not burned; let the row through.
    home = _make_home(tmp_path, message_sessions=["sid-msgs"])
    assert retaindb._should_skip_ingest(home, "sid-msgs") is False


def test_guard_fail_open(tmp_path):
    home = _make_home(tmp_path, session_ids=["sid-normal"])
    d = home / "plugin-data" / "incognito"
    d.mkdir(parents=True, exist_ok=True)
    (d / "leases.json").write_text("{not json", encoding="utf-8")
    assert retaindb._should_skip_ingest(home, "sid-normal") is False
    assert retaindb._should_skip_ingest(tmp_path / "missing-home", "sid-x") is False
    assert retaindb._should_skip_ingest(home, "") is False


def test_guard_disabled_flag(tmp_path):
    home = _make_home(tmp_path, leases={"sid-live": {}})
    assert retaindb._should_skip_ingest(home, "sid-live", enabled=False) is False


# --- queue integration ---------------------------------------------------------

def test_enqueue_respects_guard(tmp_path):
    client = _RecordingClient()
    home = _make_home(tmp_path)  # no sessions at all: every id reads as burned
    q = _WriteQueue(client, tmp_path / "q.db", ingest_guard=lambda sid: retaindb._should_skip_ingest(home, sid))
    q.enqueue("u", "sid-x", [{"role": "user", "content": "hi"}])
    time.sleep(0.2)
    q.shutdown()
    assert client.calls == []
    assert _pending_count(tmp_path / "q.db") == 0


def test_flush_drops_row_burned_after_enqueue(tmp_path):
    import threading

    client = _RecordingClient()
    home = _make_home(tmp_path, session_ids=["sid-race", "sid-blocker"])
    guard = lambda sid: retaindb._should_skip_ingest(home, sid)
    q = _WriteQueue(client, tmp_path / "q.db", ingest_guard=guard)
    release = threading.Event()

    def ingest_session(user_id, session_id, messages, timeout=15.0):
        if session_id == "sid-blocker":  # hold the writer thread until the simulated burn lands
            release.wait(timeout=5)
        client.calls.append(session_id)
        return {"ok": True}

    client.ingest_session = ingest_session
    assert guard("sid-race") is False
    q.enqueue("u", "sid-blocker", [{"role": "user", "content": "hold"}])
    q.enqueue("u", "sid-race", [{"role": "user", "content": "hi"}])  # queued behind the blocker
    conn = sqlite3.connect(str(home / "state.db"))  # simulate a burn between enqueue and flush
    conn.execute("DELETE FROM sessions WHERE id = 'sid-race'")
    conn.commit()
    conn.close()
    release.set()
    time.sleep(0.2)
    q.shutdown()
    assert "sid-race" not in client.calls
    assert client.calls == ["sid-blocker"]
    assert _pending_count(tmp_path / "q.db") == 0


def test_crash_replay_skips_burned_session(tmp_path):
    home = _make_home(tmp_path)  # sid-old was burned before the crash left its row pending
    db = tmp_path / "q.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE pending (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, session_id TEXT, "
                 "messages_json TEXT, created_at TEXT, last_error TEXT)")
    conn.execute("INSERT INTO pending (user_id, session_id, messages_json, created_at) VALUES ('u', 'sid-old', '[]', 'now')")
    conn.commit()
    conn.close()
    client = _RecordingClient()
    q = _WriteQueue(client, db, ingest_guard=lambda sid: retaindb._should_skip_ingest(home, sid))
    time.sleep(0.2)
    q.shutdown()
    assert client.calls == []
    assert _pending_count(db) == 0


def test_no_guard_keeps_legacy_behavior(tmp_path):
    home = _make_home(tmp_path)
    client = _RecordingClient()
    q = _WriteQueue(client, tmp_path / "q.db")  # no guard: pre-patch behavior
    q.enqueue("u", "sid-any", [{"role": "user", "content": "hi"}])
    time.sleep(0.2)
    q.shutdown()
    assert client.calls == ["sid-any"]
    assert _pending_count(tmp_path / "q.db") == 0
