"""Gateway SessionStore must divert, not retry forever, after structural corruption.

Mirrors ``test_session_db_replaced_fallback.py``: once the SessionDB handle
is quarantined (``StateDbCorruptError``) the pending transcript goes to the
JSONL/spool fallback and no FTS surgery runs on the damaged file.
"""

import json
import sqlite3

from gateway.config import GatewayConfig
from gateway.session import SessionStore


class _MalformedConn:
    def __init__(self, real_conn):
        self._real = real_conn

    def execute(self, *args, **kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed")

    def __getattr__(self, name):
        return getattr(self._real, name)


def _assert_diverted(tmp_path, sid, needle):
    pending = list((tmp_path / "pending_messages").glob("pending-*.json"))
    assert pending, "expected pending_messages/pending-*.json spool"
    spooled = False
    for path in pending:
        payload = json.loads(path.read_text(encoding="utf-8"))
        message = (payload.get("data") or {}).get("message") or {}
        if needle in str(message.get("content", "")):
            spooled = True
            break
    assert spooled, f"{needle!r} missing from pending spool"
    jsonl = tmp_path / "sessions" / f"{sid}.jsonl"
    assert jsonl.is_file()
    assert needle in jsonl.read_text(encoding="utf-8")


def test_corrupt_state_db_diverts_pending_without_fts_rebuild(tmp_path, monkeypatch):
    import hermes_state

    live = tmp_path / "state.db"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", live)

    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    sid = "gw-corrupt"
    store._db.create_session(session_id=sid, source="cli")
    store.append_to_transcript(
        sid, {"role": "user", "content": "before", "timestamp": 1.0}
    )
    real_conn = store._db._conn
    store._db._conn = _MalformedConn(real_conn)
    try:
        store.append_to_transcript(
            sid, {"role": "user", "content": "after-corrupt", "timestamp": 2.0}
        )
        assert store._db._db_corrupt is True
        # No FTS surgery ran on either layer.
        assert store._db._fts_enabled is True
        assert store._db._fts_stale is False
        assert store._fts_rebuild_attempted is False
        assert sid not in store._dirty_transcripts
        _assert_diverted(tmp_path, sid, "after-corrupt")
    finally:
        store._db._conn = real_conn
        store.close_all_db_handles()
