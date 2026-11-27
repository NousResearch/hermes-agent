"""Gateway SessionStore must divert, not retry forever, after structural corruption.

Mirrors ``test_session_db_replaced_fallback.py``: once the store's handle
is quarantined (``StateDbCorruptError``) the pending transcript goes to the
JSONL/spool fallback and no FTS surgery runs on the damaged file.
"""

import json
import sqlite3

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from tests.fixtures.storage_seed import seed_session
from tests.fixtures.storage_holder import raw_holder


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
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    sid = "gw-corrupt"
    seed_session(store.persistence, sid, "cli")
    store.append_to_transcript(
        sid, {"role": "user", "content": "before", "timestamp": 1.0}
    )
    holder = raw_holder(store.persistence)  # the handle the transcript writes go through
    real_conn = holder._conn
    holder._conn = _MalformedConn(real_conn)
    try:
        store.append_to_transcript(
            sid, {"role": "user", "content": "after-corrupt", "timestamp": 2.0}
        )
        assert holder._db_corrupt is True
        # No FTS surgery ran on either layer.
        assert holder._fts_enabled is True
        assert holder._fts_stale is False
        assert store._fts_rebuild_last_attempt_at is None
        assert sid not in store._dirty_transcripts
        _assert_diverted(tmp_path, sid, "after-corrupt")
    finally:
        holder._conn = real_conn
        store.close_all_db_handles()
