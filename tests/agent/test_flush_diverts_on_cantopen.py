"""Agent flush path: a raw SQLITE_CANTOPEN diverts the batch to JSONL.

``sqlite3.OperationalError: unable to open database file`` is SQLITE_CANTOPEN — generic
("could not open it": missing parent dir, unreadable file, FD exhaustion), and NOT proof that a
``-wal``/``-shm`` generation was retired; that proven case has its own
``DeletedWalGenerationError`` guard, backed by sidecar-identity and /proc fd evidence.
The batch is diverted anyway, because the symptom costs the turn whatever its cause,
while the flush still fails closed (``False`` → ``session_persistence_failed``): no
reopen/replay is added, so an invalid path can never silently mint a database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from hermes_state import SessionDB
from run_agent import AIAgent


def _flush_agent(db, session_id):
    agent = SimpleNamespace(
        _session_db=db,
        _session_db_created=True,
        _persist_disabled=False,
        session_id=session_id,
        _session_persist_lock=None,
        _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0,
        _db_flush_scan_prefix=None,
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _pending_cli_user_message=None,
        _active_session_turn_lease_holder=None,
        _last_persistence_error_cause=None,
        _compression_adoption_failed=False,
    )
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = (
        AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    )
    agent._flush_messages_to_session_db_unlocked = (
        AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    )
    return agent


def test_flush_diverts_batch_to_jsonl_on_raw_cantopen(tmp_path, monkeypatch) -> None:
    """Level 1: a raw CANTOPEN does not lose the batch — it lands in sessions/<id>.jsonl."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("live", source="cli")
        agent = _flush_agent(db, "live")

        def _cantopen(self, *, session_id, messages, **kwargs):
            raise sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr(SessionDB, "append_messages_batch", _cantopen)

        messages = [{"role": "user", "content": "kept-on-disk-after-cantopen"}]
        result = agent._flush_messages_to_session_db(messages, [])

        # Fail closed: the turn still aborts (no silent recovery), but the batch is saved.
        assert result is False
        assert agent._last_persistence_error_cause == "disk"
        jsonl = tmp_path / "sessions" / "live.jsonl"
        assert jsonl.is_file()
        assert "kept-on-disk-after-cantopen" in jsonl.read_text(encoding="utf-8")
    finally:
        db.close()


def test_flush_fails_closed_when_the_divert_itself_fails(tmp_path, monkeypatch) -> None:
    """Negative control: the divert can fail (unwritable sessions path). The flush must still
    fail closed and quietly — never raise, never mint a database, never claim a kept copy."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("live", source="cli")
        agent = _flush_agent(db, "live")
        (tmp_path / "sessions" / "live.jsonl").mkdir(parents=True)  # open("a") on a dir fails

        def _cantopen(self, *, session_id, messages, **kwargs):
            raise sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr(SessionDB, "append_messages_batch", _cantopen)

        result = agent._flush_messages_to_session_db(
            [{"role": "user", "content": "lost-if-not-diverted"}], []
        )
        assert result is False
        assert agent._last_persistence_error_cause == "disk"
        assert (tmp_path / "sessions" / "live.jsonl").is_dir()  # untouched, nothing fabricated
    finally:
        db.close()


def test_raw_cantopen_fails_closed_without_reopen(tmp_path, monkeypatch) -> None:
    """Level 1 never reopens/replays: an invalid path (same CANTOPEN string) fails visibly and
    creates no database — the negative case for a reopen-based recovery we deliberately do not add."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("live", source="cli")
        agent = _flush_agent(db, "live")

        def _cantopen(self, *, session_id, messages, **kwargs):
            raise sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr(SessionDB, "append_messages_batch", _cantopen)

        messages = [{"role": "user", "content": "still-saved"}]
        result = agent._flush_messages_to_session_db(messages, [])

        assert result is False  # visible failure, never a silent success
        # No retry/reopen path was taken: the batch is diverted exactly once and the handle is left alone.
        jsonl = tmp_path / "sessions" / "live.jsonl"
        assert jsonl.read_text(encoding="utf-8").count("still-saved") == 1
    finally:
        db.close()
