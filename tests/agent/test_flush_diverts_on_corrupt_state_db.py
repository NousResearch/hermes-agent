"""Agent flush path: a quarantined (structurally corrupt) store diverts to JSONL.

Mirrors the replaced-file contract: the batch that SQLite will never take
again is kept on disk under ``sessions/<id>.jsonl`` instead of only in RAM,
the flush fails closed (no retry loop), and the turn-end explanation gets the
``corrupt`` cause.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from persistence import StoreQuarantined
from run_agent import AIAgent
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _flush_agent(db, session_id, persistence):
    agent = SimpleNamespace(
        _session_db=db,
        persistence=persistence,
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


def test_flush_diverts_batch_to_jsonl_when_handle_is_quarantined(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = storage_handle(tmp_path)
    try:
        seed_session(db, "live", "cli")
        def _quarantined(session_id, messages, **kwargs):
            raise StoreQuarantined("database disk image is malformed (quarantined)")

        handle = SimpleNamespace(read_only=False, scope=SimpleNamespace(profile_dir=tmp_path),
                                 messages=SimpleNamespace(append_batch=_quarantined))
        agent = _flush_agent(db, "live", handle)

        messages = [{"role": "user", "content": "kept-on-disk-after-corruption"}]
        result = agent._flush_messages_to_session_db(messages, [])

        assert result is False
        assert agent._last_persistence_error_cause == "corrupt"
        jsonl = tmp_path / "sessions" / "live.jsonl"
        assert jsonl.is_file()
        assert "kept-on-disk-after-corruption" in jsonl.read_text(encoding="utf-8")
    finally:
        db.close()
