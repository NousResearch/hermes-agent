"""Regression tests for the DB-failure transcript spool fallback."""

from typing import cast
from types import SimpleNamespace

import hermes_state
from agent.session_persistence import _DB_PERSISTED_MARKER, _db_flush_failed
from agent.session_persistence import SessionPersistenceMixin


def test_db_flush_failure_attempts_spool_for_transient_errors(monkeypatch):
    calls = []

    def fake_spool(session_id, rows):
        calls.append((session_id, rows))
        return "/tmp/session-spool.jsonl"

    monkeypatch.setattr(hermes_state, "divert_session_transcript_jsonl", fake_spool)
    agent = SimpleNamespace(
        session_id="session-1",
        _db_flush_scan_prefix=["old"],
        _last_persistence_error_cause=None,
        _compression_adoption_failed=False,
    )
    rows = [{"role": "assistant", "content": "answer"}]

    assert _db_flush_failed(agent, OSError("read-only filesystem"), rows, 0) is False
    assert calls == [("session-1", rows)]
    assert agent._persistence_spool_path == "/tmp/session-spool.jsonl"
    assert agent._persistence_spool_saved is True


def test_db_flush_failure_remains_fail_closed_when_spool_fails(monkeypatch):
    monkeypatch.setattr(hermes_state, "divert_session_transcript_jsonl", lambda *_: None)
    agent = SimpleNamespace(
        session_id="session-2",
        _db_flush_scan_prefix=["old"],
        _last_persistence_error_cause=None,
        _compression_adoption_failed=False,
    )

    assert _db_flush_failed(agent, OSError("read-only filesystem"), [{"x": 1}], 0) is False
    assert agent._persistence_spool_saved is False
    assert agent._persistence_spool_path is None


def test_real_flush_writer_failure_spools_and_returns_non_receipt(monkeypatch):
    calls = []

    def fake_spool(session_id, rows):
        calls.append((session_id, rows))
        return "/tmp/recovery.jsonl"

    class BrokenDB:
        def append_messages_batch(self, **_kwargs):
            raise OSError("read-only filesystem")

    monkeypatch.setattr(hermes_state, "divert_session_transcript_jsonl", fake_spool)
    agent = cast(SessionPersistenceMixin, SimpleNamespace(
        session_id="session-e2e",
        _session_db=BrokenDB(),
        _session_db_created=True,
        _last_flushed_db_idx=0,
        _flushed_db_message_ids=set(),
        _db_flush_scan_prefix=[],
        _persist_user_message_idx=None,
        _pending_cli_user_message=None,
        _compression_adoption_failed=False,
        _last_persistence_error_cause=None,
    ))
    messages = [{"role": "assistant", "content": "answer"}]

    result = SessionPersistenceMixin._flush_messages_to_session_db_unlocked(  # type: ignore[arg-type]
        agent, messages, None
    )

    assert result is False
    assert calls[0][0] == "session-e2e"
    assert calls[0][1][0]["content"] == "answer"
    assert getattr(agent, "_persistence_spool_saved") is True
    assert getattr(agent, "_persistence_spool_path") == "/tmp/recovery.jsonl"
    assert _DB_PERSISTED_MARKER not in messages[0]


def test_session_creation_failure_also_spools_collected_rows(monkeypatch):
    calls = []

    monkeypatch.setattr(
        hermes_state,
        "divert_session_transcript_jsonl",
        lambda session_id, rows: calls.append((session_id, rows)) or "/tmp/create-failure.jsonl",
    )

    class UnavailableDB:
        pass

    agent = cast(SessionPersistenceMixin, SimpleNamespace(
        session_id="session-create-failure",
        _session_db=UnavailableDB(),
        _session_db_created=False,
        _last_flushed_db_idx=0,
        _flushed_db_message_ids=set(),
        _db_flush_scan_prefix=[],
        _persist_user_message_idx=None,
        _pending_cli_user_message=None,
        _compression_adoption_failed=False,
        _last_persistence_error_cause=None,
    ))
    setattr(agent, "_ensure_db_session", lambda: (_ for _ in ()).throw(OSError("database unavailable")))
    messages = [{"role": "assistant", "content": "answer"}]

    result = SessionPersistenceMixin._flush_messages_to_session_db_unlocked(agent, messages, None)

    assert result is False
    assert calls[0][0] == "session-create-failure"
    assert calls[0][1][0]["content"] == "answer"
