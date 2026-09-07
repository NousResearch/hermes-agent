"""Regression tests for the DB-failure transcript spool fallback."""

from types import SimpleNamespace

import hermes_state
from agent.session_persistence import _db_flush_failed


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
