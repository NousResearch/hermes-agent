"""Offline regression for #118142: PG transcript mutation failures retain tracebacks."""

import logging
import traceback
from unittest.mock import Mock

import pytest

import hermes_state_rewind
from gateway.session import SessionStore


@pytest.fixture
def pg_shell(monkeypatch):
    shell = object.__new__(SessionStore)
    pg_store = Mock()
    sqlite_lookup = Mock(side_effect=AssertionError("SQLite fallback is forbidden"))
    monkeypatch.setattr(shell, "_postgresql_transcript_store", lambda: pg_store)
    monkeypatch.setattr(shell, "_db_for_session_id", sqlite_lookup)
    return shell, pg_store, sqlite_lookup


def _failure_record(caplog, message):
    records = [
        record for record in caplog.records
        if record.name == "gateway.session" and message in record.getMessage()
    ]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    assert "RuntimeError: PG write failed" in "".join(traceback.format_exception(*record.exc_info))


def test_pg_rewrite_failure_logs_traceback_without_sqlite_fallback(pg_shell, caplog):
    shell, pg_store, sqlite_lookup = pg_shell
    pg_store.replace_messages.side_effect = RuntimeError("PG write failed")
    messages = [{"role": "user", "content": "ask"}]

    with caplog.at_level(logging.DEBUG, logger="gateway.session"):
        result = shell.rewrite_transcript("sid", messages, active_only=True)

    assert result is False
    pg_store.replace_messages.assert_called_once_with(
        "sid", messages, active_only=True, reject_active_turn_lease=False)
    sqlite_lookup.assert_not_called()
    _failure_record(caplog, "Failed to rewrite transcript in PostgreSQL")


def test_pg_rewind_failure_logs_traceback_without_sqlite_fallback(
    pg_shell, caplog, monkeypatch
):
    shell, pg_store, sqlite_lookup = pg_shell
    rewind = Mock(side_effect=RuntimeError("PG write failed"))
    monkeypatch.setattr(hermes_state_rewind, "rewind_user_turn", rewind)

    with caplog.at_level(logging.DEBUG, logger="gateway.session"):
        result = shell.rewind_session("sid", require_retryable_composite=True)

    assert result is None
    rewind.assert_called_once_with(
        pg_store, "sid", -1, require_retryable=True, require_composite=True)
    sqlite_lookup.assert_not_called()
    _failure_record(caplog, "rewind_session: PostgreSQL rewind failed")


def test_pg_rewind_retry_policy_error_still_propagates(pg_shell, monkeypatch):
    shell, pg_store, sqlite_lookup = pg_shell
    rewind = Mock(side_effect=ValueError("media cannot be retried"))
    monkeypatch.setattr(hermes_state_rewind, "rewind_user_turn", rewind)

    with pytest.raises(ValueError, match="media cannot be retried"):
        shell.rewind_session("sid", require_retryable_composite=True)

    rewind.assert_called_once_with(
        pg_store, "sid", -1, require_retryable=True, require_composite=True)
    sqlite_lookup.assert_not_called()
