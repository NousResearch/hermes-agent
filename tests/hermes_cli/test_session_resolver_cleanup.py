"""SessionDB lifecycle tests for CLI session-name resolution."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_cli import main as main_mod


def test_session_resolver_closes_database_after_success(monkeypatch):
    db = MagicMock()
    db.get_session.return_value = {"id": "session-root"}
    db.get_compression_tip.return_value = "session-tip"
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)

    result = main_mod._resolve_session_by_name_or_id("session-root")

    assert result == "session-tip"
    db.close.assert_called_once_with()


def test_session_resolver_closes_database_after_query_failure(monkeypatch):
    db = MagicMock()
    db.get_session.side_effect = RuntimeError("read failed")
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)

    result = main_mod._resolve_session_by_name_or_id("session-root")

    assert result is None
    db.close.assert_called_once_with()


def test_session_resolver_keeps_result_when_close_fails(monkeypatch):
    db = MagicMock()
    db.get_session.return_value = {"id": "session-root"}
    db.get_compression_tip.return_value = "session-tip"
    db.close.side_effect = RuntimeError("close failed")
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)

    result = main_mod._resolve_session_by_name_or_id("session-root")

    assert result == "session-tip"
    db.close.assert_called_once_with()


def test_resume_session_cwd_closes_database_after_success(monkeypatch):
    db = MagicMock()
    db.get_session.return_value = {"cwd": " /tmp/project "}
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)

    result = main_mod._resume_session_cwd("session-root")

    assert result == "/tmp/project"
    db.close.assert_called_once_with()


def test_resume_session_cwd_closes_database_after_query_failure(monkeypatch):
    db = MagicMock()
    db.get_session.side_effect = RuntimeError("read failed")
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)

    result = main_mod._resume_session_cwd("session-root")

    assert result == ""
    db.close.assert_called_once_with()


def test_cmd_chat_resume_cwd_restore_closes_database(monkeypatch, tmp_path):
    class StopAfterCwdRestore(Exception):
        pass

    db = MagicMock()
    db.get_session.return_value = {"cwd": str(tmp_path)}
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)
    monkeypatch.setattr("hermes_cli.main._resolve_use_tui", lambda _args: False)
    monkeypatch.setattr("hermes_cli.main._apply_safe_mode", lambda _args: None)
    monkeypatch.setattr(
        "hermes_cli.main._resolve_session_by_name_or_id",
        lambda session_id: session_id,
    )
    monkeypatch.setattr(
        "hermes_cli.main._has_any_provider_configured",
        MagicMock(side_effect=StopAfterCwdRestore),
    )
    monkeypatch.setattr("hermes_cli.main.os.chdir", MagicMock())
    args = SimpleNamespace(
        continue_last=None,
        resume="session-root",
        no_restore_cwd=False,
        worktree=False,
    )

    with pytest.raises(StopAfterCwdRestore):
        main_mod.cmd_chat(args)

    db.close.assert_called_once_with()
