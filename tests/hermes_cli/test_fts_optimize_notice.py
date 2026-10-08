"""Regression coverage for FTS storage upgrade discoverability."""

import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from state_store_runtime_readiness import trap_state_db_opens


@pytest.fixture
def notice_probe_home(tmp_path, monkeypatch):
    """Existing large SQLite file makes a mistaken fallback reach the open path."""
    import hermes_constants
    import hermes_state

    db_path = tmp_path / "state.db"
    db_path.touch()
    with db_path.open("r+b") as db_file:
        db_file.truncate(600 * 1024 * 1024)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    session_db = Mock(side_effect=AssertionError("SQLite SessionDB must not open"))
    monkeypatch.setattr(hermes_state, "SessionDB", session_db)
    return tmp_path, session_db


def test_update_notice_unreadable_config_never_resolves_or_probes(
    notice_probe_home, monkeypatch, caplog, capsys,
):
    from hermes_cli import config, update_cmd
    import state_store

    home, session_db = notice_probe_home
    resolve = Mock(side_effect=AssertionError("unreadable config must not resolve to SQLite"))
    monkeypatch.setattr(state_store, "resolve_state_store_config", resolve)
    monkeypatch.setattr(config, "load_config", Mock(side_effect=RuntimeError("private payload")))

    with trap_state_db_opens(home) as events:
        update_cmd._print_fts_optimize_available_notice()

    assert resolve.call_count == 0
    session_db.assert_not_called()
    assert events == []
    assert capsys.readouterr().out == ""
    assert any(
        record.exc_info and "private payload" not in record.msg
        for record in caplog.records
    )


def test_update_notice_malformed_backend_never_probes(notice_probe_home, monkeypatch, capsys):
    from hermes_cli import config, update_cmd

    home, session_db = notice_probe_home
    monkeypatch.setattr(config, "load_config", lambda: {"state_store": {"backend": "invalid"}})

    with trap_state_db_opens(home) as events:
        update_cmd._print_fts_optimize_available_notice()

    session_db.assert_not_called()
    assert events == []
    assert capsys.readouterr().out == ""


def test_update_notice_selected_postgresql_never_probes(notice_probe_home, monkeypatch, capsys):
    from hermes_cli import config, update_cmd

    home, session_db = notice_probe_home
    monkeypatch.setenv("HERMES_FTS_TEST_DSN", "postgresql://unused@localhost/unused")
    monkeypatch.setattr(config, "load_config", lambda: {
        "state_store": {"backend": "postgresql", "postgresql": {"dsn_env": "HERMES_FTS_TEST_DSN"}},
    })

    with trap_state_db_opens(home) as events:
        update_cmd._print_fts_optimize_available_notice()

    session_db.assert_not_called()
    assert events == []
    assert capsys.readouterr().out == ""


def test_update_notice_unexpected_resolver_error_propagates_before_db(
    notice_probe_home, monkeypatch,
):
    from hermes_cli import config, update_cmd
    import state_store

    home, session_db = notice_probe_home
    monkeypatch.setattr(config, "load_config", lambda: {})
    monkeypatch.setattr(
        state_store, "resolve_state_store_config",
        Mock(side_effect=RuntimeError("resolver exploded")),
    )

    with trap_state_db_opens(home) as events:
        with pytest.raises(RuntimeError, match="resolver exploded"):
            update_cmd._print_fts_optimize_available_notice()

    session_db.assert_not_called()
    assert events == []


def test_update_notice_offers_v1_trigram_tool_calls_rebuild(tmp_path, monkeypatch, capsys):
    """A deployed v1 trigram projection still receives the opt-in notice."""
    from hermes_cli import update_cmd
    import hermes_constants
    import hermes_state

    db_path = tmp_path / "state.db"
    db_path.touch()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE messages_fts (content TEXT, tool_name TEXT, tool_calls TEXT);
        CREATE TABLE messages_fts_trigram (content TEXT, tool_name TEXT, tool_calls TEXT);
        """
    )

    class FakeSessionDB:
        def __init__(self, **_kwargs):
            self._conn = conn

        def close(self):
            pass

        _db_needs_fts_storage_upgrade = staticmethod(
            hermes_state.SessionDB._db_needs_fts_storage_upgrade
        )

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(hermes_state, "SessionDB", FakeSessionDB)
    # Report a large state.db without patching Path.stat globally: a
    # 1-arg lambda on the class breaks pathlib.exists(follow_symlinks=...)
    # for every caller in the process (pytest's own teardown included).
    real_stat = update_cmd.Path.stat

    def _stat(path, *args, **kwargs):
        if path.name == "state.db":
            return SimpleNamespace(st_size=512 * 1024 ** 2)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(update_cmd.Path, "stat", _stat)

    update_cmd._print_fts_optimize_available_notice()

    assert "hermes sessions optimize-storage" in capsys.readouterr().out
    conn.close()
