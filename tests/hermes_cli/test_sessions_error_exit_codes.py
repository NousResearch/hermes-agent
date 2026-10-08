"""Regression tests: `hermes sessions` error paths return non-zero (SES-04).

Before this, delete/rename not-found, prune bad-arg, blank rename, and import
of a missing file all printed an error and returned exit 0 — a scripting/CI
hazard (a script pinning a bad id failed loudly via `pin` but deleting a bad
id "succeeded" silently). The subcommand dispatcher already maps an int
handler return to the process exit code; these tests pin the returns.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest

import hermes_cli.sessions_cmd as sc


def _args(action, **kw):
    base = dict(
        sessions_action=action,
        session_id=None, title=None, yes=True, source=None, path=None,
        from_source=None, dry_run=False, older_than=None, newer_than=None,
        before=None, after=None, limit=50,
    )
    base.update(kw)
    return Namespace(**base)


def test_delete_missing_returns_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    SessionDB(tmp_path / "state.db")  # initialize an empty store
    rc = sc.cmd_sessions(_args("delete", session_id="nope_xyz"))
    assert rc == 1
    out = capsys.readouterr().out
    assert "No session 'nope_xyz'" in out and "hermes sessions list" in out


def test_rename_missing_returns_1(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    SessionDB(tmp_path / "state.db")
    rc = sc.cmd_sessions(_args("rename", session_id="nope_xyz", title=["New"]))
    assert rc == 1


def test_import_missing_file_returns_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rc = sc.cmd_sessions(_args("import", path=str(tmp_path / "nope.jsonl")))
    assert rc == 1
    assert "file not found" in capsys.readouterr().out.lower()


@pytest.fixture
def selected_postgresql(monkeypatch):
    import hermes_cli.config as config_module
    import hermes_state
    import state_store

    config = {"state_store": {"backend": "postgresql"}}
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(state_store, "resolve_state_store_config", lambda _: SimpleNamespace(backend="postgresql"))
    def no_sqlite(*args, **kwargs):
        pytest.fail("selected PostgreSQL must not open SQLite SessionDB")
    monkeypatch.setattr(hermes_state, "SessionDB", no_sqlite)
    return config


@pytest.mark.parametrize("action", ["import", "list"])
def test_postgresql_opener_failure_logs_traceback_without_sqlite_fallback(
    selected_postgresql, monkeypatch, caplog, capsys, action,
):
    import cli_session_store

    def failed_open(*args, **kwargs):
        raise RuntimeError("offline opener failed")

    monkeypatch.setattr(cli_session_store, "open_cli_session_store", failed_open)
    with caplog.at_level("ERROR", logger=sc.__name__):
        assert sc.cmd_sessions(_args(action)) == 1
    assert "Could not open your PostgreSQL session history: offline opener failed" in capsys.readouterr().out
    assert len(caplog.records) == 1
    expected_context = "import" if action == "import" else "action"
    assert caplog.records[0].getMessage() == f"Could not open PostgreSQL session history for {expected_context}"
    assert caplog.records[0].exc_info[0] is RuntimeError
    assert "failed_open" in caplog.text


def test_config_resolution_failure_logs_traceback_and_returns_nonzero(monkeypatch, caplog, capsys):
    import hermes_cli.config as config_module
    import hermes_state

    def failed_config():
        raise RuntimeError("offline config failure")

    monkeypatch.setattr(config_module, "load_config", failed_config)
    monkeypatch.setattr(hermes_state, "SessionDB", lambda **_: pytest.fail("SQLite fallback"))
    with caplog.at_level("ERROR", logger=sc.__name__):
        assert sc.cmd_sessions(_args("list")) == 1
    assert "Could not resolve your session history store: offline config failure; no SQLite fallback is permitted." in capsys.readouterr().out
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage() == "Could not resolve session history store configuration"
    assert caplog.records[0].exc_info[0] is RuntimeError
    assert "failed_config" in caplog.text


def test_postgresql_optimize_preflight_skips_generic_opener(selected_postgresql, monkeypatch):
    import cli_session_store

    monkeypatch.setattr(cli_session_store, "open_cli_session_store", lambda *a, **k: pytest.fail("generic opener"))
    monkeypatch.setattr(sc, "_cmd_postgresql_optimize", lambda config: 7 if config is selected_postgresql else pytest.fail("config"))
    assert sc.cmd_sessions(_args("optimize")) == 7


@pytest.mark.parametrize("action", ["optimize-storage", "export"])
def test_postgresql_unsupported_preflight_skips_store(selected_postgresql, monkeypatch, capsys, action):
    import cli_session_store

    monkeypatch.setattr(cli_session_store, "open_cli_session_store", lambda *a, **k: pytest.fail("generic opener"))
    assert sc.cmd_sessions(_args(action)) == 2
    assert "no SQLite fallback is permitted" in capsys.readouterr().out


@pytest.mark.parametrize("action", ["prune", "archive", "clean-markers"])
def test_postgresql_maintenance_refusal_skips_store(selected_postgresql, monkeypatch, capsys, action):
    import cli_session_store
    import state_store_maintenance as maintenance

    monkeypatch.setattr(cli_session_store, "open_cli_session_store", lambda *a, **k: pytest.fail("generic opener"))
    def refused(*args, **kwargs):
        raise maintenance.StateStoreMaintenanceError("maintenance denied")
    monkeypatch.setattr(maintenance, "require_state_store_maintenance", refused)
    assert sc.cmd_sessions(_args(action)) == 2
    assert "maintenance denied" in capsys.readouterr().out
