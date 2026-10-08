"""Regression for #118142: selected-store status errors remain visible and traceable."""

import logging
import traceback
from types import SimpleNamespace

from hermes_cli.status import _render_sessions
from state_store_runtime_readiness import trap_state_db_opens


def _assert_selected_store_failure(home, capsys, caplog, expected_error, expected_log):
    output = capsys.readouterr().out
    assert f"error reading selected state store: {expected_error}" in output
    assert "Active:       0" not in output
    assert not (home / "state.db").exists()

    records = [record for record in caplog.records if record.name == "hermes_cli.status"]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.ERROR
    assert record.getMessage() == expected_log
    assert record.exc_info is not None
    assert expected_error in "".join(traceback.format_exception(*record.exc_info))


def test_invalid_backend_reports_error_and_traceback_without_sqlite(tmp_path, monkeypatch, capsys, caplog):
    home = tmp_path / ".hermes"
    home.mkdir()
    config = {"state_store": {"backend": "invalid"}}
    (home / "config.yaml").write_text("state_store:\n  backend: invalid\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    with caplog.at_level(logging.ERROR, logger="hermes_cli.status"), trap_state_db_opens(home) as events:
        _render_sessions(SimpleNamespace(config=config))

    assert events == []
    _assert_selected_store_failure(
        home, capsys, caplog, "state_store.backend must be one of:",
        "Failed to resolve selected state store for status",
    )


def test_unavailable_selected_postgresql_reports_error_and_traceback_without_sqlite(
    tmp_path, monkeypatch, capsys, caplog,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    config = {"state_store": {"backend": "postgresql", "postgresql": {"dsn_env": "STATE_STORE_TEST_DSN"}}}
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: STATE_STORE_TEST_DSN\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("STATE_STORE_TEST_DSN", "postgresql://unused/never-connected")
    import cli_session_store

    def unavailable(_config):
        raise RuntimeError("simulated selected-store outage")

    monkeypatch.setattr(cli_session_store, "open_selected_read_store", unavailable)
    with caplog.at_level(logging.ERROR, logger="hermes_cli.status"), trap_state_db_opens(home) as events:
        _render_sessions(SimpleNamespace(config=config))

    assert events == []
    _assert_selected_store_failure(
        home, capsys, caplog, "simulated selected-store outage",
        "Failed to read selected PostgreSQL state store for status",
    )
