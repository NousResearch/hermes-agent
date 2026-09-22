from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from cli import HermesCLI
from hermes_cli.main_agent_cmds import cmd_insights


@pytest.fixture(autouse=True)
def existing_store(tmp_path, monkeypatch):
    # insights short-circuits before opening when state.db is absent (fresh install).
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
    (tmp_path / "state.db").touch()
    return tmp_path / "state.db"


class _InsightsEngineStub:
    calls = []

    def __init__(self, db):
        self.db = db

    def generate(self, *, days=30, source=None):
        self.calls.append({"days": days, "source": source})
        return {"days": days, "source": source}

    def format_terminal(self, report):
        return f"days={report['days']} source={report['source']}"


def _run_show_insights(command: str):
    cli_obj = HermesCLI.__new__(HermesCLI)
    db = MagicMock()
    _InsightsEngineStub.calls = []
    with patch("hermes_state.SessionDB", return_value=db), \
         patch("agent.insights.InsightsEngine", _InsightsEngineStub):
        cli_obj._show_insights(command)
    return _InsightsEngineStub.calls, db


def test_cli_insights_accepts_positional_days(capsys):
    calls, db = _run_show_insights("/insights 7")

    assert calls == [{"days": 7, "source": None}]
    db.close.assert_called_once()
    assert "days=7 source=None" in capsys.readouterr().out


def test_cli_insights_keeps_days_flag_and_source(capsys):
    calls, db = _run_show_insights("/insights --days 14 --source discord")

    assert calls == [{"days": 14, "source": "discord"}]
    db.close.assert_called_once()
    assert "days=14 source=discord" in capsys.readouterr().out


def test_insights_skips_open_when_store_missing(capsys, existing_store):
    # Fresh install: no state.db yet → no open at all (a read-only open needs an existing file).
    existing_store.unlink()
    for run in (lambda: HermesCLI.__new__(HermesCLI)._show_insights("/insights 7"),
                lambda: cmd_insights(SimpleNamespace(days=30, source=None))):
        with patch("hermes_state.SessionDB") as ctor:
            run()
        ctor.assert_not_called()
    assert "No sessions found in the last" in capsys.readouterr().out


def test_show_insights_opens_read_only(existing_store):
    cli_obj = HermesCLI.__new__(HermesCLI)
    db = MagicMock()
    with patch("hermes_state.SessionDB", return_value=db) as ctor, \
         patch("agent.insights.InsightsEngine", _InsightsEngineStub):
        _InsightsEngineStub.calls = []
        cli_obj._show_insights("/insights 7")
    ctor.assert_called_once_with(db_path=existing_store, read_only=True)
    db.close.assert_called_once()


def test_subcommand_insights_opens_read_only(existing_store):
    db = MagicMock()
    with patch("hermes_state.SessionDB", return_value=db) as ctor, \
         patch("agent.insights.InsightsEngine", _InsightsEngineStub):
        cmd_insights(SimpleNamespace(days=30, source=None))
    ctor.assert_called_once_with(db_path=existing_store, read_only=True)
    db.close.assert_called_once()


def test_subcommand_insights_closes_database_when_generation_fails(capsys):
    db = MagicMock()
    with patch("hermes_state.SessionDB", return_value=db), \
         patch("agent.insights.InsightsEngine", side_effect=RuntimeError("boom")):
        cmd_insights(SimpleNamespace(days=30, source=None))

    db.close.assert_called_once()
    assert "Error generating insights: boom" in capsys.readouterr().out
