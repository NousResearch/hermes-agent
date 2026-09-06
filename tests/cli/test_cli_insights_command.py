from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from cli import HermesCLI
from hermes_cli.main_agent_cmds import cmd_insights


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


def test_subcommand_insights_closes_database_when_generation_fails(capsys):
    db = MagicMock()
    with patch("hermes_state.SessionDB", return_value=db), \
         patch("agent.insights.InsightsEngine", side_effect=RuntimeError("boom")):
        cmd_insights(SimpleNamespace(days=30, source=None))

    db.close.assert_called_once()
    assert "Error generating insights: boom" in capsys.readouterr().out
