from unittest.mock import MagicMock, patch

from cli import HermesCLI
from hermes_constants import get_hermes_home


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
    with patch("hermes_state.SessionDB.for_home", return_value=db) as open_for_home, \
         patch("agent.insights.InsightsEngine", _InsightsEngineStub):
        cli_obj._show_insights(command)
    return _InsightsEngineStub.calls, db, open_for_home


def test_cli_insights_accepts_positional_days(capsys):
    calls, db, open_for_home = _run_show_insights("/insights 7")

    assert calls == [{"days": 7, "source": None}]
    open_for_home.assert_called_once_with(get_hermes_home())
    db.close.assert_called_once()
    assert "days=7 source=None" in capsys.readouterr().out


def test_cli_insights_keeps_days_flag_and_source(capsys):
    calls, db, open_for_home = _run_show_insights("/insights --days 14 --source discord")

    assert calls == [{"days": 14, "source": "discord"}]
    open_for_home.assert_called_once_with(get_hermes_home())
    db.close.assert_called_once()
    assert "days=14 source=discord" in capsys.readouterr().out
