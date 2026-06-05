from unittest.mock import MagicMock, patch

from cli import HermesCLI


class _InsightsEngineStub:
    calls = []

    def __init__(self, db):
        self.db = db

    def generate(self, *, days=30, source=None, workflow=True):
        self.calls.append({"days": days, "source": source, "workflow": workflow})
        return {"days": days, "source": source, "workflow": workflow}

    def format_terminal(self, report):
        return f"days={report['days']} source={report['source']} workflow={report['workflow']}"

    def format_markdown(self, report):
        return f"# days={report['days']} source={report['source']} workflow={report['workflow']}"

    def format_html(self, report):
        return f"<!doctype html><title>days={report['days']}</title>"


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

    assert calls == [{"days": 7, "source": None, "workflow": True}]
    db.close.assert_called_once()
    assert "days=7 source=None workflow=True" in capsys.readouterr().out


def test_cli_insights_keeps_days_flag_and_source(capsys):
    calls, db = _run_show_insights("/insights --days 14 --source discord")

    assert calls == [{"days": 14, "source": "discord", "workflow": True}]
    db.close.assert_called_once()
    assert "days=14 source=discord workflow=True" in capsys.readouterr().out


def test_cli_insights_can_disable_workflow_layer(capsys):
    calls, db = _run_show_insights("/insights --days 3 --no-recommendations")

    assert calls == [{"days": 3, "source": None, "workflow": False}]
    db.close.assert_called_once()
    assert "workflow=False" in capsys.readouterr().out


def test_cli_insights_can_render_markdown(capsys):
    calls, db = _run_show_insights("/insights --days 5 --markdown")

    assert calls == [{"days": 5, "source": None, "workflow": True}]
    db.close.assert_called_once()
    assert capsys.readouterr().out.startswith("# days=5")


def test_cli_insights_can_render_html(capsys):
    calls, db = _run_show_insights("/insights --days 5 --html")

    assert calls == [{"days": 5, "source": None, "workflow": True}]
    db.close.assert_called_once()
    assert capsys.readouterr().out.startswith("<!doctype html>")
