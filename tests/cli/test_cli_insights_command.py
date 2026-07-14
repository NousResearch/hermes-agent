from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


class _InsightsEngineStub:
    calls = []

    def __init__(self, db):
        self.db = db

    def generate(self, *, days=30, source=None):
        self.calls.append({"days": days, "source": source})
        return {"days": days, "source": source}

    def format_terminal(self, report):
        return f"days={report['days']} source={report['source']}"


def _run_show_insights(command: str, *, app_active: bool = False):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.console = MagicMock()
    cli_obj._app = object() if app_active else None
    db = MagicMock()
    live_console = MagicMock()
    _InsightsEngineStub.calls = []
    with (
        patch("hermes_state.SessionDB", return_value=db),
        patch("agent.insights.InsightsEngine", _InsightsEngineStub),
        patch("cli.ChatConsole", return_value=live_console),
    ):
        cli_obj._show_insights(command)
    return _InsightsEngineStub.calls, db, cli_obj.console, live_console


def test_cli_insights_accepts_positional_days():
    calls, db, console, live_console = _run_show_insights("/insights 7")

    assert calls == [{"days": 7, "source": None}]
    db.close.assert_called_once()
    console.print.assert_called_once_with("days=7 source=None")
    live_console.print.assert_not_called()


def test_cli_insights_uses_live_chat_console():
    calls, db, console, live_console = _run_show_insights(
        "/insights --days 14 --source discord",
        app_active=True,
    )

    assert calls == [{"days": 14, "source": "discord"}]
    db.close.assert_called_once()
    live_console.print.assert_called_once_with("days=14 source=discord")
    console.print.assert_not_called()


def test_cli_insights_validation_error_uses_live_chat_console():
    calls, db, console, live_console = _run_show_insights(
        "/insights --days nope",
        app_active=True,
    )

    assert calls == []
    db.close.assert_not_called()
    live_console.print.assert_called_once_with("  Invalid --days value: nope")
    console.print.assert_not_called()


def test_standalone_insights_uses_rich_console():
    from hermes_cli.main import cmd_insights

    db = MagicMock()
    console = MagicMock()
    _InsightsEngineStub.calls = []
    args = SimpleNamespace(days=21, source="telegram")
    with (
        patch("hermes_state.SessionDB", return_value=db),
        patch("agent.insights.InsightsEngine", _InsightsEngineStub),
        patch("rich.console.Console", return_value=console),
    ):
        cmd_insights(args)

    assert _InsightsEngineStub.calls == [{"days": 21, "source": "telegram"}]
    console.print.assert_called_once_with("days=21 source=telegram")
    db.close.assert_called_once()
