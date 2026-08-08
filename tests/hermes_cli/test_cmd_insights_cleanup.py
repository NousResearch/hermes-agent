"""SessionDB lifecycle coverage for the CLI insights command."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_cli.main import cmd_insights


def _args():
    return SimpleNamespace(days=30, source=None)


def test_cmd_insights_closes_database_after_success(monkeypatch, capsys):
    db = MagicMock()
    engine = MagicMock()
    engine.generate.return_value = {"summary": "ok"}
    engine.format_terminal.return_value = "formatted"
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)
    monkeypatch.setattr("agent.insights.InsightsEngine", lambda _db: engine)

    cmd_insights(_args())

    assert capsys.readouterr().out.strip() == "formatted"
    db.close.assert_called_once_with()


def test_cmd_insights_closes_database_after_failure(monkeypatch, capsys):
    db = MagicMock()
    engine = MagicMock()
    engine.generate.side_effect = RuntimeError("analytics failed")
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)
    monkeypatch.setattr("agent.insights.InsightsEngine", lambda _db: engine)

    cmd_insights(_args())

    assert "analytics failed" in capsys.readouterr().out
    db.close.assert_called_once_with()
