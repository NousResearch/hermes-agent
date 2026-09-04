"""Tests for `hermes insights --json` (machine-readable insights report)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _args(**kw):
    base = {"days": 7, "source": None, "json": True}
    base.update(kw)
    return SimpleNamespace(**base)


def test_insights_json_outputs_report(capsys):
    from hermes_cli.main import cmd_insights

    report = {
        "days": 7,
        "source_filter": None,
        "empty": True,
        "overview": {},
        "models": [],
        "platforms": [],
        "tools": [],
        "skills": {"summary": {}, "top_skills": []},
        "activity": {},
        "top_sessions": [],
    }
    with (
        patch("hermes_state.SessionDB") as mock_db,
        patch("agent.insights.InsightsEngine") as mock_engine,
    ):
        mock_engine.return_value.generate.return_value = report
        mock_db.return_value = MagicMock()
        cmd_insights(_args())

    data = json.loads(capsys.readouterr().out)
    assert data["days"] == 7
    assert data["empty"] is True
    # format_terminal must not have been used
    mock_engine.return_value.format_terminal.assert_not_called()


def test_insights_json_error_is_json(capsys):
    from hermes_cli.main import cmd_insights

    with patch("hermes_state.SessionDB", side_effect=RuntimeError("boom")):
        cmd_insights(_args())

    data = json.loads(capsys.readouterr().out)
    assert "boom" in data["error"]


def test_insights_text_path_unchanged(capsys):
    from hermes_cli.main import cmd_insights

    with (
        patch("hermes_state.SessionDB") as mock_db,
        patch("agent.insights.InsightsEngine") as mock_engine,
    ):
        mock_engine.return_value.generate.return_value = {"empty": True}
        mock_engine.return_value.format_terminal.return_value = "TERMINAL-REPORT"
        mock_db.return_value = MagicMock()
        cmd_insights(_args(json=False))

    out = capsys.readouterr().out
    assert "TERMINAL-REPORT" in out
