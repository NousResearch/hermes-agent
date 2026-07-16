"""Tests for the /behavior CLI command parsing.

Mirrors tests/cli/test_cli_insights_command.py: stubs BehavioralAnalyzer,
verifies --days / --source / positional arg parsing, config gate, and
that the SessionDB is closed after use.
"""

from unittest.mock import MagicMock, patch

from cli import HermesCLI


class _BehavioralAnalyzerStub:
    calls = []

    def __init__(self, db, config=None):
        self.db = db
        self.config = config

    def generate(self, *, days=30, source=None, user_id=None):
        self.calls.append({"days": days, "source": source, "user_id": user_id})
        return {"days": days, "source": source, "empty": False,
                "scores": {}, "cards": {}, "session_count": 1,
                "llm_available": False}

    def format_terminal(self, report):
        return f"days={report['days']} source={report['source']}"


def _run_show_behavior(command: str, config=None):
    """Invoke _show_behavior with stubs, return (calls, db)."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = config if config is not None else {"behavior": {"enabled": True}}
    db = MagicMock()
    _BehavioralAnalyzerStub.calls = []
    with patch("hermes_state.SessionDB", return_value=db), \
         patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
        cli_obj._show_behavior(command)
    return _BehavioralAnalyzerStub.calls, db


# =========================================================================
# Config gate
# =========================================================================

class TestBehaviorConfigGate:
    def test_disabled_prints_message(self, capsys):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.config = {"behavior": {"enabled": False}}
        cli_obj._show_behavior("/behavior")
        captured = capsys.readouterr()
        assert "disabled" in captured.out.lower()
        assert "behavior.enabled: true" in captured.out

    def test_no_config_section_disabled(self, capsys):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.config = {}
        cli_obj._show_behavior("/behavior")
        captured = capsys.readouterr()
        assert "disabled" in captured.out.lower()

    def test_no_behavior_key_disabled(self, capsys):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.config = {"behavior": {}}  # no enabled key
        cli_obj._show_behavior("/behavior")
        captured = capsys.readouterr()
        assert "disabled" in captured.out.lower()


# =========================================================================
# Argument parsing
# =========================================================================

class TestBehaviorArgParsing:
    def test_default_days_30(self, capsys):
        calls, db = _run_show_behavior("/behavior")
        assert calls == [{"days": 30, "source": None, "user_id": None}]
        db.close.assert_called_once()
        assert "days=30 source=None" in capsys.readouterr().out

    def test_positional_days(self, capsys):
        calls, db = _run_show_behavior("/behavior 7")
        assert calls == [{"days": 7, "source": None, "user_id": None}]
        db.close.assert_called_once()
        assert "days=7 source=None" in capsys.readouterr().out

    def test_days_flag(self, capsys):
        calls, db = _run_show_behavior("/behavior --days 14")
        assert calls == [{"days": 14, "source": None, "user_id": None}]
        db.close.assert_called_once()
        assert "days=14 source=None" in capsys.readouterr().out

    def test_source_flag(self, capsys):
        calls, db = _run_show_behavior("/behavior --source discord")
        assert calls == [{"days": 30, "source": "discord", "user_id": None}]
        db.close.assert_called_once()

    def test_days_and_source_flags(self, capsys):
        calls, db = _run_show_behavior("/behavior --days 14 --source discord")
        assert calls == [{"days": 14, "source": "discord", "user_id": None}]
        db.close.assert_called_once()
        assert "days=14 source=discord" in capsys.readouterr().out

    def test_invalid_days_value(self, capsys):
        """Non-integer --days value prints error, does not call analyzer."""
        calls, db = _run_show_behavior("/behavior --days abc")
        assert calls == []  # analyzer never called
        captured = capsys.readouterr()
        assert "Invalid" in captured.out or "invalid" in captured.out.lower()

    def test_positional_and_flag_combined(self, capsys):
        """Positional days followed by --source flag."""
        calls, db = _run_show_behavior("/behavior 7 --source cli")
        assert calls == [{"days": 7, "source": "cli", "user_id": None}]


# =========================================================================
# Config is passed to analyzer
# =========================================================================

class TestBehaviorConfigPassed:
    def test_config_dict_passed_to_analyzer(self):
        """The behavior config dict is passed to BehavioralAnalyzer."""
        captured_config = []

        class _CapturingStub:
            def __init__(self, db, config=None):
                captured_config.append(config)

            def generate(self, *, days=30, source=None, user_id=None):
                return {"days": days, "source": source, "empty": False,
                        "scores": {}, "cards": {}, "session_count": 0,
                        "llm_available": False}

            def format_terminal(self, report):
                return "ok"

        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.config = {"behavior": {"enabled": True, "model": "gpt-4o-mini"}}
        db = MagicMock()
        with patch("hermes_state.SessionDB", return_value=db), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _CapturingStub):
            cli_obj._show_behavior("/behavior")
        assert len(captured_config) == 1
        assert captured_config[0].get("model") == "gpt-4o-mini"