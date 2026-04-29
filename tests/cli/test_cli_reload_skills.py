"""Tests for the ``/reload-skills`` CLI slash command (``HermesCLI._reload_skills``)."""

from unittest.mock import MagicMock, patch


def _make_cli():
    """Build a minimal HermesCLI shell exposing ``_reload_skills``."""
    import cli as cli_mod

    obj = object.__new__(cli_mod.HermesCLI)
    obj._command_running = False
    obj.conversation_history = []
    obj.agent = None
    return obj


class TestReloadSkillsCLI:
    def test_reports_added_and_removed(self, capsys):
        cli = _make_cli()
        with patch(
            "agent.skill_commands.reload_skills",
            return_value={
                "added": ["alpha", "beta"],
                "removed": ["gamma"],
                "unchanged": ["delta"],
                "total": 3,
                "commands": 3,
            },
        ):
            cli._reload_skills()

        out = capsys.readouterr().out
        assert "Added: alpha, beta" in out
        assert "Removed: gamma" in out
        assert "3 skill(s) available" in out
        # An informational message should be appended to the conversation
        # so the model picks up the diff on the next turn.
        assert len(cli.conversation_history) == 1
        msg = cli.conversation_history[0]
        assert msg["role"] == "user"
        assert "Skills have been reloaded" in msg["content"]
        assert "alpha" in msg["content"]
        assert "gamma" in msg["content"]

    def test_reports_no_changes(self, capsys):
        cli = _make_cli()
        with patch(
            "agent.skill_commands.reload_skills",
            return_value={
                "added": [],
                "removed": [],
                "unchanged": ["alpha"],
                "total": 1,
                "commands": 1,
            },
        ):
            cli._reload_skills()

        out = capsys.readouterr().out
        assert "No changes detected" in out
        assert "1 skill(s) available" in out
        # Nothing changed — don't pollute history.
        assert cli.conversation_history == []

    def test_handles_reload_failure_gracefully(self, capsys):
        cli = _make_cli()
        with patch(
            "agent.skill_commands.reload_skills",
            side_effect=RuntimeError("boom"),
        ):
            cli._reload_skills()

        out = capsys.readouterr().out
        assert "Skills reload failed" in out
        assert "boom" in out
        # Failure must not append a misleading "skills reloaded" note.
        assert cli.conversation_history == []
