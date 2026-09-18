"""Tests for the ``/reload-skills`` CLI slash command (``HermesCLI._reload_skills``).

The CLI handler prints the diff (name + description) for the user and —
when any skills were added or removed — queues a one-shot note on
``self._pending_skills_reload_note``. The note is prepended to the NEXT
user message (see cli.py ~L8770, same pattern as
``_pending_model_switch_note``) and cleared after use, so no phantom user
turn is persisted to ``conversation_history``.
"""

from unittest.mock import patch


def _make_cli():
    """Build a minimal HermesCLI shell exposing ``_reload_skills``."""
    import cli as cli_mod

    obj = object.__new__(cli_mod.HermesCLI)
    obj._command_running = False
    obj.conversation_history = []
    obj.agent = None
    return obj


class TestReloadSkillsCLI:
    def test_reports_added_and_removed_and_queues_note(self, capsys):
        cli = _make_cli()
        with patch(
            "agent.skill_commands.reload_skills",
            return_value={
                "added": [
                    {"name": "alpha", "description": "Run alpha to do xyz"},
                    {"name": "beta", "description": "Run beta to do abc"},
                ],
                "removed": [
                    {"name": "gamma", "description": "Old removed skill"},
                ],
                "unchanged": ["delta"],
                "total": 3,
                "commands": 3,
            },
        ):
            cli._reload_skills()

        out = capsys.readouterr().out
        assert "Added Skills:" in out
        assert "- alpha: Run alpha to do xyz" in out
        assert "- beta: Run beta to do abc" in out
        assert "Removed Skills:" in out
        assert "- gamma: Old removed skill" in out
        assert "3 skill(s) available" in out

        # Must NOT pollute conversation_history — alternation-safe.
        assert cli.conversation_history == []

        # One-shot note queued with system-prompt-style formatting.
        note = getattr(cli, "_pending_skills_reload_note", None)
        assert note is not None
        assert note.startswith("[USER INITIATED SKILLS RELOAD:")
        assert note.endswith("Use skills_list to see the updated catalog.]")
        assert "Added Skills:" in note
        assert "    - alpha: Run alpha to do xyz" in note
        assert "    - beta: Run beta to do abc" in note
        assert "Removed Skills:" in note
        assert "    - gamma: Old removed skill" in note

    def test_reports_no_changes_and_queues_nothing(self, capsys):
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
        assert "No new skills detected" in out
        assert "1 skill(s) available" in out
        assert cli.conversation_history == []
        assert getattr(cli, "_pending_skills_reload_note", None) is None

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
        assert cli.conversation_history == []
        assert getattr(cli, "_pending_skills_reload_note", None) is None
