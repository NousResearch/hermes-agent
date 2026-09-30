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
    def test_renders_real_plugin_removal_and_queues_next_turn_note(self, tmp_path, monkeypatch, capsys):
        import agent.skill_commands as skill_commands
        from hermes_cli import plugins

        home = tmp_path / "home"
        plugin = home / "plugins" / "render-probe"
        skill = plugin / "skills" / "guide" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: render-probe\nversion: 0.1.0\n")
        (plugin / "__init__.py").write_text(
            "from pathlib import Path\ndef register(ctx):\n"
            "    ctx.register_skill('guide', Path(__file__).parent / 'skills' / 'guide' / 'SKILL.md')\n")
        skill.write_text("---\nname: guide\ndescription: Rendered guide.\n---\nBody.\n")
        config = home / "config.yaml"
        config.write_text("plugins:\n  enabled: [render-probe]\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        plugins._reset_plugin_managers_for_tests()
        try:
            assert "/render-probe:guide" in skill_commands.get_interactive_skill_commands()
            config.write_text("plugins:\n  enabled: []\n")
            plugins.discover_plugins(force=True)
            skill_commands.invalidate_plugin_skill_commands()

            cli = _make_cli()
            cli._reload_skills()
            output = capsys.readouterr().out
            assert "render-probe:guide" in output
            assert "0 skill(s) available" in output
            assert "render-probe:guide" in cli._pending_skills_reload_note
        finally:
            plugins._reset_plugin_managers_for_tests()

    def test_reload_keeps_plugin_skill_lookup_live_after_plugin_lifecycle(self, tmp_path, monkeypatch):
        import cli as cli_mod
        from hermes_cli import plugins

        home = tmp_path / "home"
        plugin = home / "plugins" / "live-probe"
        skill = plugin / "skills" / "guide" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: live-probe\nversion: 0.1.0\n")
        (plugin / "__init__.py").write_text(
            "from pathlib import Path\ndef register(ctx):\n"
            "    ctx.register_skill('guide', Path(__file__).parent / 'skills' / 'guide' / 'SKILL.md')\n")
        skill.write_text("---\nname: guide\ndescription: Live guide.\n---\nBody.\n")
        config = home / "config.yaml"
        config.write_text("plugins:\n  enabled: [live-probe]\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(cli_mod, "_skill_commands", None)
        plugins._reset_plugin_managers_for_tests()
        try:
            shell = _make_cli()
            shell._reload_skills()
            assert "/live-probe:guide" in cli_mod.get_skill_commands()
            config.write_text("plugins:\n  enabled: [live-probe]\nskills:\n  disabled: [guide]\n")
            assert "/live-probe:guide" not in cli_mod.get_skill_commands()
            config.write_text("plugins:\n  enabled: [live-probe]\n")
            assert "/live-probe:guide" in cli_mod.get_skill_commands()
        finally:
            plugins._reset_plugin_managers_for_tests()

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
        for name in ("alpha", "beta", "gamma"):
            assert name in out

        # Must NOT pollute conversation_history — alternation-safe.
        assert cli.conversation_history == []

        # One-shot note queued with system-prompt-style formatting.
        note = getattr(cli, "_pending_skills_reload_note", None)
        assert note is not None
        for name in ("alpha", "beta", "gamma"):
            assert name in note

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

        assert cli.conversation_history == []
        assert getattr(cli, "_pending_skills_reload_note", None) is None

    def test_handles_reload_failure_gracefully(self, capsys):
        cli = _make_cli()
        with patch(
            "agent.skill_commands.reload_skills",
            side_effect=RuntimeError("boom"),
        ):
            cli._reload_skills()

        assert "boom" in capsys.readouterr().out
        assert cli.conversation_history == []
        assert getattr(cli, "_pending_skills_reload_note", None) is None
