"""Test that ``/reload-skills`` syncs cli.py's module-level ``_skill_commands``.

Regression test for https://github.com/NousResearch/hermes-agent/issues/26441.

The bug: ``SlashCommandCompleter`` captures cli.py's ``_skill_commands`` via a
lambda at construction time.  ``_reload_skills()`` called
``agent.skill_commands.reload_skills()`` but never updated cli.py's own
module-level ``_skill_commands``, so the Tab-completion lambda kept seeing the
stale dict from startup.
"""

import shutil
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _write_skill(skills_dir: Path, name: str, description: str = "") -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: {description or f'{name} skill'}
            ---
            body
            """
        )
    )
    return skill_dir


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    """Set up an isolated HERMES_HOME with a skills dir."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "skills").mkdir()

    import tools.skills_tool as _st
    import agent.skill_commands as _sc

    monkeypatch.setattr(_st, "HERMES_HOME", home, raising=False)
    monkeypatch.setattr(_st, "SKILLS_DIR", home / "skills", raising=False)
    monkeypatch.setattr(_sc, "_skill_commands", {}, raising=False)

    yield home


class TestReloadSkillsTabCompletionSync:
    """After ``_reload_skills()``, cli.py's ``_skill_commands`` must reflect
    the new skill set so the Tab-completion lambda sees fresh data."""

    def test_skill_commands_updates_after_reload(self, isolated_home, monkeypatch):
        """_skill_commands in cli.py is updated after _reload_skills()."""
        skills_dir = isolated_home / "skills"
        # Pre-populate one skill
        _write_skill(skills_dir, "alpha", "Alpha skill")

        # Import cli.py's module-level _skill_commands
        import cli as cli_mod

        # Simulate initial scan
        from agent.skill_commands import scan_skill_commands
        monkeypatch.setattr(cli_mod, "_skill_commands", scan_skill_commands())
        initial = cli_mod._skill_commands.copy()
        assert "/alpha" in initial

        # Add a new skill
        _write_skill(skills_dir, "beta", "Beta skill")

        # Build a minimal mock CLI instance and call _reload_skills
        mock_cli = MagicMock(spec=cli_mod.HermesCLI)
        mock_cli._command_running = False
        mock_cli._pending_skills_reload_note = None

        # Call the actual method
        cli_mod.HermesCLI._reload_skills(mock_cli)

        # The module-level _skill_commands should now include /beta
        updated = cli_mod._skill_commands
        assert "/beta" in updated, (
            f"/beta not in _skill_commands after reload. "
            f"Keys: {sorted(updated.keys())}"
        )
        assert "/alpha" in updated

    def test_skill_commands_reflects_removal(self, isolated_home, monkeypatch):
        """Removing a skill dir and reloading drops it from _skill_commands."""
        skills_dir = isolated_home / "skills"
        _write_skill(skills_dir, "gamma", "Gamma skill")
        _write_skill(skills_dir, "delta", "Delta skill")

        import cli as cli_mod
        from agent.skill_commands import scan_skill_commands
        monkeypatch.setattr(cli_mod, "_skill_commands", scan_skill_commands())
        assert "/gamma" in cli_mod._skill_commands
        assert "/delta" in cli_mod._skill_commands

        # Remove gamma
        shutil.rmtree(skills_dir / "gamma")

        mock_cli = MagicMock(spec=cli_mod.HermesCLI)
        mock_cli._command_running = False
        mock_cli._pending_skills_reload_note = None
        cli_mod.HermesCLI._reload_skills(mock_cli)

        assert "/gamma" not in cli_mod._skill_commands
        assert "/delta" in cli_mod._skill_commands
