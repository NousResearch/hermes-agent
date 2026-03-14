"""Tests for hermes claw commands."""

from argparse import Namespace
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import claw as claw_mod


# ---------------------------------------------------------------------------
# _find_migration_script
# ---------------------------------------------------------------------------


class TestFindMigrationScript:
    """Test script discovery in known locations."""

    def test_finds_project_root_script(self, tmp_path):
        script = tmp_path / "openclaw_to_hermes.py"
        script.write_text("# placeholder")
        with patch.object(claw_mod, "_OPENCLAW_SCRIPT", script):
            assert claw_mod._find_migration_script() == script

    def test_finds_installed_script(self, tmp_path):
        installed = tmp_path / "installed.py"
        installed.write_text("# placeholder")
        with (
            patch.object(claw_mod, "_OPENCLAW_SCRIPT", tmp_path / "nonexistent.py"),
            patch.object(claw_mod, "_OPENCLAW_SCRIPT_INSTALLED", installed),
        ):
            assert claw_mod._find_migration_script() == installed

    def test_returns_none_when_missing(self, tmp_path):
        with (
            patch.object(claw_mod, "_OPENCLAW_SCRIPT", tmp_path / "a.py"),
            patch.object(claw_mod, "_OPENCLAW_SCRIPT_INSTALLED", tmp_path / "b.py"),
        ):
            assert claw_mod._find_migration_script() is None


# ---------------------------------------------------------------------------
# claw_command routing
# ---------------------------------------------------------------------------


class TestClawCommand:
    """Test the claw_command router."""

    def test_routes_to_migrate(self):
        args = Namespace(claw_action="migrate", source=None, dry_run=True,
                         preset="full", overwrite=False, migrate_secrets=False,
                         workspace_target=None, skill_conflict="skip", yes=False)
        with patch.object(claw_mod, "_cmd_migrate") as mock:
            claw_mod.claw_command(args)
        mock.assert_called_once_with(args)

    def test_shows_help_for_no_action(self, capsys):
        args = Namespace(claw_action=None)
        claw_mod.claw_command(args)
        captured = capsys.readouterr()
        assert "migrate" in captured.out


# ---------------------------------------------------------------------------
# _cmd_migrate
# ---------------------------------------------------------------------------


class TestCmdMigrate:
    """Test the migrate command handler."""

    def test_error_when_source_missing(self, tmp_path, capsys):
        args = Namespace(
            source=str(tmp_path / "nonexistent"),
            dry_run=True, preset="full", overwrite=False,
            migrate_secrets=False, workspace_target=None,
            skill_conflict="skip", yes=False,
        )
        claw_mod._cmd_migrate(args)
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_error_when_script_missing(self, tmp_path, capsys):
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        args = Namespace(
            source=str(openclaw_dir),
            dry_run=True, preset="full", overwrite=False,
            migrate_secrets=False, workspace_target=None,
            skill_conflict="skip", yes=False,
        )
        with (
            patch.object(claw_mod, "_OPENCLAW_SCRIPT", tmp_path / "a.py"),
            patch.object(claw_mod, "_OPENCLAW_SCRIPT_INSTALLED", tmp_path / "b.py"),
        ):
            claw_mod._cmd_migrate(args)
        captured = capsys.readouterr()
        assert "Migration script not found" in captured.out

    def test_dry_run_succeeds(self, tmp_path, capsys):
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        script = tmp_path / "script.py"
        script.write_text("# placeholder")

        # Build a fake migration module
        fake_mod = ModuleType("openclaw_to_hermes")
        fake_mod.resolve_selected_options = MagicMock(return_value={"soul", "memory"})
        fake_migrator = MagicMock()
        fake_migrator.migrate.return_value = {
            "summary": {"migrated": 0, "skipped": 5, "conflict": 0, "error": 0},
            "items": [
                {"kind": "soul", "status": "skipped", "reason": "Not found"},
            ],
            "preset": "full",
        }
        fake_mod.Migrator = MagicMock(return_value=fake_migrator)

        args = Namespace(
            source=str(openclaw_dir),
            dry_run=True, preset="full", overwrite=False,
            migrate_secrets=False, workspace_target=None,
            skill_conflict="skip", yes=False,
        )

        with (
            patch.object(claw_mod, "_find_migration_script", return_value=script),
            patch.object(claw_mod, "_load_migration_module", return_value=fake_mod),
            patch.object(claw_mod, "get_config_path", return_value=tmp_path / "config.yaml"),
            patch.object(claw_mod, "save_config"),
            patch.object(claw_mod, "load_config", return_value={}),
        ):
            claw_mod._cmd_migrate(args)

        captured = capsys.readouterr()
        assert "Dry Run Results" in captured.out
        assert "5 skipped" in captured.out

    def test_execute_with_confirmation(self, tmp_path, capsys):
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        config_path = tmp_path / "config.yaml"
        config_path.write_text("agent:\n  max_turns: 90\n")

        fake_mod = ModuleType("openclaw_to_hermes")
        fake_mod.resolve_selected_options = MagicMock(return_value={"soul"})
        fake_migrator = MagicMock()
        fake_migrator.migrate.return_value = {
            "summary": {"migrated": 2, "skipped": 1, "conflict": 0, "error": 0},
            "items": [
                {"kind": "soul", "status": "migrated", "destination": str(tmp_path / "SOUL.md")},
                {"kind": "memory", "status": "migrated", "destination": str(tmp_path / "memories/MEMORY.md")},
            ],
        }
        fake_mod.Migrator = MagicMock(return_value=fake_migrator)

        args = Namespace(
            source=str(openclaw_dir),
            dry_run=False, preset="user-data", overwrite=False,
            migrate_secrets=False, workspace_target=None,
            skill_conflict="skip", yes=False,
        )

        with (
            patch.object(claw_mod, "_find_migration_script", return_value=tmp_path / "s.py"),
            patch.object(claw_mod, "_load_migration_module", return_value=fake_mod),
            patch.object(claw_mod, "get_config_path", return_value=config_path),
            patch.object(claw_mod, "prompt_yes_no", return_value=True),
        ):
            claw_mod._cmd_migrate(args)

        captured = capsys.readouterr()
        assert "Migration Results" in captured.out
        assert "Migration complete!" in captured.out

    def test_execute_cancelled_by_user(self, tmp_path, capsys):
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        args = Namespace(
            source=str(openclaw_dir),
            dry_run=False, preset="full", overwrite=False,
            migrate_secrets=False, workspace_target=None,
            skill_conflict="skip", yes=False,
        )

        with (
            patch.object(claw_mod, "_find_migration_script", return_value=tmp_path / "s.py"),
            patch.object(claw_mod, "prompt_yes_no", return_value=False),
        ):
            claw_mod._cmd_migrate(args)

        captured = capsys.readouterr()
        assert "Migration cancelled" in captured.out

    def test_execute_with_yes_skips_confirmation(self, tmp_path, capsys):
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        fake_mod = ModuleType("openclaw_to_hermes")
        fake_mod.resolve_selected_options = MagicMock(return_value=set())
        fake_migrator = MagicMock()
        fake_migrator.migrate.return_value = {
            "summary": {"migrated": 0, "skipped": 0, "conflict": 0, "error": 0},
            "items": [],
        }
        fake_mod.Migrator = MagicMock(return_value=fake_migrator)

        args = Namespace(
            source=str(openclaw_dir),
            dry_run=False, preset="full", overwrite=False,
            migrate_secrets=False, workspace_target=None,
            skill_conflict="skip", yes=True,
        )

        with (
            patch.object(claw_mod, "_find_migration_script", return_value=tmp_path / "s.py"),
            patch.object(claw_mod, "_load_migration_module", return_value=fake_mod),
            patch.object(claw_mod, "get_config_path", return_value=config_path),
            patch.object(claw_mod, "prompt_yes_no") as mock_prompt,
        ):
            claw_mod._cmd_migrate(args)

        mock_prompt.assert_not_called()

    def test_handles_migration_error(self, tmp_path, capsys):
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        args = Namespace(
            source=str(openclaw_dir),
            dry_run=True, preset="full", overwrite=False,
            migrate_secrets=False, workspace_target=None,
            skill_conflict="skip", yes=False,
        )

        with (
            patch.object(claw_mod, "_find_migration_script", return_value=tmp_path / "s.py"),
            patch.object(claw_mod, "_load_migration_module", side_effect=RuntimeError("boom")),
            patch.object(claw_mod, "get_config_path", return_value=config_path),
            patch.object(claw_mod, "save_config"),
            patch.object(claw_mod, "load_config", return_value={}),
        ):
            claw_mod._cmd_migrate(args)

        captured = capsys.readouterr()
        assert "Migration failed" in captured.out

    def test_full_preset_enables_secrets(self, tmp_path, capsys):
        """The 'full' preset should set migrate_secrets=True automatically."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()

        fake_mod = ModuleType("openclaw_to_hermes")
        fake_mod.resolve_selected_options = MagicMock(return_value=set())
        fake_migrator = MagicMock()
        fake_migrator.migrate.return_value = {
            "summary": {"migrated": 0, "skipped": 0, "conflict": 0, "error": 0},
            "items": [],
        }
        fake_mod.Migrator = MagicMock(return_value=fake_migrator)

        args = Namespace(
            source=str(openclaw_dir),
            dry_run=True, preset="full", overwrite=False,
            migrate_secrets=False,  # Not explicitly set by user
            workspace_target=None,
            skill_conflict="skip", yes=False,
        )

        with (
            patch.object(claw_mod, "_find_migration_script", return_value=tmp_path / "s.py"),
            patch.object(claw_mod, "_load_migration_module", return_value=fake_mod),
            patch.object(claw_mod, "get_config_path", return_value=tmp_path / "config.yaml"),
            patch.object(claw_mod, "save_config"),
            patch.object(claw_mod, "load_config", return_value={}),
        ):
            claw_mod._cmd_migrate(args)

        # Migrator should have been called with migrate_secrets=True
        call_kwargs = fake_mod.Migrator.call_args[1]
        assert call_kwargs["migrate_secrets"] is True


# ---------------------------------------------------------------------------
# _print_migration_report
# ---------------------------------------------------------------------------


class TestPrintMigrationReport:
    """Test the report formatting function."""

    def test_dry_run_report(self, capsys):
        report = {
            "summary": {"migrated": 2, "skipped": 1, "conflict": 1, "error": 0},
            "items": [
                {"kind": "soul", "status": "migrated", "destination": "/home/user/.hermes/SOUL.md"},
                {"kind": "memory", "status": "migrated", "destination": "/home/user/.hermes/memories/MEMORY.md"},
                {"kind": "skills", "status": "conflict", "reason": "already exists"},
                {"kind": "tts-assets", "status": "skipped", "reason": "not found"},
            ],
            "preset": "full",
        }
        claw_mod._print_migration_report(report, dry_run=True)
        captured = capsys.readouterr()
        assert "Dry Run Results" in captured.out
        assert "Would migrate" in captured.out
        assert "2 would migrate" in captured.out
        assert "--dry-run" in captured.out

    def test_execute_report(self, capsys):
        report = {
            "summary": {"migrated": 3, "skipped": 0, "conflict": 0, "error": 0},
            "items": [
                {"kind": "soul", "status": "migrated", "destination": "/home/user/.hermes/SOUL.md"},
            ],
            "output_dir": "/home/user/.hermes/migration/openclaw/20250312T120000",
        }
        claw_mod._print_migration_report(report, dry_run=False)
        captured = capsys.readouterr()
        assert "Migration Results" in captured.out
        assert "Migrated" in captured.out
        assert "Full report saved to" in captured.out

    def test_empty_report(self, capsys):
        report = {
            "summary": {"migrated": 0, "skipped": 0, "conflict": 0, "error": 0},
            "items": [],
        }
        claw_mod._print_migration_report(report, dry_run=False)
        captured = capsys.readouterr()
        assert "Nothing to migrate" in captured.out
