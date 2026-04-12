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
# _find_openclaw_dirs
# ---------------------------------------------------------------------------


class TestFindOpenclawDirs:
    """Test discovery of OpenClaw directories."""

    def test_finds_openclaw_dir(self, tmp_path):
        openclaw = tmp_path / ".openclaw"
        openclaw.mkdir()
        with patch("pathlib.Path.home", return_value=tmp_path):
            found = claw_mod._find_openclaw_dirs()
        assert openclaw in found

    def test_finds_legacy_dirs(self, tmp_path):
        clawdbot = tmp_path / ".clawdbot"
        clawdbot.mkdir()
        moltbot = tmp_path / ".moltbot"
        moltbot.mkdir()
        with patch("pathlib.Path.home", return_value=tmp_path):
            found = claw_mod._find_openclaw_dirs()
        assert len(found) == 2
        assert clawdbot in found
        assert moltbot in found

    def test_returns_empty_when_none_exist(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            found = claw_mod._find_openclaw_dirs()
        assert found == []


# ---------------------------------------------------------------------------
# _scan_workspace_state
# ---------------------------------------------------------------------------


class TestScanWorkspaceState:
    """Test scanning for workspace state files."""

    def test_finds_root_state_files(self, tmp_path):
        (tmp_path / "todo.json").write_text("{}")
        (tmp_path / "sessions").mkdir()
        findings = claw_mod._scan_workspace_state(tmp_path)
        descs = [desc for _, desc in findings]
        assert any("todo.json" in d for d in descs)
        assert any("sessions" in d for d in descs)

    def test_finds_workspace_state_files(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "todo.json").write_text("{}")
        (ws / "sessions").mkdir()
        findings = claw_mod._scan_workspace_state(tmp_path)
        descs = [desc for _, desc in findings]
        assert any("workspace/todo.json" in d for d in descs)
        assert any("workspace/sessions" in d for d in descs)

    def test_ignores_hidden_dirs(self, tmp_path):
        scan_dir = tmp_path / "scan_target"
        scan_dir.mkdir()
        hidden = scan_dir / ".git"
        hidden.mkdir()
        (hidden / "todo.json").write_text("{}")
        findings = claw_mod._scan_workspace_state(scan_dir)
        assert len(findings) == 0

    def test_empty_dir_returns_empty(self, tmp_path):
        scan_dir = tmp_path / "scan_target"
        scan_dir.mkdir()
        findings = claw_mod._scan_workspace_state(scan_dir)
        assert findings == []


# ---------------------------------------------------------------------------
# _archive_directory
# ---------------------------------------------------------------------------


class TestArchiveDirectory:
    """Test directory archival (rename)."""

    def test_renames_to_pre_migration(self, tmp_path):
        source = tmp_path / ".openclaw"
        source.mkdir()
        (source / "test.txt").write_text("data")

        archive_path = claw_mod._archive_directory(source)
        assert archive_path == tmp_path / ".openclaw.pre-migration"
        assert archive_path.is_dir()
        assert not source.exists()
        assert (archive_path / "test.txt").read_text() == "data"

    def test_adds_timestamp_when_archive_exists(self, tmp_path):
        source = tmp_path / ".openclaw"
        source.mkdir()
        # Pre-existing archive
        (tmp_path / ".openclaw.pre-migration").mkdir()

        archive_path = claw_mod._archive_directory(source)
        assert ".pre-migration-" in archive_path.name
        assert archive_path.is_dir()
        assert not source.exists()

    def test_dry_run_does_not_rename(self, tmp_path):
        source = tmp_path / ".openclaw"
        source.mkdir()

        archive_path = claw_mod._archive_directory(source, dry_run=True)
        assert archive_path == tmp_path / ".openclaw.pre-migration"
        assert source.is_dir()  # Still exists


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

    def test_routes_to_cleanup(self):
        args = Namespace(claw_action="cleanup", source=None, dry_run=False, yes=False)
        with patch.object(claw_mod, "_cmd_cleanup") as mock:
            claw_mod.claw_command(args)
        mock.assert_called_once_with(args)

    def test_routes_clean_alias(self):
        args = Namespace(claw_action="clean", source=None, dry_run=False, yes=False)
        with patch.object(claw_mod, "_cmd_cleanup") as mock:
            claw_mod.claw_command(args)
        mock.assert_called_once_with(args)

    def test_shows_help_for_no_action(self, capsys):
        args = Namespace(claw_action=None)
        claw_mod.claw_command(args)
        captured = capsys.readouterr()
        assert "migrate" in captured.out
        assert "cleanup" in captured.out


# ---------------------------------------------------------------------------
# _cmd_migrate
# ---------------------------------------------------------------------------


class TestCmdMigrate:
    """Test the migrate command handler."""

    @pytest.fixture(autouse=True)
    def _mock_openclaw_running(self):
        with patch.object(claw_mod, "_is_openclaw_running", return_value=False):
            yield

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

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True

        with (
            patch.object(claw_mod, "_find_migration_script", return_value=tmp_path / "s.py"),
            patch.object(claw_mod, "_load_migration_module", return_value=fake_mod),
            patch.object(claw_mod, "get_config_path", return_value=config_path),
            patch.object(claw_mod, "prompt_yes_no", return_value=True),
            patch("sys.stdin", mock_stdin),
        ):
            claw_mod._cmd_migrate(args)

        captured = capsys.readouterr()
        assert "Migration Results" in captured.out
        assert "Migration complete!" in captured.out

    def test_dry_run_does_not_touch_source(self, tmp_path, capsys):
        """Dry run should not modify the source directory."""
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()

        fake_mod = ModuleType("openclaw_to_hermes")
        fake_mod.resolve_selected_options = MagicMock(return_value=set())
        fake_migrator = MagicMock()
        fake_migrator.migrate.return_value = {
            "summary": {"migrated": 2, "skipped": 0, "conflict": 0, "error": 0},
            "items": [],
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
            patch.object(claw_mod, "_find_migration_script", return_value=tmp_path / "s.py"),
            patch.object(claw_mod, "_load_migration_module", return_value=fake_mod),
            patch.object(claw_mod, "get_config_path", return_value=tmp_path / "config.yaml"),
            patch.object(claw_mod, "save_config"),
            patch.object(claw_mod, "load_config", return_value={}),
        ):
            claw_mod._cmd_migrate(args)

        assert openclaw_dir.is_dir()  # Source untouched

    def test_execute_cancelled_by_user(self, tmp_path, capsys):
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        # Preview must succeed before the confirmation prompt is shown
        fake_mod = ModuleType("openclaw_to_hermes")
        fake_mod.resolve_selected_options = MagicMock(return_value=set())
        fake_migrator = MagicMock()
        fake_migrator.migrate.return_value = {
            "summary": {"migrated": 1, "skipped": 0, "conflict": 0, "error": 0},
            "items": [{"kind": "soul", "status": "migrated", "source": "s", "destination": "d", "reason": ""}],
        }
        fake_mod.Migrator = MagicMock(return_value=fake_migrator)

        args = Namespace(
            source=str(openclaw_dir),
            dry_run=False, preset="full", overwrite=False,
            migrate_secrets=False, workspace_target=None,
            skill_conflict="skip", yes=False,
        )

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True

        with (
            patch.object(claw_mod, "_find_migration_script", return_value=tmp_path / "s.py"),
            patch.object(claw_mod, "_load_migration_module", return_value=fake_mod),
            patch.object(claw_mod, "get_config_path", return_value=config_path),
            patch.object(claw_mod, "prompt_yes_no", return_value=False),
            patch("sys.stdin", mock_stdin),
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
        assert "Could not load migration script" in captured.out

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
# _cmd_cleanup
# ---------------------------------------------------------------------------


class TestCmdCleanup:
    """Test the cleanup command handler."""

    def test_no_dirs_found(self, tmp_path, capsys):
        args = Namespace(source=None, dry_run=False, yes=False)
        with patch.object(claw_mod, "_find_openclaw_dirs", return_value=[]):
            claw_mod._cmd_cleanup(args)
        captured = capsys.readouterr()
        assert "No OpenClaw directories found" in captured.out

    def test_dry_run_lists_dirs(self, tmp_path, capsys):
        openclaw = tmp_path / ".openclaw"
        openclaw.mkdir()
        ws = openclaw / "workspace"
        ws.mkdir()
        (ws / "todo.json").write_text("{}")

        args = Namespace(source=None, dry_run=True, yes=False)
        with patch.object(claw_mod, "_find_openclaw_dirs", return_value=[openclaw]):
            claw_mod._cmd_cleanup(args)

        captured = capsys.readouterr()
        assert "Would archive" in captured.out
        assert openclaw.is_dir()  # Not actually archived

    def test_archives_with_yes(self, tmp_path, capsys):
        openclaw = tmp_path / ".openclaw"
        openclaw.mkdir()
        (openclaw / "workspace").mkdir()
        (openclaw / "workspace" / "todo.json").write_text("{}")

        args = Namespace(source=None, dry_run=False, yes=True)
        with patch.object(claw_mod, "_find_openclaw_dirs", return_value=[openclaw]):
            claw_mod._cmd_cleanup(args)

        captured = capsys.readouterr()
        assert "Archived" in captured.out
        assert "Cleaned up 1" in captured.out
        assert not openclaw.exists()
        assert (tmp_path / ".openclaw.pre-migration").is_dir()

    def test_skips_when_user_declines(self, tmp_path, capsys):
        openclaw = tmp_path / ".openclaw"
        openclaw.mkdir()

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True

        args = Namespace(source=None, dry_run=False, yes=False)
        with (
            patch.object(claw_mod, "_find_openclaw_dirs", return_value=[openclaw]),
            patch.object(claw_mod, "prompt_yes_no", return_value=False),
            patch("sys.stdin", mock_stdin),
        ):
            claw_mod._cmd_cleanup(args)

        captured = capsys.readouterr()
        assert "Skipped" in captured.out
        assert openclaw.is_dir()

    def test_explicit_source(self, tmp_path, capsys):
        custom_dir = tmp_path / "my-openclaw"
        custom_dir.mkdir()
        (custom_dir / "todo.json").write_text("{}")

        args = Namespace(source=str(custom_dir), dry_run=False, yes=True)
        claw_mod._cmd_cleanup(args)

        captured = capsys.readouterr()
        assert "Archived" in captured.out
        assert not custom_dir.exists()

    def test_shows_workspace_details(self, tmp_path, capsys):
        openclaw = tmp_path / ".openclaw"
        openclaw.mkdir()
        ws = openclaw / "workspace"
        ws.mkdir()
        (ws / "todo.json").write_text("{}")
        (ws / "SOUL.md").write_text("# Soul")

        args = Namespace(source=None, dry_run=True, yes=False)
        with patch.object(claw_mod, "_find_openclaw_dirs", return_value=[openclaw]):
            claw_mod._cmd_cleanup(args)

        captured = capsys.readouterr()
        assert "workspace/" in captured.out
        assert "todo.json" in captured.out

    def test_handles_multiple_dirs(self, tmp_path, capsys):
        openclaw = tmp_path / ".openclaw"
        openclaw.mkdir()
        clawdbot = tmp_path / ".clawdbot"
        clawdbot.mkdir()

        args = Namespace(source=None, dry_run=False, yes=True)
        with patch.object(claw_mod, "_find_openclaw_dirs", return_value=[openclaw, clawdbot]):
            claw_mod._cmd_cleanup(args)

        captured = capsys.readouterr()
        assert "Cleaned up 2" in captured.out
        assert not openclaw.exists()
        assert not clawdbot.exists()


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


class TestIsOpenclawRunning:
    def test_returns_true_when_pgrep_finds_openclaw(self):
        with patch.object(claw_mod, "sys") as mock_sys:
            mock_sys.platform = "darwin"
            with patch.object(claw_mod, "subprocess") as mock_subprocess:
                mock_subprocess.run.side_effect = [
                    MagicMock(returncode=0),
                ]
                assert claw_mod._is_openclaw_running() is True

    def test_returns_true_when_pgrep_finds_clawd(self):
        with patch.object(claw_mod, "sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch.object(claw_mod, "subprocess") as mock_subprocess:
                mock_subprocess.run.side_effect = [
                    MagicMock(returncode=1),
                    MagicMock(returncode=0),
                ]
                assert claw_mod._is_openclaw_running() is True

    def test_returns_false_when_pgrep_finds_nothing(self):
        with patch.object(claw_mod, "sys") as mock_sys:
            mock_sys.platform = "darwin"
            with patch.object(claw_mod, "subprocess") as mock_subprocess:
                mock_subprocess.run.side_effect = [
                    MagicMock(returncode=1),
                    MagicMock(returncode=1),
                ]
                assert claw_mod._is_openclaw_running() is False

    def test_returns_true_on_windows_tasklist(self):
        with patch.object(claw_mod, "sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch.object(claw_mod, "subprocess") as mock_subprocess:
                mock_subprocess.run.return_value = MagicMock(
                    returncode=0,
                    stdout="node.exe openclaw-gateway",
                )
                assert claw_mod._is_openclaw_running() is True

    def test_returns_false_on_windows_when_not_found(self):
        with patch.object(claw_mod, "sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch.object(claw_mod, "subprocess") as mock_subprocess:
                mock_subprocess.run.return_value = MagicMock(
                    returncode=0,
                    stdout="node.exe some-other-app",
                )
                assert claw_mod._is_openclaw_running() is False


class TestWarnIfOpenclawRunning:
    def test_noop_when_not_running(self, capsys):
        with patch.object(claw_mod, "_is_openclaw_running", return_value=False):
            claw_mod._warn_if_openclaw_running(auto_yes=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_warns_and_exits_when_running_and_user_declines(self, capsys):
        with patch.object(claw_mod, "_is_openclaw_running", return_value=True):
            with patch.object(claw_mod, "prompt_yes_no", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    claw_mod._warn_if_openclaw_running(auto_yes=False)
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "OpenClaw appears to be running" in captured.out

    def test_warns_and_continues_when_running_and_user_accepts(self, capsys):
        with patch.object(claw_mod, "_is_openclaw_running", return_value=True):
            with patch.object(claw_mod, "prompt_yes_no", return_value=True):
                claw_mod._warn_if_openclaw_running(auto_yes=False)
        captured = capsys.readouterr()
        assert "OpenClaw appears to be running" in captured.out

    def test_warns_and_continues_in_auto_yes_mode(self, capsys):
        with patch.object(claw_mod, "_is_openclaw_running", return_value=True):
            claw_mod._warn_if_openclaw_running(auto_yes=True)
        captured = capsys.readouterr()
        assert "OpenClaw appears to be running" in captured.out
