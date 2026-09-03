"""Tests for hermes claw commands."""

from argparse import Namespace
import subprocess
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


    def test_ignores_hidden_dirs(self, tmp_path):
        scan_dir = tmp_path / "scan_target"
        scan_dir.mkdir()
        hidden = scan_dir / ".git"
        hidden.mkdir()
        (hidden / "todo.json").write_text("{}")
        findings = claw_mod._scan_workspace_state(scan_dir)
        assert len(findings) == 0


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
        with patch.object(claw_mod, "_detect_openclaw_processes", return_value=[]):
            yield








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

    def test_full_preset_does_not_enable_secrets_silently(self, tmp_path, capsys):
        """The 'full' preset must NOT auto-enable migrate_secrets.

        Users have to opt in to secret import explicitly via --migrate-secrets,
        even under the 'full' preset.  This mirrors OpenClaw's migrate-hermes
        posture (two-phase import) and prevents a 'full' run from silently
        copying API keys.
        """
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
            no_backup=False,
        )

        with (
            patch.object(claw_mod, "_find_migration_script", return_value=tmp_path / "s.py"),
            patch.object(claw_mod, "_load_migration_module", return_value=fake_mod),
            patch.object(claw_mod, "get_config_path", return_value=tmp_path / "config.yaml"),
            patch.object(claw_mod, "save_config"),
            patch.object(claw_mod, "load_config", return_value={}),
        ):
            claw_mod._cmd_migrate(args)

        # Migrator should have been called with migrate_secrets=False — the
        # 'full' preset on its own no longer opts the user into secret import.
        call_kwargs = fake_mod.Migrator.call_args[1]
        assert call_kwargs["migrate_secrets"] is False

    def test_full_preset_with_explicit_migrate_secrets_passes_through(self, tmp_path, capsys):
        """Explicit --migrate-secrets still works under --preset full."""
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
            migrate_secrets=True,  # Explicitly requested
            workspace_target=None,
            skill_conflict="skip", yes=False,
            no_backup=False,
        )

        with (
            patch.object(claw_mod, "_find_migration_script", return_value=tmp_path / "s.py"),
            patch.object(claw_mod, "_load_migration_module", return_value=fake_mod),
            patch.object(claw_mod, "get_config_path", return_value=tmp_path / "config.yaml"),
            patch.object(claw_mod, "save_config"),
            patch.object(claw_mod, "load_config", return_value={}),
        ):
            claw_mod._cmd_migrate(args)

        call_kwargs = fake_mod.Migrator.call_args[1]
        assert call_kwargs["migrate_secrets"] is True


# ---------------------------------------------------------------------------
# _cmd_cleanup
# ---------------------------------------------------------------------------


class TestCmdCleanup:
    """Test the cleanup command handler."""

    @pytest.fixture(autouse=True)
    def _mock_openclaw_running(self):
        with patch.object(claw_mod, "_detect_openclaw_processes", return_value=[]):
            yield


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


    def test_explicit_source(self, tmp_path, capsys):
        custom_dir = tmp_path / "my-openclaw"
        custom_dir.mkdir()
        (custom_dir / "todo.json").write_text("{}")

        args = Namespace(source=str(custom_dir), dry_run=False, yes=True)
        claw_mod._cmd_cleanup(args)

        captured = capsys.readouterr()
        assert "Archived" in captured.out
        assert not custom_dir.exists()




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


    def test_empty_report(self, capsys):
        report = {
            "summary": {"migrated": 0, "skipped": 0, "conflict": 0, "error": 0},
            "items": [],
        }
        claw_mod._print_migration_report(report, dry_run=False)
        captured = capsys.readouterr()
        assert "Nothing to migrate" in captured.out


class TestDetectOpenclawProcesses:
    def test_returns_match_when_pgrep_finds_openclaw(self):
        """Validated pgrep hits are reported (POSIX scan helper directly,
        so the assertion holds regardless of the host OS)."""

        def fake_run(cmd, **kwargs):
            if cmd[0] == "pgrep":
                return MagicMock(returncode=0, stdout="1234\n")
            if cmd[0] == "ps":
                return MagicMock(returncode=0, stdout="/usr/local/bin/openclaw gateway\n")
            return MagicMock(returncode=1, stdout="")

        with patch.object(claw_mod.subprocess, "run", side_effect=fake_run), patch.object(
            claw_mod.os, "getpid", return_value=999999
        ):
            result = claw_mod._posix_pgrep_openclaw_hits()
        assert len(result) == 1
        assert "1234" in result[0]


    @pytest.mark.windows_only
    def test_returns_empty_on_windows_when_nothing_found(self):
        """Faking win32 picked the tasklist/powershell branch on a host that has
        neither; only a real Windows host resolves those executables.

        ``return_value`` rather than a ``side_effect`` list: the branch's call
        count is not the assertion, and pinning it breaks whenever the host
        shells out once more than the dev box did.
        """
        with patch.object(claw_mod, "subprocess") as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(returncode=0, stdout="")
            result = claw_mod._detect_openclaw_processes()
            assert result == []


class TestWarnIfOpenclawRunning:
    def test_noop_when_not_running(self, capsys):
        with patch.object(claw_mod, "_detect_openclaw_processes", return_value=[]):
            claw_mod._warn_if_openclaw_running(auto_yes=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_warns_and_exits_when_running_and_user_declines(self, capsys):
        with patch.object(claw_mod, "_detect_openclaw_processes", return_value=["openclaw process(es) (PIDs: 1234)"]):
            with patch.object(claw_mod, "prompt_yes_no", return_value=False):
                with patch.object(claw_mod.sys.stdin, "isatty", return_value=True):
                    with pytest.raises(SystemExit) as exc_info:
                        claw_mod._warn_if_openclaw_running(auto_yes=False)
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "OpenClaw appears to be running" in captured.out




class TestLooksLikeOpenclawCommand:
    def test_real_executable_basenames(self):
        assert claw_mod._looks_like_openclaw_command("openclaw gateway --port 1") is True
        assert claw_mod._looks_like_openclaw_command("/usr/local/bin/openclaw") is True
        assert claw_mod._looks_like_openclaw_command("openclaw-gateway serve") is True
        assert claw_mod._looks_like_openclaw_command("clawd --daemon") is True

    def test_node_hosted_installs_detected(self):
        assert (
            claw_mod._looks_like_openclaw_command(
                "node /home/u/.openclaw/gateway.js"
            )
            is True
        )
        assert (
            claw_mod._looks_like_openclaw_command(
                "node /srv/app/node_modules/openclaw/dist/index.js"
            )
            is True
        )
        assert (
            claw_mod._looks_like_openclaw_command(
                "node server.js --config /home/u/.clawdbot/config.yaml"
            )
            is True
        )

    def test_incidental_mentions_rejected(self):
        # An editor with an openclaw-ish path, a grep over docs, and this
        # migration's own skill path are NOT running gateways.
        assert (
            claw_mod._looks_like_openclaw_command(
                "vim /home/u/openclaw-notes/a.md"
            )
            is False
        )
        assert claw_mod._looks_like_openclaw_command("grep openclaw notes.txt") is False
        assert (
            claw_mod._looks_like_openclaw_command("hermes claw cleanup") is False
        )
        assert (
            claw_mod._looks_like_openclaw_command("python skills/openclaw-migration/openclaw_to_hermes.py")
            is False
        )
        assert claw_mod._looks_like_openclaw_command("") is False


class TestPgrepHitsValidated:
    """pgrep -f hits must be filtered to validated OpenClaw commands."""

    def test_unrelated_cmdline_not_reported(self):
        pgrep_result = MagicMock(returncode=0, stdout="111 222\n")

        def fake_run(cmd, **kwargs):
            if cmd[0] == "pgrep":
                return pgrep_result
            if cmd[0] == "ps":
                pid = cmd[cmd.index("-p") + 1]
                lines = {
                    "111": "vim /home/u/openclaw-notes/x.md\n",
                    "222": "/usr/local/bin/openclaw gateway\n",
                }
                return MagicMock(returncode=0, stdout=lines.get(pid, ""))
            return MagicMock(returncode=1, stdout="")

        with patch.object(claw_mod.subprocess, "run", side_effect=fake_run), patch.object(
            claw_mod.os, "getpid", return_value=999999
        ):
            found = claw_mod._posix_pgrep_openclaw_hits()
        joined = "\n".join(found)
        assert "111" not in joined, "unrelated process flagged as OpenClaw"
        assert any("222" in f for f in found), "real openclaw process lost"

    def test_self_pid_excluded(self):
        me = claw_mod.os.getpid()
        pgrep_result = MagicMock(returncode=0, stdout=f"{me}\n")

        def fake_run(cmd, **kwargs):
            if cmd[0] == "pgrep":
                return pgrep_result
            if cmd[0] == "ps":
                return MagicMock(returncode=0, stdout="openclaw serve\n")
            return MagicMock(returncode=1, stdout="")

        with patch.object(claw_mod.subprocess, "run", side_effect=fake_run):
            found = claw_mod._posix_pgrep_openclaw_hits()
        assert found == []

    def test_all_hits_invalid_means_no_detection(self):
        pgrep_result = MagicMock(returncode=0, stdout="300 301 302\n")

        def fake_run(cmd, **kwargs):
            if cmd[0] == "pgrep":
                return pgrep_result
            if cmd[0] == "ps":
                return MagicMock(returncode=0, stdout="tail -f /var/log/syslog\n")
            return MagicMock(returncode=1, stdout="")

        with patch.object(claw_mod.subprocess, "run", side_effect=fake_run), patch.object(
            claw_mod.os, "getpid", return_value=1
        ):
            assert claw_mod._posix_pgrep_openclaw_hits() == []
