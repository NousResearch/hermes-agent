"""Tests for hermes config backup — git-based config auto-save feature."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, call

import pytest

from hermes_cli.config import (
    backup_command,
    BACKUP_TRACKED,
    BACKUP_GITIGNORE,
    BACKUP_CRON_MARKER,
    BACKUP_CRON_LINE,
    get_hermes_home,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(backup_command_name, **kwargs):
    """Build a minimal args namespace for backup_command()."""
    class Args:
        pass
    a = Args()
    a.backup_command = backup_command_name
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


def _seed_hermes_home(home: Path):
    """Create minimal tracked files so git add has something to commit."""
    (home / "config.yaml").write_text("model:\n  default: test-model\n")
    (home / "SOUL.md").write_text("# Soul\n")
    for d in ("skills", "cron", "memories"):
        (home / d).mkdir(exist_ok=True)


def _git(args, cwd):
    return subprocess.run(
        ["git"] + args, cwd=cwd, capture_output=True, text=True
    )


# ---------------------------------------------------------------------------
# TestBackupInit
# ---------------------------------------------------------------------------

class TestBackupInit:

    def test_init_creates_git_repo(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        # Ensure git identity for CI environments
        _git(["config", "--global", "user.email", "test@test.com"], cwd=tmp_path)
        _git(["config", "--global", "user.name", "Test"], cwd=tmp_path)

        backup_command(_make_args("init", remote=None))

        assert (tmp_path / ".git").is_dir()

    def test_init_writes_gitignore(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)

        backup_command(_make_args("init", remote=None))

        gi = tmp_path / ".gitignore"
        assert gi.exists()
        content = gi.read_text()
        assert ".env" in content
        assert "auth.json" in content
        assert "logs/" in content
        assert "state.db" in content

    def test_init_does_not_overwrite_existing_gitignore(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        custom = "# my custom gitignore\n"
        (tmp_path / ".gitignore").write_text(custom)

        backup_command(_make_args("init", remote=None))

        assert (tmp_path / ".gitignore").read_text() == custom

    def test_init_creates_initial_commit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)

        backup_command(_make_args("init", remote=None))

        result = _git(["log", "--oneline"], cwd=tmp_path)
        assert result.returncode == 0
        assert "initial hermes config snapshot" in result.stdout

    def test_init_only_commits_gitignore_when_no_tracked_files_exist(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # No config.yaml / SOUL.md seeded — only .gitignore will be staged

        backup_command(_make_args("init", remote=None))

        result = _git(["log", "--oneline"], cwd=tmp_path)
        # .gitignore itself gets committed (it's in BACKUP_TRACKED)
        assert result.returncode == 0
        assert "initial hermes config snapshot" in result.stdout

    def test_init_adds_remote(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        remote_url = "https://github.com/YanSte/hermes-config"

        backup_command(_make_args("init", remote=remote_url))

        result = _git(["remote", "get-url", "origin"], cwd=tmp_path)
        assert result.returncode == 0
        assert result.stdout.strip() == remote_url

    def test_init_updates_existing_remote(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        _git(["init"], cwd=tmp_path)
        _git(["remote", "add", "origin", "https://old-url.com"], cwd=tmp_path)

        new_url = "https://github.com/YanSte/hermes-config-new"
        backup_command(_make_args("init", remote=new_url))

        result = _git(["remote", "get-url", "origin"], cwd=tmp_path)
        assert result.stdout.strip() == new_url

    def test_init_idempotent_when_repo_already_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)

        backup_command(_make_args("init", remote=None))
        backup_command(_make_args("init", remote=None))  # second call

        assert (tmp_path / ".git").is_dir()
        result = _git(["log", "--oneline"], cwd=tmp_path)
        # Only one initial commit
        assert result.stdout.count("\n") <= 1

    def test_init_env_file_excluded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-secret\n")

        backup_command(_make_args("init", remote=None))

        result = _git(["show", "HEAD:.env"], cwd=tmp_path)
        assert result.returncode != 0  # .env must NOT be in commit


# ---------------------------------------------------------------------------
# TestBackupPush
# ---------------------------------------------------------------------------

class TestBackupPush:

    def _init(self, home):
        _seed_hermes_home(home)
        backup_command(_make_args("init", remote=None))

    def test_push_commits_changed_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._init(tmp_path)

        # Modify a tracked file
        (tmp_path / "config.yaml").write_text("model:\n  default: changed-model\n")

        backup_command(_make_args("push"))

        result = _git(["log", "--oneline"], cwd=tmp_path)
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 2
        assert "auto: config snapshot" in lines[0]

    def test_push_noop_when_nothing_changed(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._init(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            backup_command(_make_args("push"))

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "nothing to commit" in captured.out.lower()

    def test_push_without_init_prints_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with pytest.raises(SystemExit):
            backup_command(_make_args("push"))

        captured = capsys.readouterr()
        assert "init" in captured.out.lower()

    def test_push_skips_remote_when_none_configured(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._init(tmp_path)
        (tmp_path / "config.yaml").write_text("model:\n  default: new\n")

        git_calls = []
        original_run = subprocess.run

        def fake_run(cmd, **kwargs):
            if cmd[0] == "git" and "push" in cmd:
                git_calls.append(cmd)
            return original_run(cmd, **kwargs)

        with patch("hermes_cli.config.subprocess.run", side_effect=fake_run):
            backup_command(_make_args("push"))

        assert not any("push" in c for c in git_calls)

    def test_push_with_remote_calls_git_push(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._init(tmp_path)
        # Add a fake remote
        _git(["remote", "add", "origin", "https://github.com/YanSte/hermes-config"], cwd=tmp_path)
        (tmp_path / "config.yaml").write_text("model:\n  default: pushed\n")

        push_calls = []
        original_git = __import__("hermes_cli.config", fromlist=["_git"])._git

        def fake_git(args, cwd):
            if "push" in args:
                push_calls.append(args)
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return original_git(args, cwd)

        with patch("hermes_cli.config._git", side_effect=fake_git):
            backup_command(_make_args("push"))

        assert any("push" in c for c in push_calls)

    def test_push_warn_on_remote_failure(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._init(tmp_path)
        _git(["remote", "add", "origin", "https://github.com/YanSte/hermes-config"], cwd=tmp_path)
        (tmp_path / "config.yaml").write_text("model:\n  default: fail-push\n")

        original_git = __import__("hermes_cli.config", fromlist=["_git"])._git

        def fake_git(args, cwd):
            if "push" in args:
                return SimpleNamespace(returncode=1, stdout="", stderr="Authentication failed")
            return original_git(args, cwd)

        with patch("hermes_cli.config._git", side_effect=fake_git):
            backup_command(_make_args("push"))

        captured = capsys.readouterr()
        assert "push failed" in captured.out.lower() or "authentication" in captured.out.lower()

    def test_push_excludes_env_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._init(tmp_path)
        (tmp_path / ".env").write_text("SECRET=should-not-be-committed\n")
        (tmp_path / "config.yaml").write_text("model:\n  default: updated\n")

        backup_command(_make_args("push"))

        result = _git(["show", "HEAD:.env"], cwd=tmp_path)
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# TestBackupPull
# ---------------------------------------------------------------------------

class TestBackupPull:

    def test_pull_without_repo_prints_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with pytest.raises(SystemExit):
            backup_command(_make_args("pull"))

        captured = capsys.readouterr()
        assert "init" in captured.out.lower()

    def test_pull_without_remote_prints_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        backup_command(_make_args("init", remote=None))

        with pytest.raises(SystemExit):
            backup_command(_make_args("pull"))

        captured = capsys.readouterr()
        assert "remote" in captured.out.lower()

    def test_pull_calls_git_pull(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        backup_command(_make_args("init", remote=None))
        _git(["remote", "add", "origin", "https://github.com/YanSte/hermes-config"], cwd=tmp_path)

        pull_calls = []
        original_git = __import__("hermes_cli.config", fromlist=["_git"])._git

        def fake_git(args, cwd):
            if "pull" in args:
                pull_calls.append(args)
                return SimpleNamespace(returncode=0, stdout="Already up to date.", stderr="")
            return original_git(args, cwd)

        with patch("hermes_cli.config._git", side_effect=fake_git):
            backup_command(_make_args("pull"))

        assert any("pull" in c for c in pull_calls)


# ---------------------------------------------------------------------------
# TestBackupStatus
# ---------------------------------------------------------------------------

class TestBackupStatus:

    def test_status_without_repo(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        backup_command(_make_args("status"))

        captured = capsys.readouterr()
        assert "init" in captured.out.lower()

    def test_status_shows_last_commit(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        backup_command(_make_args("init", remote=None))

        backup_command(_make_args("status"))

        captured = capsys.readouterr()
        assert "initial hermes config snapshot" in captured.out

    def test_status_shows_no_remote_when_local_only(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        backup_command(_make_args("init", remote=None))

        backup_command(_make_args("status"))

        captured = capsys.readouterr()
        assert "local only" in captured.out.lower() or "none" in captured.out.lower()

    def test_status_shows_remote_url(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        remote_url = "https://github.com/YanSte/hermes-config"
        backup_command(_make_args("init", remote=remote_url))

        backup_command(_make_args("status"))

        captured = capsys.readouterr()
        assert remote_url in captured.out

    def test_status_shows_dirty_files(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        backup_command(_make_args("init", remote=None))
        (tmp_path / "config.yaml").write_text("model:\n  default: dirty\n")

        backup_command(_make_args("status"))

        captured = capsys.readouterr()
        assert "config.yaml" in captured.out

    def test_status_shows_auto_on(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_hermes_home(tmp_path)
        backup_command(_make_args("init", remote=None))

        fake_cron = f"{BACKUP_CRON_LINE}  {BACKUP_CRON_MARKER}\n"

        original_run = subprocess.run

        def fake_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd == ["crontab", "-l"]:
                return SimpleNamespace(returncode=0, stdout=fake_cron, stderr="")
            return original_run(cmd, **kwargs)

        with patch("hermes_cli.config.subprocess.run", side_effect=fake_run):
            backup_command(_make_args("status"))

        captured = capsys.readouterr()
        assert "on" in captured.out.lower()


# ---------------------------------------------------------------------------
# TestBackupAuto
# ---------------------------------------------------------------------------

class TestBackupAuto:

    def _run_auto(self, state, existing_cron=""):
        cron_written = []

        def fake_run(cmd, **kwargs):
            if cmd == ["crontab", "-l"]:
                return SimpleNamespace(returncode=0, stdout=existing_cron, stderr="")
            if cmd[0] == "crontab" and cmd[1] == "-":
                cron_written.append(kwargs.get("input", ""))
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return subprocess.run(cmd, **kwargs)

        with patch("hermes_cli.config.subprocess.run", side_effect=fake_run):
            backup_command(_make_args("auto", state=state))

        return cron_written

    def test_auto_on_adds_cron_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        written = self._run_auto("on")
        assert written
        assert BACKUP_CRON_MARKER in written[0]
        assert "hermes config backup push" in written[0]

    def test_auto_on_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        existing = f"{BACKUP_CRON_LINE}  {BACKUP_CRON_MARKER}\n"
        written = self._run_auto("on", existing_cron=existing)
        # Should appear exactly once
        assert written[0].count(BACKUP_CRON_MARKER) == 1

    def test_auto_off_removes_cron_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        existing = f"0 5 * * * /other/job\n{BACKUP_CRON_LINE}  {BACKUP_CRON_MARKER}\n"
        written = self._run_auto("off", existing_cron=existing)
        assert BACKUP_CRON_MARKER not in written[0]
        assert "/other/job" in written[0]  # other entries preserved

    def test_auto_off_noop_when_not_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        existing = "0 5 * * * /other/job\n"
        written = self._run_auto("off", existing_cron=existing)
        assert BACKUP_CRON_MARKER not in written[0]
        assert "/other/job" in written[0]

    def test_auto_on_prints_confirmation(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._run_auto("on")
        captured = capsys.readouterr()
        assert "enabled" in captured.out.lower()

    def test_auto_off_prints_confirmation(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._run_auto("off")
        captured = capsys.readouterr()
        assert "disabled" in captured.out.lower()

    def test_auto_on_empty_existing_crontab(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        cron_written = []

        def fake_run(cmd, **kwargs):
            if cmd == ["crontab", "-l"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="no crontab for user")
            if cmd[0] == "crontab" and cmd[1] == "-":
                cron_written.append(kwargs.get("input", ""))
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return subprocess.run(cmd, **kwargs)

        with patch("hermes_cli.config.subprocess.run", side_effect=fake_run):
            backup_command(_make_args("auto", state="on"))

        assert cron_written
        assert BACKUP_CRON_MARKER in cron_written[0]


# ---------------------------------------------------------------------------
# TestBackupConstants
# ---------------------------------------------------------------------------

class TestBackupConstants:

    def test_env_in_gitignore_template(self):
        assert ".env" in BACKUP_GITIGNORE

    def test_auth_json_in_gitignore_template(self):
        assert "auth.json" in BACKUP_GITIGNORE

    def test_state_db_in_gitignore_template(self):
        assert "state.db" in BACKUP_GITIGNORE

    def test_logs_in_gitignore_template(self):
        assert "logs/" in BACKUP_GITIGNORE

    def test_tracked_includes_config_yaml(self):
        assert "config.yaml" in BACKUP_TRACKED

    def test_tracked_includes_soul_md(self):
        assert "SOUL.md" in BACKUP_TRACKED

    def test_tracked_includes_skills(self):
        assert "skills" in BACKUP_TRACKED

    def test_tracked_includes_memories(self):
        assert "memories" in BACKUP_TRACKED

    def test_cron_line_has_correct_command(self):
        assert "hermes config backup push" in BACKUP_CRON_LINE

    def test_cron_line_has_hourly_schedule(self):
        assert BACKUP_CRON_LINE.startswith("0 * * * *")
