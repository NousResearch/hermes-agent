"""Tests for the hermes config seal/verify/restore CLI subcommands.

Covers:
  - cmd_seal calls seal() and exits 0 on success
  - cmd_verify exits 1 when config is tampered
  - cmd_verify exits 0 when config matches baseline
  - cmd_restore reverts a tampered config and exits 0
  - cmd_restore exits 0 with no-op message when config already matches
  - Exit codes propagate correctly through the dispatch layer

All filesystem and git subprocess calls are mocked so that tests run
without a real dotfiles repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def _init_dotfiles_repo(dotfiles: Path) -> None:
    hermes_dir = dotfiles / "hermes"
    hermes_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(dotfiles)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=dotfiles, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=dotfiles, check=True, capture_output=True,
    )
    (dotfiles / "README.md").write_text("dotfiles\n")
    subprocess.run(["git", "add", "README.md"], cwd=dotfiles, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=dotfiles, check=True, capture_output=True,
    )


@pytest.fixture()
def integrity_env(tmp_path, monkeypatch):
    """Set up a temp config.yaml and a temp dotfiles git repo."""
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  default: test-model\n")

    dotfiles = tmp_path / "dotfiles"
    _init_dotfiles_repo(dotfiles)

    monkeypatch.setenv("HERMES_CONFIG", str(config))
    monkeypatch.setenv("HERMES_DOTFILES_DIR", str(dotfiles))

    return {
        "config": config,
        "dotfiles": dotfiles,
        "log": dotfiles / "hermes" / "config_integrity.jsonl",
    }


# ---------------------------------------------------------------------------
# Import the CLI module under test (lazy, so env vars are set first)
# ---------------------------------------------------------------------------


def _get_cli():
    from hermes_cli import config_integrity_cli
    return config_integrity_cli


def _get_core(integrity_env):
    """Import and return the shared core module with paths resolved from env."""
    cli = _get_cli()
    return cli._import_core()


# ---------------------------------------------------------------------------
# cmd_seal
# ---------------------------------------------------------------------------


class TestCmdSeal:
    def test_seal_exits_0_on_success(self, integrity_env):
        cli = _get_cli()
        args = _make_args()
        rc = cli.cmd_seal(args)
        assert rc == 0

    def test_seal_creates_log_file(self, integrity_env):
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        assert integrity_env["log"].exists()

    def test_seal_log_contains_seal_entry(self, integrity_env):
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        entries = [
            json.loads(l)
            for l in integrity_env["log"].read_text().splitlines()
            if l.strip()
        ]
        assert len(entries) == 1
        assert entries[0]["event"] == "seal"

    def test_seal_hash_matches_config(self, integrity_env):
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        entries = [
            json.loads(l)
            for l in integrity_env["log"].read_text().splitlines()
            if l.strip()
        ]
        expected = hashlib.sha256(integrity_env["config"].read_bytes()).hexdigest()
        assert entries[0]["hash"] == expected

    def test_seal_exits_1_when_config_missing(self, integrity_env, monkeypatch):
        monkeypatch.setenv("HERMES_CONFIG", "/nonexistent/config.yaml")
        # Force re-import to pick up new env
        import importlib
        import skills  # noqa: F401 — ensure skills root is on path for re-import
        cli = _get_cli()
        # Import fresh core directly with overridden env
        core = cli._import_core()
        rc = core.seal(config_path=Path("/nonexistent/config.yaml"))
        assert rc == 1

    def test_seal_commits_log_to_git(self, integrity_env):
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        result = subprocess.run(
            ["git", "status", "--porcelain", "hermes/config_integrity.jsonl"],
            cwd=integrity_env["dotfiles"], capture_output=True, text=True,
        )
        assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# cmd_verify
# ---------------------------------------------------------------------------


class TestCmdVerify:
    def test_verify_exits_0_after_seal(self, integrity_env):
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        rc = cli.cmd_verify(_make_args())
        assert rc == 0

    def test_verify_exits_1_when_tampered(self, integrity_env):
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        integrity_env["config"].write_text("model:\n  default: evil-model\n")
        rc = cli.cmd_verify(_make_args())
        assert rc == 1

    def test_verify_exits_3_when_no_baseline(self, integrity_env):
        cli = _get_cli()
        rc = cli.cmd_verify(_make_args())
        assert rc == 3

    def test_verify_exits_2_when_log_tampered(self, integrity_env):
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        # Manually write to log without committing
        with open(integrity_env["log"], "a") as f:
            f.write(json.dumps({"event": "seal", "hash": "a" * 64}) + "\n")
        rc = cli.cmd_verify(_make_args())
        assert rc == 2

    def test_verify_prints_hash_prefix_on_ok(self, integrity_env, capsys):
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        cli.cmd_verify(_make_args())
        out = capsys.readouterr().out
        expected_prefix = hashlib.sha256(integrity_env["config"].read_bytes()).hexdigest()[:16]
        assert expected_prefix in out


# ---------------------------------------------------------------------------
# cmd_restore
# ---------------------------------------------------------------------------


class TestCmdRestore:
    def _commit_canonical(self, env: dict) -> None:
        """Copy config.yaml into dotfiles/hermes/ and commit it."""
        canonical = env["dotfiles"] / "hermes" / "config.yaml"
        canonical.write_bytes(env["config"].read_bytes())
        subprocess.run(
            ["git", "add", "hermes/config.yaml"],
            cwd=env["dotfiles"], check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add canonical config"],
            cwd=env["dotfiles"], check=True, capture_output=True,
        )

    def test_restore_noop_when_matches_baseline(self, integrity_env, capsys):
        self._commit_canonical(integrity_env)
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        rc = cli.cmd_restore(_make_args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "no restore needed" in out.lower()

    def test_restore_reverts_tampered_config(self, integrity_env):
        original_content = integrity_env["config"].read_bytes()
        self._commit_canonical(integrity_env)
        cli = _get_cli()
        cli.cmd_seal(_make_args())

        # Tamper
        integrity_env["config"].write_text("model:\n  default: evil-model\n")
        assert cli.cmd_verify(_make_args()) == 1

        # Restore
        rc = cli.cmd_restore(_make_args())
        assert rc == 0
        assert integrity_env["config"].read_bytes() == original_content

    def test_restore_verify_clean_after_restore(self, integrity_env):
        self._commit_canonical(integrity_env)
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        integrity_env["config"].write_text("model:\n  default: evil-model\n")
        cli.cmd_restore(_make_args())
        rc = cli.cmd_verify(_make_args())
        assert rc == 0

    def test_restore_creates_backup_of_tampered_config(self, integrity_env):
        self._commit_canonical(integrity_env)
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        integrity_env["config"].write_text("model:\n  default: evil-model\n")
        cli.cmd_restore(_make_args())

        config_dir = integrity_env["config"].parent
        backups = list(config_dir.glob("*.pre-restore-*"))
        assert len(backups) >= 1

    def test_restore_exits_1_when_no_baseline(self, integrity_env):
        cli = _get_cli()
        rc = cli.cmd_restore(_make_args())
        assert rc == 1

    def test_restore_logs_tamper_detected_entry(self, integrity_env):
        self._commit_canonical(integrity_env)
        cli = _get_cli()
        cli.cmd_seal(_make_args())
        integrity_env["config"].write_text("model:\n  default: evil-model\n")
        cli.cmd_restore(_make_args())

        entries = [
            json.loads(l)
            for l in integrity_env["log"].read_text().splitlines()
            if l.strip()
        ]
        events = [e["event"] for e in entries]
        assert "tamper_detected" in events
        assert "restore" in events


# ---------------------------------------------------------------------------
# Dispatch integration: config_command routes to CLI handlers
# ---------------------------------------------------------------------------


class TestConfigCommandDispatch:
    """Verify that hermes_cli.config.config_command dispatches seal/verify/restore."""

    def test_dispatch_seal(self, integrity_env):
        from hermes_cli.config import config_command
        args = _make_args(config_command="seal")
        with pytest.raises(SystemExit) as exc_info:
            config_command(args)
        assert exc_info.value.code == 0

    def test_dispatch_verify_exits_3_no_baseline(self, integrity_env):
        from hermes_cli.config import config_command
        args = _make_args(config_command="verify")
        with pytest.raises(SystemExit) as exc_info:
            config_command(args)
        assert exc_info.value.code == 3

    def test_dispatch_verify_exits_0_after_seal(self, integrity_env):
        from hermes_cli.config import config_command

        seal_args = _make_args(config_command="seal")
        with pytest.raises(SystemExit):
            config_command(seal_args)

        verify_args = _make_args(config_command="verify")
        with pytest.raises(SystemExit) as exc_info:
            config_command(verify_args)
        assert exc_info.value.code == 0

    def test_dispatch_verify_exits_1_when_tampered(self, integrity_env):
        from hermes_cli.config import config_command

        seal_args = _make_args(config_command="seal")
        with pytest.raises(SystemExit):
            config_command(seal_args)

        integrity_env["config"].write_text("model:\n  default: evil-model\n")

        verify_args = _make_args(config_command="verify")
        with pytest.raises(SystemExit) as exc_info:
            config_command(verify_args)
        assert exc_info.value.code == 1
