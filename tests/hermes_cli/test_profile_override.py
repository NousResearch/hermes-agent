"""Regression tests for _apply_profile_override in hermes_cli.main.

Issue #19299: `hermes profile use` stopped switching the active profile after
v0.12.0 introduced an early-return when HERMES_HOME was already set. The early
return was too broad — it also suppressed the active_profile check when
HERMES_HOME pointed to the root directory rather than a specific profile dir.
"""

import os
import sys
from pathlib import Path

import pytest


def _run_override(monkeypatch, tmp_path, hermes_home_val, argv=None):
    """Set up env, call _apply_profile_override, return resulting HERMES_HOME."""
    if argv is None:
        argv = ["hermes", "chat"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("HERMES_HOME", hermes_home_val)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli.main import _apply_profile_override
    _apply_profile_override()
    return os.environ.get("HERMES_HOME")


class TestApplyProfileOverrideRootHome:
    """HERMES_HOME points to the root: active_profile must still be honoured."""

    def test_root_home_with_active_profile_switches(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        profile_dir = hermes_home / "profiles" / "coder"
        profile_dir.mkdir(parents=True)
        (hermes_home / "active_profile").write_text("coder\n")

        result = _run_override(monkeypatch, tmp_path, str(hermes_home))

        assert result == str(profile_dir)

    def test_root_home_no_active_profile_unchanged(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)

        result = _run_override(monkeypatch, tmp_path, str(hermes_home))

        assert result == str(hermes_home)

    def test_docker_root_with_active_profile_switches(self, monkeypatch, tmp_path):
        docker_root = tmp_path / "opt" / "data"
        docker_root.mkdir(parents=True)
        profile_dir = docker_root / "profiles" / "prod"
        profile_dir.mkdir(parents=True)
        (docker_root / "active_profile").write_text("prod\n")

        result = _run_override(monkeypatch, tmp_path, str(docker_root))

        assert result == str(profile_dir)


class TestApplyProfileOverrideProfileDir:
    """HERMES_HOME already points to a profile dir: early return, no change."""

    def test_profile_dir_trusts_env(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        profile_dir = hermes_home / "profiles" / "dev"
        profile_dir.mkdir(parents=True)
        # active_profile says something different — should be ignored
        (hermes_home / "active_profile").write_text("staging\n")

        result = _run_override(monkeypatch, tmp_path, str(profile_dir))

        assert result == str(profile_dir)

    def test_docker_profile_dir_trusts_env(self, monkeypatch, tmp_path):
        docker_root = tmp_path / "opt" / "data"
        docker_root.mkdir(parents=True)
        profile_dir = docker_root / "profiles" / "qa"
        profile_dir.mkdir(parents=True)
        (docker_root / "active_profile").write_text("prod\n")

        result = _run_override(monkeypatch, tmp_path, str(profile_dir))

        assert result == str(profile_dir)


class TestApplyProfileOverrideExplicitFlag:
    """Explicit --profile flag always wins, regardless of HERMES_HOME."""

    def test_explicit_flag_overrides_root_home(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        profile_dir = hermes_home / "profiles" / "staging"
        profile_dir.mkdir(parents=True)
        (hermes_home / "active_profile").write_text("coder\n")

        result = _run_override(
            monkeypatch, tmp_path, str(hermes_home),
            argv=["hermes", "--profile", "staging", "chat"],
        )

        assert result == str(profile_dir)
