"""Tests for config migration on the \"Already up to date\" repair path.

Covers ``_check_and_apply_config_migration`` on the ``commit_count == 0``
retry paths (#91360): a previous update attempt can pull new code onto disk
and then fail before reaching the config-migration block (e.g. PyPI timeout
during dependency sync); the retry then enters the ``commit_count == 0``
branch and returns early, skipping config migration entirely. The fresh
code (which may require a newer ``_config_version``) keeps running against
the old config and the next Hermes launch refuses to start.

Matrix: config behind / current / ahead of the code's version, plus the
#86656 contract (quiet-migration warnings must be re-surfaced) and the
check-failure contract (a config-check failure must not break the repair
path).
"""

from __future__ import annotations

import contextlib
import io
from unittest.mock import patch

import hermes_cli.update_cmd as update_cmd


def _run(current: int, latest: int):
    """Run _check_and_apply_config_migration with mocked config checks.

    Returns (stdout, migrate_calls).
    """
    migrate_calls = []

    def _fake_migrate(interactive=False, quiet=False):
        migrate_calls.append((interactive, quiet))
        return {"env_added": [], "config_added": [], "warnings": []}

    with patch.object(update_cmd, "_reload_config_modules"), patch(
        "hermes_cli.config.get_missing_env_vars", return_value=[]
    ), patch(
        "hermes_cli.config.get_missing_config_fields", return_value=[]
    ), patch.object(
        update_cmd, "_run_config_check_fresh", return_value=(current, latest)
    ), patch.object(
        update_cmd, "_run_migrate_config_fresh", side_effect=_fake_migrate
    ), patch.object(
        update_cmd, "_migrate_sibling_profile_configs", return_value=[]
    ):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            update_cmd._check_and_apply_config_migration()
        return buf.getvalue(), migrate_calls


def test_migrates_when_config_behind():
    """Version bump on the repair path must be applied silently."""
    out, calls = _run(current=37, latest=38)
    assert "v37 → v38" in out
    assert "Config format updated" in out
    assert calls == [(False, True)]  # non-interactive, quiet


def test_noop_when_config_current():
    """No migration when the config version is already current."""
    out, calls = _run(current=38, latest=38)
    assert "Configuration is up to date" in out
    assert calls == []


def test_noop_when_config_ahead():
    """No migration when local config is newer than the code's default."""
    out, calls = _run(current=39, latest=38)
    assert "Configuration is up to date" in out
    assert calls == []


def test_surfaces_migration_warnings():
    """Warnings from a quiet migration must be re-surfaced (#86656)."""

    def _fake_migrate(interactive=False, quiet=False):
        return {
            "env_added": [],
            "config_added": [],
            "warnings": ["personality reset: kawaii → default"],
        }

    with patch.object(update_cmd, "_reload_config_modules"), patch(
        "hermes_cli.config.get_missing_env_vars", return_value=[]
    ), patch(
        "hermes_cli.config.get_missing_config_fields", return_value=[]
    ), patch.object(
        update_cmd, "_run_config_check_fresh", return_value=(37, 38)
    ), patch.object(
        update_cmd, "_run_migrate_config_fresh", side_effect=_fake_migrate
    ), patch.object(
        update_cmd, "_migrate_sibling_profile_configs", return_value=[]
    ):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            update_cmd._check_and_apply_config_migration()
        out = buf.getvalue()

    assert "personality reset" in out


def test_check_failure_does_not_break_repair_path():
    """A config-check failure must not break the repair path."""
    with patch.object(update_cmd, "_reload_config_modules"), patch(
        "hermes_cli.config.get_missing_env_vars", return_value=[]
    ), patch(
        "hermes_cli.config.get_missing_config_fields", return_value=[]
    ), patch.object(
        update_cmd, "_run_config_check_fresh", side_effect=RuntimeError("boom")
    ), patch.object(
        update_cmd, "_run_migrate_config_fresh", return_value={}
    ) as mig:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # Should not raise and should not attempt migration.
            update_cmd._check_and_apply_config_migration()
        out = buf.getvalue()
        assert "Could not check config version" in out
        mig.assert_not_called()
