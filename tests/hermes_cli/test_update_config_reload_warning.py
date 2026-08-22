"""Tests for the post-update config-module reload failure warning (#90945).

``hermes update`` runs in the PRE-pull Python process, so after ``git pull``
the cached ``sys.modules`` still holds the OLD config code.  If
``_reload_config_modules()`` fails to reload a module, the version check runs
against the STALE module and reports the config as "up to date" — silently
skipping the pending migration (the exact failure in #90945).

The fix: reload failures are surfaced (warning log + a printed hint to run
``hermes doctor --fix``) instead of swallowed at debug level, so an update
never claims "Configuration is up to date" while a migration is pending.
"""

from unittest.mock import patch

import hermes_cli.update_cmd as update_cmd


def test_reload_config_modules_returns_failed_modules():
    """A failed reload must be collected and returned, not swallowed."""
    import types

    fake = types.ModuleType("hermes_cli.config")
    with patch.object(update_cmd.sys, "modules", {"hermes_cli.config": fake}):
        with patch("importlib.reload", side_effect=RuntimeError("boom")):
            failed = update_cmd._reload_config_modules()
    assert "hermes_cli.config" in failed


def test_reload_config_modules_success_returns_empty():
    """A clean reload returns an empty failure list."""
    # Use a real, already-imported module (importlib itself) so reload()
    # succeeds; patch reload to a no-op so no actual re-import happens.
    with patch("importlib.reload", side_effect=lambda mod: mod):
        failed = update_cmd._reload_config_modules()
    assert failed == []


def test_config_check_fresh_warns_on_reload_failure(capsys):
    """_run_config_check_fresh prints a doctor --fix hint when reload fails."""
    with patch.object(
        update_cmd, "_reload_config_modules", return_value=["hermes_cli.config"]
    ):
        with patch(
            "hermes_cli.config.check_config_version", return_value=(33, 37)
        ):
            current, latest = update_cmd._run_config_check_fresh()
    assert (current, latest) == (33, 37)
    out = capsys.readouterr().out
    assert "Could not reload updated config modules" in out
    assert "hermes doctor --fix" in out


def test_config_check_fresh_silent_on_success(capsys):
    """No warning printed when reload succeeds."""
    with patch.object(update_cmd, "_reload_config_modules", return_value=[]):
        with patch(
            "hermes_cli.config.check_config_version", return_value=(37, 37)
        ):
            update_cmd._run_config_check_fresh()
    out = capsys.readouterr().out
    assert "Could not reload" not in out
    assert "hermes doctor --fix" not in out


def test_migrate_config_fresh_skips_on_reload_failure(capsys):
    """_run_migrate_config_fresh refuses to migrate against stale modules."""
    with patch.object(
        update_cmd, "_reload_config_modules", return_value=["hermes_cli.config"]
    ):
        result = update_cmd._run_migrate_config_fresh()
    assert result == {"skipped": True, "reason": "config-module reload failed"}
    out = capsys.readouterr().out
    assert "Config migration skipped" in out
    assert "hermes doctor --fix" in out


def test_migrate_config_fresh_runs_on_success():
    """_run_migrate_config_fresh migrates normally when reload succeeds."""
    with patch.object(update_cmd, "_reload_config_modules", return_value=[]):
        with patch(
            "hermes_cli.config.migrate_config",
            return_value={"migrated": True},
        ):
            result = update_cmd._run_migrate_config_fresh()
    assert result == {"migrated": True}
