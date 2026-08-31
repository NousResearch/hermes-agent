"""Tests for config migration on the commit_count == 0 / retry path (#91360).

When an update is interrupted (or dependency install fails) after code has already
been pulled, the subsequent update run takes the 'commit_count == 0' path.
It must run config version checking and migrations before completing so the
install is not left in a non-bootable state with new code on old config version.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hermes_cli import update_cmd


def test_repair_node_deps_runs_config_migration_on_version_bump(capsys):
    """When on-disk config version is behind, _repair_node_deps_on_current_checkout
    must run _check_and_apply_config_migration and migrate the config."""
    completion = MagicMock()
    with (
        patch.object(update_cmd, "_update_node_dependencies", return_value=[]),
        patch.object(update_cmd, "_m") as m,
        patch.object(update_cmd, "_reload_config_modules"),
        patch.object(update_cmd, "_run_config_check_fresh", return_value=(37, 38)),
        patch("hermes_cli.config.get_missing_env_vars", return_value=[]),
        patch("hermes_cli.config.get_missing_config_fields", return_value=[]),
        patch.object(
            update_cmd,
            "_run_migrate_config_fresh",
            return_value={"env_added": [], "config_added": ["migrated to v38"], "warnings": []},
        ) as mock_migrate,
    ):
        update_cmd._repair_node_deps_on_current_checkout(completion)

    m.return_value._build_web_ui.assert_called_once()
    mock_migrate.assert_called_once_with(interactive=False, quiet=True)
    completion.assert_called_once_with("✓ Already up to date!")
    out = capsys.readouterr().out
    assert "Checking configuration for new options..." in out
    assert "Updating config format (v37 → v38)…" in out
    assert "Config format updated" in out


def test_repair_node_deps_up_to_date_config(capsys):
    """When config is already up to date, it reports up to date without error."""
    completion = MagicMock()
    with (
        patch.object(update_cmd, "_update_node_dependencies", return_value=[]),
        patch.object(update_cmd, "_m") as m,
        patch.object(update_cmd, "_reload_config_modules"),
        patch.object(update_cmd, "_run_config_check_fresh", return_value=(38, 38)),
        patch("hermes_cli.config.get_missing_env_vars", return_value=[]),
        patch("hermes_cli.config.get_missing_config_fields", return_value=[]),
        patch.object(update_cmd, "_run_migrate_config_fresh") as mock_migrate,
    ):
        update_cmd._repair_node_deps_on_current_checkout(completion)

    m.return_value._build_web_ui.assert_called_once()
    mock_migrate.assert_not_called()
    completion.assert_called_once_with("✓ Already up to date!")
    out = capsys.readouterr().out
    assert "Checking configuration for new options..." in out
    assert "Configuration is up to date" in out


def test_check_and_apply_config_migration_interactive_prompt():
    """When new config options exist in an interactive session, it prompts the user."""
    with (
        patch.object(update_cmd, "_reload_config_modules"),
        patch.object(update_cmd, "_run_config_check_fresh", return_value=(37, 38)),
        patch("hermes_cli.config.get_missing_env_vars", return_value=[{"name": "NEW_KEY", "description": "desc"}]),
        patch("hermes_cli.config.get_missing_config_fields", return_value=[]),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value="y"),
        patch.object(
            update_cmd,
            "_run_migrate_config_fresh",
            return_value={"env_added": ["NEW_KEY"], "config_added": [], "warnings": []},
        ) as mock_migrate,
    ):
        update_cmd._check_and_apply_config_migration(assume_yes=False, gateway_mode=False)

    mock_migrate.assert_called_once_with(interactive=True, quiet=False)


def test_check_and_apply_config_migration_assume_yes():
    """When assume_yes=True, it applies migrations non-interactively without prompting."""
    with (
        patch.object(update_cmd, "_reload_config_modules"),
        patch.object(update_cmd, "_run_config_check_fresh", return_value=(37, 38)),
        patch("hermes_cli.config.get_missing_env_vars", return_value=[{"name": "NEW_KEY"}]),
        patch("hermes_cli.config.get_missing_config_fields", return_value=[]),
        patch.object(
            update_cmd,
            "_run_migrate_config_fresh",
            return_value={"env_added": [], "config_added": ["opt"], "warnings": []},
        ) as mock_migrate,
    ):
        update_cmd._check_and_apply_config_migration(assume_yes=True, gateway_mode=False)

    mock_migrate.assert_called_once_with(interactive=False, quiet=False)


def test_check_and_apply_config_migration_non_interactive():
    """In a non-interactive session (e.g. CI/scripts), it applies safe migrations automatically."""
    with (
        patch.object(update_cmd, "_reload_config_modules"),
        patch.object(update_cmd, "_run_config_check_fresh", return_value=(37, 38)),
        patch("hermes_cli.config.get_missing_env_vars", return_value=[]),
        patch("hermes_cli.config.get_missing_config_fields", return_value=[{"key": "new_setting"}]),
        patch("sys.stdin.isatty", return_value=False),
        patch("sys.stdout.isatty", return_value=False),
        patch.object(
            update_cmd,
            "_run_migrate_config_fresh",
            return_value={"env_added": [], "config_added": ["new_setting"], "warnings": []},
        ) as mock_migrate,
    ):
        update_cmd._check_and_apply_config_migration(assume_yes=False, gateway_mode=False)

    mock_migrate.assert_called_once_with(interactive=False, quiet=False)
