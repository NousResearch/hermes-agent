"""The commit_count == 0 path must repair Node deps, not just Python (#77211).

A previous ``hermes update`` whose npm install failed printed "Fix npm and
re-run `hermes update`" — but re-running hit the "Already up to date!" early
return before the Node refresh, so the advice could never work. The repair
now runs through ``_repair_node_deps_on_current_checkout``, which delegates
to ``_update_node_dependencies`` (self-gating on the lockfile hash, recorded
only after a successful install, so healthy installs stay a cheap no-op).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hermes_cli import update_cmd


def test_current_checkout_repairs_failed_node_deps(capsys):
    """A recorded failure surfaces the fix-npm hint, not 'Already up to date!'."""
    completion = MagicMock()
    with patch.object(
        update_cmd, "_update_node_dependencies", return_value=["ui-tui, web workspaces"]
    ), patch.object(update_cmd, "_m") as m:
        update_cmd._repair_node_deps_on_current_checkout(completion)

    m.return_value._build_web_ui.assert_not_called()
    completion.assert_called_once()
    assert "could not be repaired" in completion.call_args[0][0]
    out = capsys.readouterr().out
    assert "Node.js refresh failed for: ui-tui, web workspaces" in out
    assert "Fix npm and re-run `hermes update`." in out


def test_current_checkout_healthy_node_deps_reports_up_to_date():
    """A clean refresh (or lockfile-hash no-op) still says 'Already up to date!'."""
    completion = MagicMock()
    with patch.object(
        update_cmd, "_update_node_dependencies", return_value=[]
    ), patch.object(update_cmd, "_m") as m:
        update_cmd._repair_node_deps_on_current_checkout(completion)

    # The refresh pairs with the web build like every other call site.
    m.return_value._build_web_ui.assert_called_once()
    completion.assert_called_once_with("✓ Already up to date!")


def test_current_checkout_migrates_config_from_already_pulled_code(capsys):
    """An interrupted-install retry must not leave old config unbootable."""
    with (
        patch.object(update_cmd, "_run_config_check_fresh", return_value=(37, 38)),
        patch.object(
            update_cmd,
            "_run_migrate_config_fresh",
            return_value={"config_added": [], "warnings": []},
        ) as migrate,
    ):
        update_cmd._repair_config_on_current_checkout()

    migrate.assert_called_once_with(interactive=False, quiet=True)
    assert "Updating config format (v37 → v38)" in capsys.readouterr().out


def test_current_checkout_skips_config_migration_when_current(capsys):
    with (
        patch.object(update_cmd, "_run_config_check_fresh", return_value=(38, 38)),
        patch.object(update_cmd, "_run_migrate_config_fresh") as migrate,
    ):
        update_cmd._repair_config_on_current_checkout()

    migrate.assert_not_called()
    assert "Configuration is up to date" in capsys.readouterr().out


def test_current_checkout_config_migration_failure_is_actionable(capsys):
    with patch.object(
        update_cmd, "_run_config_check_fresh", side_effect=RuntimeError("broken")
    ):
        update_cmd._repair_config_on_current_checkout()

    out = capsys.readouterr().out
    assert "Config format update failed: broken" in out
    assert "hermes config migrate" in out
