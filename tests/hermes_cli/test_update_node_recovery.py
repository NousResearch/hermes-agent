"""Regression tests for Node dependency recovery on an already-current checkout."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import main as cli_main
from hermes_cli import update_cmd


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        backup=False,
        branch=None,
        force=False,
        force_venv=False,
        gateway=False,
        no_backup=True,
        yes=True,
    )


def _current_checkout_git(cmd, **_kwargs):
    joined = " ".join(str(part) for part in cmd)
    if "rev-parse --abbrev-ref HEAD" in joined:
        return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
    if "rev-list HEAD..origin/main --count" in joined:
        return subprocess.CompletedProcess(cmd, 0, stdout="0\n", stderr="")
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def test_current_checkout_repairs_incomplete_node_refresh(capsys):
    """No new commits must not bypass a known-incomplete Node dependency tree."""

    with (
        patch("subprocess.run", side_effect=_current_checkout_git),
        patch.object(cli_main, "_is_windows", return_value=False),
        patch.object(cli_main, "_run_pre_update_backup", return_value=None),
        patch.object(cli_main, "_pause_windows_gateways_for_update", return_value=None),
        patch.object(cli_main, "_resume_windows_gateways_after_update"),
        patch.object(update_cmd, "_discard_lockfile_churn"),
        patch.object(update_cmd, "_normalize_managed_eol"),
        patch.object(
            cli_main,
            "_get_origin_url",
            return_value="https://github.com/NousResearch/hermes-agent.git",
        ),
        patch.object(cli_main, "_stash_local_changes_if_needed", return_value=None),
        patch.object(update_cmd, "_invalidate_update_cache"),
        patch("hermes_cli.managed_uv.update_managed_uv"),
        patch("hermes_cli.managed_uv.ensure_uv", return_value=None),
        patch.object(update_cmd, "_venv_core_imports_healthy", return_value=(True, "")),
        patch.object(update_cmd, "_npm_lockfile_changed", return_value=True) as changed,
        patch.object(update_cmd, "_update_node_dependencies", return_value=[]) as repair,
        patch.object(cli_main, "_build_web_ui", return_value=True) as build,
        patch.object(update_cmd, "_finish_dashboard_update_cleanup") as cleanup,
    ):
        cli_main._cmd_update_impl(_args(), gateway_mode=False)

    changed.assert_called_once()
    repair.assert_called_once_with()
    build.assert_called_once_with(cli_main.PROJECT_ROOT / "web")
    cleanup.assert_called_once_with([])
    output = capsys.readouterr().out
    assert "Node.js dependencies repaired" in output


def test_current_checkout_surfaces_failed_node_repair(capsys):
    """A failed retry must not rebuild/restart or claim the install is healthy."""

    failures = ["repo root"]
    with (
        patch("subprocess.run", side_effect=_current_checkout_git),
        patch.object(cli_main, "_is_windows", return_value=False),
        patch.object(cli_main, "_run_pre_update_backup", return_value=None),
        patch.object(cli_main, "_pause_windows_gateways_for_update", return_value=None),
        patch.object(cli_main, "_resume_windows_gateways_after_update"),
        patch.object(update_cmd, "_discard_lockfile_churn"),
        patch.object(update_cmd, "_normalize_managed_eol"),
        patch.object(
            cli_main,
            "_get_origin_url",
            return_value="https://github.com/NousResearch/hermes-agent.git",
        ),
        patch.object(cli_main, "_stash_local_changes_if_needed", return_value=None),
        patch.object(update_cmd, "_invalidate_update_cache"),
        patch("hermes_cli.managed_uv.update_managed_uv"),
        patch("hermes_cli.managed_uv.ensure_uv", return_value=None),
        patch.object(update_cmd, "_venv_core_imports_healthy", return_value=(True, "")),
        patch.object(update_cmd, "_npm_lockfile_changed", return_value=True),
        patch.object(update_cmd, "_update_node_dependencies", return_value=failures),
        patch.object(cli_main, "_build_web_ui", return_value=True) as build,
        patch.object(update_cmd, "_finish_dashboard_update_cleanup") as cleanup,
    ):
        cli_main._cmd_update_impl(_args(), gateway_mode=False)

    build.assert_not_called()
    cleanup.assert_called_once_with(failures)
    output = capsys.readouterr().out
    assert "Node.js dependencies remain incomplete" in output
    assert "Already up to date!" not in output
