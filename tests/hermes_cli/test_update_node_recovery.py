"""Residual Node recovery guarantees for an already-current checkout."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_cli import update_cmd


def _module_mock(*health: bool) -> MagicMock:
    module = MagicMock()
    module.return_value.PROJECT_ROOT = Path("/tmp/hermes-agent")
    module.return_value._npm_lockfile_changed.side_effect = health
    return module


def test_current_checkout_reports_repair_and_refreshes_dashboard(capsys):
    """A repaired Node tree is rebuilt, activated, and reported as repaired."""
    completion = MagicMock()
    module = _module_mock(True, False)

    with (
        patch.object(update_cmd, "_m", module),
        patch.object(update_cmd, "_update_node_dependencies", return_value=[]) as repair,
        patch.object(update_cmd, "_finish_dashboard_update_cleanup") as cleanup,
    ):
        update_cmd._repair_node_deps_on_current_checkout(completion)

    repair.assert_called_once_with()
    module.return_value._build_web_ui.assert_called_once_with(
        module.return_value.PROJECT_ROOT / "web"
    )
    cleanup.assert_called_once_with([])
    completion.assert_called_once_with("✓ Update complete!")
    assert "Node.js dependencies repaired" in capsys.readouterr().out


def test_current_checkout_does_not_claim_success_while_health_stays_incomplete(capsys):
    """A no-error npm result is not success while dependency health stays stale."""
    completion = MagicMock()
    module = _module_mock(True, True)

    with (
        patch.object(update_cmd, "_m", module),
        patch.object(update_cmd, "_update_node_dependencies", return_value=[]),
        patch.object(update_cmd, "_finish_dashboard_update_cleanup") as cleanup,
    ):
        update_cmd._repair_node_deps_on_current_checkout(completion)

    module.return_value._build_web_ui.assert_not_called()
    cleanup.assert_called_once_with(["dependency health check"])
    assert "could not be repaired" in completion.call_args.args[0]
    output = capsys.readouterr().out
    assert "remain incomplete" in output
    assert "Already up to date" not in output


def test_current_checkout_failed_repair_leaves_dashboard_untouched(capsys):
    """An npm failure remains visible and cannot restart a dashboard on stale deps."""
    completion = MagicMock()
    module = _module_mock(True)
    failures = ["ui-tui, web workspaces"]

    with (
        patch.object(update_cmd, "_m", module),
        patch.object(update_cmd, "_update_node_dependencies", return_value=failures),
        patch.object(update_cmd, "_finish_dashboard_update_cleanup") as cleanup,
    ):
        update_cmd._repair_node_deps_on_current_checkout(completion)

    module.return_value._build_web_ui.assert_not_called()
    cleanup.assert_called_once_with(failures)
    assert "could not be repaired" in completion.call_args.args[0]
    output = capsys.readouterr().out
    assert "Node.js refresh failed" in output
    assert "Already up to date" not in output
