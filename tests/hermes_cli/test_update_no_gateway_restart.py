"""`hermes update --no-gateway-restart` (#93649).

A cron running inside the gateway's own cgroup cannot survive the fleet
restart phase (SIGUSR1 drain + systemd KillMode=mixed kills the updater
itself). The flag runs the full update pipeline but defers the restart;
the pending-restart marker is kept so a later normal update catches up.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli import update_cmd as uc
from hermes_cli import update_cmd_fleet as fleet


def _opts(**overrides):
    base = dict(
        assume_yes=True, gw_input_fn=None, active_lazy_features=[],
        active_tool_dependencies=[], pre_update_version="1.0",
        discard_local_changes=False, keep_stash=False, switch_branch=False,
        no_gateway_restart=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_pulled_update_defers_restart_and_keeps_marker_under_flag():
    with (
        patch.object(uc, "_invalidate_update_cache"),
        patch.object(uc, "_verify_head_after_pull", return_value="newsha"),
        patch.object(uc, "_write_fleet_restart_pending_marker") as mock_marker,
        patch.object(uc, "_clear_fleet_restart_pending_marker") as mock_clear,
        patch.object(uc, "_sweep_bytecode_after_update"),
        patch.object(uc, "_sync_python_dependencies_after_pull"),
        patch.object(uc, "_update_node_dependencies", return_value=[]),
        patch.object(uc, "_rebuild_desktop_after_update", return_value=True),
        patch.object(uc, "_run_post_update_maintenance", return_value=True),
        patch.object(uc, "_branch_head_suffix", return_value=""),
        patch.object(uc, "_m", return_value=MagicMock()),
        patch.object(uc, "_restart_gateway_fleet_after_update") as mock_restart,
        patch.object(uc, "_resume_windows_gateways_and_merge_outcome") as mock_resume,
        patch.object(uc, "_verify_fleet_after_update") as mock_verify,
        patch.object(uc, "_defer_fleet_restart_after_update") as mock_defer,
    ):
        uc._apply_pulled_update(
            "git", "main", "oldsha", SimpleNamespace(in_place_update=False),
            _opts(no_gateway_restart=True), gateway_mode=False,
            is_fork=False, desktop_dir="/tmp", had_desktop_app_before_update=False,
            pre_update_snapshot_id=None, _pre_update_plan=None,
            _windows_gateway_resume=None, args=SimpleNamespace(no_gateway_restart=True, yes=True),
        )
    mock_restart.assert_not_called()
    mock_verify.assert_not_called()
    mock_defer.assert_called_once_with(update_complete=True, resume_incomplete=False)
    mock_resume.assert_called_once()  # Windows pause/resume still runs
    mock_marker.assert_called_once()  # pending marker kept for catch-up
    mock_clear.assert_not_called()  # deferred stale fleet alone is not partial


def test_already_current_catchup_is_deferred_under_flag():
    """Already-up-to-date + pending marker + flag: no restart, no exit, marker kept."""
    with (
        patch.object(fleet, "_pending_fleet_restart_needed", return_value=True),
        patch.object(fleet, "_warn_pending_fleet_restart"),
        patch.object(uc, "_run_pending_fleet_restart") as mock_run,
        patch.object(fleet, "_clear_fleet_restart_pending_marker") as mock_clear,
    ):
        fleet._apply_pending_fleet_restart_catchup(defer=True)
    mock_run.assert_not_called()
    mock_clear.assert_not_called()
