"""Focused coverage for `hermes update --no-gateway-restart`.

A cron running inside the gateway's own cgroup cannot survive the fleet
restart phase (SIGUSR1 drain + systemd KillMode=mixed kills the updater
itself). The flag runs the full update pipeline but defers the restart;
the pending-restart marker is kept so a later normal update catches up.
"""
import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.subcommands.update import build_update_parser
from hermes_cli import update_cmd as uc
from hermes_cli import update_cmd_fleet as fleet


def _h(name):
    def handler(args):  # pragma: no cover - identity only
        return name
    handler.__name__ = f"cmd_{name}"
    return handler


def _parse(argv):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_update_parser(sub, cmd_update=_h("update"))
    return parser.parse_args(argv)


def _opts(**overrides):
    base = dict(
        assume_yes=True, gw_input_fn=None, active_lazy_features=[],
        active_tool_dependencies=[], pre_update_version="1.0",
        discard_local_changes=False, keep_stash=False, switch_branch=False,
        no_gateway_restart=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── flag parsing ──────────────────────────────────────────────────────


def test_parser_defaults_to_restart():
    ns = _parse(["update"])
    assert ns.no_gateway_restart is False


def test_parser_accepts_no_gateway_restart():
    ns = _parse(["update", "--no-gateway-restart"])
    assert ns.no_gateway_restart is True


def test_resolve_options_threads_flag():
    with (
        patch.object(uc, "_m") as mock_m,
        patch.object(uc, "_read_project_version", return_value="1.0"),
        patch.object(uc, "_updates_config", return_value={}),
    ):
        mock_m.return_value._capture_active_lazy_features.return_value = []
        mock_m.return_value._capture_active_tool_dependencies.return_value = []
        assert uc._resolve_update_options(
            SimpleNamespace(no_gateway_restart=True), False).no_gateway_restart is True
        assert uc._resolve_update_options(
            SimpleNamespace(), False).no_gateway_restart is False


# ── pending catch-up deferral ─────────────────────────────────────────


def test_catchup_deferred_under_flag_when_pending():
    """Already-up-to-date + pending marker + flag: no restart, no exit, marker kept."""
    with (
        patch.object(fleet, "_pending_fleet_restart_needed", return_value=True),
        patch.object(fleet, "_warn_pending_fleet_restart"),
        patch.object(uc, "_run_pending_fleet_restart") as mock_run,
        patch.object(fleet, "_clear_fleet_restart_pending_marker") as mock_clear,
    ):
        fleet._apply_pending_fleet_restart_catchup(
            respect_no_gateway_restart=True, no_gateway_restart=True)
    mock_run.assert_not_called()
    mock_clear.assert_not_called()


def test_catchup_default_path_unchanged():
    """Without the flag the pending restart still runs and clears the marker."""
    with (
        patch.object(fleet, "_pending_fleet_restart_needed", return_value=True),
        patch.object(fleet, "_warn_pending_fleet_restart"),
        patch.object(uc, "_run_pending_fleet_restart", return_value=True) as mock_run,
        patch.object(fleet, "_clear_fleet_restart_pending_marker") as mock_clear,
    ):
        fleet._apply_pending_fleet_restart_catchup()
    mock_run.assert_called_once()
    mock_clear.assert_called_once()


def test_catchup_noop_when_nothing_pending():
    with (
        patch.object(fleet, "_pending_fleet_restart_needed", return_value=False),
        patch.object(uc, "_run_pending_fleet_restart") as mock_run,
    ):
        fleet._apply_pending_fleet_restart_catchup(
            respect_no_gateway_restart=True, no_gateway_restart=True)
    mock_run.assert_not_called()


# ── defer helper ──────────────────────────────────────────────────────


def test_defer_records_skip_and_finalizes_success():
    with (
        patch("hermes_cli.update_receipt.record_skip") as mock_skip,
        patch("hermes_cli.update_receipt.finalize_update_receipt") as mock_final,
    ):
        fleet._defer_fleet_restart_after_update()
    mock_skip.assert_called_once()
    assert mock_skip.call_args[0][0] == "gateway_restart"
    mock_final.assert_called_once_with("success")


# ── pulled-update orchestration ───────────────────────────────────────


def test_pulled_update_skips_restart_phase_under_flag():
    with (
        patch.object(uc, "_invalidate_update_cache"),
        patch.object(uc, "_verify_head_after_pull", return_value="newsha"),
        patch.object(uc, "_write_fleet_restart_pending_marker") as mock_marker,
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
            _windows_gateway_resume=None,
        )
    mock_restart.assert_not_called()
    mock_verify.assert_not_called()
    mock_defer.assert_called_once()
    mock_resume.assert_called_once()  # Windows pause/resume still runs
    mock_marker.assert_called_once()  # pending marker kept for catch-up


def test_pulled_update_default_path_unchanged():
    with (
        patch.object(uc, "_invalidate_update_cache"),
        patch.object(uc, "_verify_head_after_pull", return_value="newsha"),
        patch.object(uc, "_write_fleet_restart_pending_marker"),
        patch.object(uc, "_sweep_bytecode_after_update"),
        patch.object(uc, "_sync_python_dependencies_after_pull"),
        patch.object(uc, "_update_node_dependencies", return_value=[]),
        patch.object(uc, "_rebuild_desktop_after_update", return_value=True),
        patch.object(uc, "_run_post_update_maintenance", return_value=True),
        patch.object(uc, "_branch_head_suffix", return_value=""),
        patch.object(uc, "_m", return_value=MagicMock()),
        patch.object(uc, "_restart_gateway_fleet_after_update") as mock_restart,
        patch.object(uc, "_resume_windows_gateways_and_merge_outcome"),
        patch.object(uc, "_verify_fleet_after_update") as mock_verify,
        patch.object(uc, "_defer_fleet_restart_after_update") as mock_defer,
    ):
        uc._apply_pulled_update(
            "git", "main", "oldsha", SimpleNamespace(in_place_update=False),
            _opts(no_gateway_restart=False), gateway_mode=False,
            is_fork=False, desktop_dir="/tmp", had_desktop_app_before_update=False,
            pre_update_snapshot_id=None, _pre_update_plan=None,
            _windows_gateway_resume=None,
        )
    mock_restart.assert_called_once()
    mock_verify.assert_called_once()
    mock_defer.assert_not_called()
