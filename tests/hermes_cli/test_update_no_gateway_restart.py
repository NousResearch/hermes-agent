"""`hermes update --no-gateway-restart` (#93649).

A cron running inside the gateway's own cgroup cannot survive the fleet
restart phase (SIGUSR1 drain + systemd KillMode=mixed kills the updater
itself). The flag runs the full update pipeline but defers the restart;
the pending-restart marker is kept so a later normal update catches up.
"""
from types import SimpleNamespace
from unittest.mock import patch

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


def test_defer_surfaces_externally_supervised_gateways(capsys):
    """--no-gateway-restart + a live externally-supervised gateway must not read as green (#118643).

    The deferred marker never restarts them: their supervisor (launchd/systemd) keeps the old
    process alive against the post-update checkout, so the session later dies on an ImportError
    after new symbols land. Surface the explicit post-update step instead.
    """
    with (
        patch.object(fleet, "_externally_supervised_live_gateways",
                     return_value=[(4321, "python -m hermes_cli.main gateway run --external-supervisor")]),
        patch("hermes_cli.update_receipt.record_skip") as mock_skip,
        patch("hermes_cli.update_receipt.finalize_update_receipt"),
    ):
        fleet._defer_fleet_restart_after_update(update_complete=True)

    out = capsys.readouterr().out
    assert "EXTERNALLY SUPERVISED" in out  # RED pre-fix: no such warning
    assert "pid 4321" in out
    reasons = [str(call) for call in mock_skip.call_args_list]
    assert any("external_supervisor_gateway" in reason for reason in reasons)  # RED


def test_defer_without_supervised_gateways_stays_quiet(capsys):
    with (
        patch.object(fleet, "_externally_supervised_live_gateways", return_value=[]),
        patch("hermes_cli.update_receipt.record_skip") as mock_skip,
        patch("hermes_cli.update_receipt.finalize_update_receipt"),
    ):
        fleet._defer_fleet_restart_after_update(update_complete=True)

    out = capsys.readouterr().out
    assert "EXTERNALLY SUPERVISED" not in out
    # The pre-existing skip entry is the only one recorded.
    assert mock_skip.call_count == 1

