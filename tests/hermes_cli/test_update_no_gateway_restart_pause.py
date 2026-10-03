"""#131149: --no-gateway-restart skips the Windows gateway pause; the pause refuses gateway-ancestor tree-kill."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import main, update_cmd, update_cmd_windows


def _drive_to_checkout(monkeypatch, tmp_path, **cmd_kwargs):
    """Drive main.cmd_update to checkout preparation; return the trail and the pause mock."""
    from hermes_cli import update_inventory

    class ReachedCheckout(BaseException):
        pass

    reached = []
    pause = Mock(return_value=None)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kwargs: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda token: None)
    monkeypatch.setattr(
        update_inventory, "collect_runtime_inventory", lambda: update_inventory.UpdatePlan()
    )
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda args: reached.append("backup"))
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", pause)
    monkeypatch.setattr(main, "_desktop_packaged_executable", lambda root: None)
    monkeypatch.setattr(main, "_desktop_dist_exists", lambda root: False)

    def prepare_checkout():
        reached.append("checkout")
        raise ReachedCheckout

    monkeypatch.setattr(update_cmd, "_prepare_git_command", prepare_checkout)
    args = dict(gateway=False, check=False, yes=True, force=False, force_venv=False)
    args.update(cmd_kwargs)
    with pytest.raises(ReachedCheckout):
        main.cmd_update(SimpleNamespace(**args))
    return reached, pause


def test_no_gateway_restart_skips_windows_pause(monkeypatch, tmp_path, capsys):
    """--no-gateway-restart must not stop the fleet it runs inside (cron in its own cgroup)."""
    import hermes_cli.update_receipt as update_receipt

    skips = []
    monkeypatch.setattr(
        update_receipt, "record_skip", lambda step, reason: skips.append((step, reason))
    )
    reached, pause = _drive_to_checkout(monkeypatch, tmp_path, no_gateway_restart=True)

    assert reached == ["backup", "checkout"]
    pause.assert_not_called()
    assert any(
        step == "windows_gateway_pause" and "--no-gateway-restart" in reason
        for step, reason in skips
    )
    assert "--no-gateway-restart" in capsys.readouterr().out


def test_default_update_still_pauses_windows_gateways(monkeypatch, tmp_path):
    """Without the flag the pause still runs (and learns whether it is gateway-parented)."""
    _reached, pause = _drive_to_checkout(monkeypatch, tmp_path)

    assert pause.call_count == 1
    assert pause.call_args.kwargs == {"gateway_mode": False}


def test_pause_refuses_gateway_ancestor_tree_kill(monkeypatch, capsys):
    """A gateway-parented plain update must refuse before stopping anything (#98814 on the pause path)."""
    import hermes_cli.gateway as gateway_cli
    import gateway.status as status_mod

    monkeypatch.setattr(main, "_is_windows", lambda: True)
    monkeypatch.setattr(
        update_cmd_windows, "_discover_windows_gateways", lambda: ({}, [], set(), [300])
    )
    monkeypatch.setattr(
        gateway_cli, "_is_pid_ancestor_of_current_process", lambda pid: pid == 300
    )
    terminate = Mock()
    monkeypatch.setattr(status_mod, "terminate_pid", terminate)

    with pytest.raises(SystemExit) as excinfo:
        update_cmd_windows._pause_windows_gateways_for_update(gateway_mode=False)

    assert excinfo.value.code == 2
    terminate.assert_not_called()
    assert "taskkill /T" in capsys.readouterr().out


def test_pause_gateway_mode_exempt_from_ancestor_refusal(monkeypatch):
    """The detached --gateway delivery owns gateway cleanup even when parented."""
    import hermes_cli.gateway as gateway_cli

    monkeypatch.setattr(main, "_is_windows", lambda: True)
    monkeypatch.setattr(
        update_cmd_windows, "_discover_windows_gateways", lambda: ({}, [], set(), [300])
    )
    monkeypatch.setattr(gateway_cli, "_is_pid_ancestor_of_current_process", lambda pid: True)
    import gateway.status as status_mod
    terminate = Mock()
    monkeypatch.setattr(status_mod, "terminate_pid", terminate)
    monkeypatch.setattr(status_mod, "get_process_start_time", lambda pid: 0.0)
    monkeypatch.setattr(update_cmd_windows, "_request_socket_pauses", lambda *a: ({}, [], []))
    monkeypatch.setattr(main, "_venv_launcher_ancestors", lambda pids: [])
    monkeypatch.setattr(main, "_wait_for_windows_update_gateway_exit", lambda pids, **k: set())
    monkeypatch.setattr(
        update_cmd_windows, "_record_attested_cold_start_profiles", lambda *a: None
    )
    monkeypatch.setattr(
        update_cmd_windows,
        "_pause_windows_gateway_services",
        lambda services, token, profiles, unmapped: token,
    )

    token = update_cmd_windows._pause_windows_gateways_for_update(gateway_mode=True)

    assert token["resume_needed"] is True
    assert terminate.call_count == 1
