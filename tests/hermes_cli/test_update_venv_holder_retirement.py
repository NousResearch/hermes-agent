"""PM updates do not require the currently running venv to become unoccupied."""

import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import main, update_cmd, update_cmd_windows


@pytest.mark.real_concurrent_gate
@pytest.mark.parametrize(
    "module,name,args,kwargs",
    [
        (update_cmd, "_refuse_gateway_ancestor_tree_kill", ([123, 456],), {"gateway_mode": False}),
        (update_cmd, "_refuse_gateway_ancestor_tree_kill", ([123, 456],), {"gateway_mode": True}),
        (main, "_filter_non_gateway_concurrent_instances", ([(123, "hermes.exe")],), {}),
        (main, "_detect_concurrent_hermes_instances", (Path("Scripts"),), {"exclude_pid": 123}),
        (main, "_leftover_pausable_gateway_pids", ([(123, "python.exe", "hermes serve")],), {}),
        (main, "_ledger_manual_serve_holders", ([(123, "python.exe", "hermes serve")],), {}),
        (main, "_ledger_reapable_backend_pids", ([(123, "python.exe", "hermes serve")],), {}),
        (main, "_orphaned_desktop_backend_pids", ([(123, "python.exe", "hermes serve")],), {}),
        (main, "_handoff_reapable_backend_pids", ([(123, "python.exe", "hermes serve")],), {}),
        (main, "_relaunch_stopped_serves", ({"pending": True, "entries": [{"pid": 123, "port": 9000}]},), {}),
        (main, "_stop_process_trees", ([123, (456, 789)],), {}),
    ],
)
def test_historical_holder_hooks_stop_without_inspecting_or_killing(
    monkeypatch, capsys, module, name, args, kwargs,
):
    import hermes_cli.gateway as gateway
    from hermes_cli import process_identity
    import psutil

    forbidden = Mock(side_effect=AssertionError("retired holder gate performed work"))
    monkeypatch.setattr(gateway, "_is_pid_ancestor_of_current_process", forbidden)
    monkeypatch.setattr(update_cmd_windows, "_psutil", forbidden)
    monkeypatch.setattr(process_identity, "ledger_entries", forbidden)
    monkeypatch.setattr(psutil, "process_iter", forbidden)
    monkeypatch.setattr(psutil, "Process", forbidden)
    monkeypatch.setattr(update_cmd_windows.subprocess, "run", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)
    before = deepcopy((args, kwargs))
    with pytest.raises(SystemExit) as stopped:
        # Frozen and historical main addresses must not return into the old
        # updater's destructive ladder, even when asked for a read-only probe.
        getattr(module, name)(*args, **kwargs)
    assert stopped.value.code == 0
    assert "run `hermes` again" in capsys.readouterr().err.lower()
    assert (args, kwargs) == before
    forbidden.assert_not_called()


def test_command_reaches_checkout_preparation_without_holder_gates(monkeypatch, tmp_path):
    from hermes_cli import update_inventory

    class ReachedCheckout(BaseException):
        pass

    reached = []
    forbidden = Mock(side_effect=AssertionError("current update called a retired holder gate"))
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kwargs: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda token: None)
    monkeypatch.setattr(update_inventory, "collect_runtime_inventory", lambda: update_inventory.UpdatePlan())
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda args: reached.append("backup"))
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", lambda: reached.append("pause"))
    monkeypatch.setattr(main, "_desktop_packaged_executable", lambda root: None)
    monkeypatch.setattr(main, "_desktop_dist_exists", lambda root: False)

    # Recreate the former call-site names as tripwires, not host-OS fakes.
    # The scan remains live for lifecycle ownership, but must not gate updates.
    monkeypatch.setattr(main, "_detect_concurrent_hermes_instances", forbidden, raising=False)
    monkeypatch.setattr(update_cmd, "_clear_windows_venv_holders_or_exit", forbidden, raising=False)
    monkeypatch.setattr(update_cmd_windows, "_detect_venv_python_processes", forbidden)
    monkeypatch.setattr(update_cmd, "_refuse_gateway_ancestor_tree_kill", forbidden)
    monkeypatch.setattr(main, "_is_windows", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)

    def prepare_checkout():
        reached.append("checkout")
        raise ReachedCheckout

    monkeypatch.setattr(update_cmd, "_prepare_git_command", prepare_checkout)
    with pytest.raises(ReachedCheckout):
        main.cmd_update(SimpleNamespace(gateway=False, check=False, yes=True, force=False, force_venv=False))
    assert reached == ["backup", "pause", "checkout"]
    forbidden.assert_not_called()
