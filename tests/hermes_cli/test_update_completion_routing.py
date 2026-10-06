"""All source selection routes hand off once, without old-process maintenance."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import main, main_web_build, update_cmd, update_cmd_zip, update_cmd_maint
from tests.compat.old_updater_support import fresh_child, no_external_work  # noqa: F401


@pytest.mark.parametrize("hook,args,kwargs", [
    (update_cmd._prepare_updated_checkout, ("unused",), {"desktop": False}),
    (update_cmd._reload_config_modules, (), {}),
    (update_cmd._reload_process_scan_modules, (), {}),
    (update_cmd._run_pending_fleet_restart, (), {}),
    (main_web_build._run_with_idle_timeout, (["unused"], "unused"), {"idle_timeout_seconds": 10}),
    (main_web_build._run_npm_install_deterministic, ("unused", "unused"), {"extra_args": ("arg",)}),
    (main_web_build._nixos_build_env, (), {}),
    (main._reexec_dependency_sync_off_windows_shim, (), {}),
    (update_cmd_maint._print_update_summary, (), {
        "node_failures": [], "desktop_build_ok": True, "pre_update_version": None}),
    (update_cmd_maint._print_update_summary, (), {
        "node_failures": ["dashboard"], "desktop_build_ok": False, "pre_update_version": "0.20.1"}),
    (update_cmd_maint._finish_dashboard_update_cleanup, ([],), {}),
    (update_cmd_maint._finish_dashboard_update_cleanup, (["dashboard"],), {
        "already_restarted_units": {"hermes-serve"}}),
])
def test_historical_completion_hook_never_reports_success(hook, args, kwargs, fresh_child, capsys):
    if hook in {update_cmd._prepare_updated_checkout, update_cmd._reload_config_modules,
                update_cmd._reload_process_scan_modules, update_cmd._run_pending_fleet_restart}:
        with pytest.raises(SystemExit) as error:
            hook(*args, **kwargs)
        assert error.value.code == 1
        assert fresh_child.requests == []
        assert 'run `hermes update` again' in capsys.readouterr().err
        return
    with fresh_child.exits():
        hook(*args, **kwargs)


def test_incomplete_handoff_requires_explicit_update_retry(tmp_path, monkeypatch, capsys):
    from hermes_cli import main

    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    completion = Mock()
    monkeypatch.setattr(update_cmd, "run_completion", completion)
    with pytest.raises(SystemExit) as error:
        update_cmd._complete_source_update(None)
    assert error.value.code == 1
    completion.assert_not_called()
    output = capsys.readouterr()
    assert "run `hermes update` again" in output.err
    assert "Update complete" not in output.out
    assert not (tmp_path / ".update-incomplete").exists()
    assert not (tmp_path / ".lazy-refresh-incomplete").exists()


@pytest.mark.parametrize("route", ["pulled", "current", "zip"])
def test_every_route_hands_off_once(route, tmp_path, monkeypatch):
    request = {"branch": "main", "receipt": {"update_id": "c" * 32}}
    handed_off = []
    monkeypatch.setattr(update_cmd, "_complete_source_update", handed_off.append, raising=False)
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(
        PROJECT_ROOT=tmp_path, _resolve_update_branch=lambda args: "main"))
    monkeypatch.setattr(update_cmd, "_verify_head_after_pull", lambda *a, **kw: "new-sha")
    monkeypatch.setattr(update_cmd, "_prepare_updated_checkout", lambda *a, **kw: pytest.fail("old-process preparation"))
    monkeypatch.setattr(update_cmd, "_write_fleet_restart_pending_marker", lambda **kw: None)
    monkeypatch.setattr(update_cmd, "_sweep_bytecode_after_update", lambda *a: None)
    monkeypatch.setattr(update_cmd_zip, "_abort_zip_update_if_dirty_tree", lambda: None)
    swap = Mock()
    monkeypatch.setattr(update_cmd_zip, "_download_and_swap_zip", swap)
    plan = update_cmd._CheckoutPlan(
        in_place_update=False, auto_stash_ref=None, parked_branch_switched=False,
        upstream_checked=True, commit_count=1, prompt_for_restore=False, switch_block_reason=None)
    if route == "pulled":
        update_cmd._apply_pulled_update(
            ["git"], "main", "old-sha", plan, _windows_gateway_resume=None,
            completion_request=request)
    elif route == "current":
        update_cmd._finish_already_up_to_date(
            ["git"], "main", "main", plan, gw_input_fn=None, completion_request=request)
    else:
        assert update_cmd_zip._update_via_zip(SimpleNamespace(), completion_request=request) is True
        swap.assert_called_once()
    assert handed_off == [request]
