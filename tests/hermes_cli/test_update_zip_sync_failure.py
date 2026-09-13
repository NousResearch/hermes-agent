"""A failed dependency transaction must stop the ZIP update, not report success."""
from types import SimpleNamespace
import subprocess
from unittest.mock import patch

import pytest

import pm
import hermes_cli.main as main
from hermes_cli import update_cmd, update_cmd_maint, update_cmd_zip


def test_zip_dependency_failure_propagates_before_followup_mutations(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    with (
        patch.object(update_cmd_zip, "_abort_zip_update_if_dirty_tree"),
        patch.object(update_cmd_zip, "_download_and_swap_zip"),
        patch.object(update_cmd_maint, "_sweep_bytecode_after_update"),
        patch.object(
            update_cmd_maint, "_prepare_updated_checkout",
            side_effect=pm.InstallError("venv", "network unavailable"),
        ) as prepare,
        patch.object(update_cmd_maint, "_print_bundled_skills_sync_report") as skills,
        patch.object(update_cmd_maint, "_print_verified_update_completion") as summary,
    ):
        with pytest.raises(pm.InstallError, match="network unavailable"):
            update_cmd_zip._update_via_zip(SimpleNamespace(branch="main"))

    prepare.assert_called_once_with(tmp_path, desktop=False)
    skills.assert_not_called()
    summary.assert_not_called()


def test_pull_dependency_failure_keeps_recovery_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    post_pull_sha = "b" * 40
    with (
        patch.object(update_cmd, "_verify_head_after_pull", return_value=post_pull_sha),
        patch.object(update_cmd, "_sweep_bytecode_after_update"),
        patch.object(
            update_cmd, "_prepare_updated_checkout",
            side_effect=pm.InstallError("venv", "sync stopped"),
        ) as prepare,
        patch.object(update_cmd, "_run_post_update_maintenance") as maintenance,
        patch.object(update_cmd, "_restart_gateway_fleet_after_update") as restart,
    ):
        with pytest.raises(pm.InstallError, match="sync stopped"):
            update_cmd._apply_pulled_update(
                ["git"], "main", "a" * 40, SimpleNamespace(in_place_update=False),
                SimpleNamespace(assume_yes=True, gw_input_fn=None),
                gateway_mode=False, is_fork=False, desktop_dir=tmp_path / "apps" / "desktop",
                had_desktop_app_before_update=False, pre_update_snapshot_id=None,
                _pre_update_plan=None, _windows_gateway_resume=[],
            )

    prepare.assert_called_once_with(tmp_path, desktop=False)
    marker = update_cmd._fleet_restart_pending_marker_path()
    assert f"expected_sha={post_pull_sha}" in marker.read_text(encoding="utf-8")
    maintenance.assert_not_called()
    restart.assert_not_called()


@pytest.mark.parametrize("route", ["direct", "git-failure"])
@pytest.mark.parametrize("gateway_mode", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_zip_callers_propagate_completion(tmp_path, monkeypatch, route, gateway_mode, complete):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    exit_markers = []
    monkeypatch.setattr(update_cmd, "_write_gateway_update_exit_code", exit_markers.append)

    def zip_completion(received, **context):
        assert received is args
        assert context["gateway_mode"] is gateway_mode
        # Completion owns the marker now; neither caller may emit it again.
        if context["gateway_mode"]:
            update_cmd._write_gateway_update_exit_code(complete)
        return complete
    monkeypatch.setattr(update_cmd, "_update_via_zip", zip_completion)
    resumed = []
    args = SimpleNamespace(branch="main")

    if route == "direct":
        monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *args: SimpleNamespace(
            gw_input_fn=None, assume_yes=True, pre_update_version=None))
        monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda args: None)
        monkeypatch.setattr(main, "_run_pre_update_backup", lambda args: None)
        monkeypatch.setattr(main, "_pause_windows_gateways_for_update", lambda: None)
        monkeypatch.setattr(main, "_resume_windows_gateways_after_update", resumed.append)
        monkeypatch.setattr(main, "_desktop_packaged_executable", lambda root: None)
        monkeypatch.setattr(main, "_desktop_dist_exists", lambda root: False)
        monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (True, [], False))
        monkeypatch.setattr(update_cmd, "_source_update_channel", lambda args: "main")

        def invoke():
            update_cmd._cmd_update_impl(args, gateway_mode=gateway_mode)
    else:
        monkeypatch.setattr(update_cmd, "_should_zip_fallback_on_update_error", lambda error: True)

        def invoke():
            update_cmd._handle_update_called_process_error(
                subprocess.CalledProcessError(1, ["git", "fetch"]), args, gateway_mode, False)

    if complete:
        invoke()
    else:
        with pytest.raises(SystemExit) as failure:
            invoke()
        assert failure.value.code == 1
    assert exit_markers == ([complete] if gateway_mode else [])
    assert resumed == []  # no paused gateways: completion owns the success-path resume
