"""Retired soft-build hooks stop old updaters instead of claiming completion.

Live source builds now raise on failure before maintenance runs. Old updaters
can still import these frozen names after swapping their checkout.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd
from hermes_cli import update_cmd_maint


@pytest.mark.parametrize(
    "name,args,kwargs",
    [
        ("_print_update_summary", (), {
            "node_failures": [], "desktop_build_ok": True, "pre_update_version": None,
        }),
        ("_print_update_summary", (), {
            "node_failures": ["dashboard"], "desktop_build_ok": False,
            "pre_update_version": "0.20.1",
        }),
        ("_finish_dashboard_update_cleanup", ([],), {}),
        ("_finish_dashboard_update_cleanup", (["dashboard"],), {
            "already_restarted_units": {"hermes-serve"},
        }),
    ],
)
def test_historical_completion_hooks_stop_before_work(name, args, kwargs, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("old updater attempted post-update work")

    monkeypatch.setattr(update_cmd, "_post_update_sqlite_runtime_status", forbidden)
    monkeypatch.setattr(update_cmd, "_reload_process_scan_modules", forbidden)
    monkeypatch.setattr(
        update_cmd, "_m", lambda: SimpleNamespace(_kill_stale_dashboard_processes=forbidden),
    )
    with pytest.raises(SystemExit) as exc:
        getattr(update_cmd_maint, name)(*args, **kwargs)
    assert exc.value.code == 0
    output = capsys.readouterr()
    assert "run `hermes` again" in output.err
    assert "Update complete" not in output.out


def test_gateway_exit_code_file_tracks_verified_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: tmp_path)
    update_cmd._write_gateway_update_exit_code(True)
    assert (tmp_path / ".update_exit_code").read_text(encoding="utf-8") == "0"
    update_cmd._write_gateway_update_exit_code(False)
    assert (tmp_path / ".update_exit_code").read_text(encoding="utf-8") == "1"


def test_verified_completion_keeps_success_banner(capsys, monkeypatch):
    monkeypatch.setattr(update_cmd, "_branch_head_suffix", lambda: "")
    monkeypatch.setattr(update_cmd, "_post_update_sqlite_runtime_status", lambda: (True, None))

    assert update_cmd_maint._print_verified_update_completion("✓ Update complete!") is True

    out = capsys.readouterr().out
    assert "✓ Update complete!" in out
    assert "partially complete" not in out
    assert "desktop" not in out
    assert "Node" not in out
    assert "=== hermes-update completed" not in out


def test_verified_completion_emits_dashboard_receipt_only_on_success(monkeypatch, capsys):
    action_id = "a" * 32
    monkeypatch.setenv("HERMES_ACTION_ID", action_id)
    monkeypatch.setattr(update_cmd, "_branch_head_suffix", lambda: "")
    monkeypatch.setattr(
        update_cmd, "_post_update_sqlite_runtime_status",
        lambda: (False, SimpleNamespace(sqlite_version_string="3.46.1")),
    )
    assert update_cmd_maint._print_verified_update_completion("✓ Update complete!") is False
    assert f"=== hermes-update completed {action_id} ===" not in capsys.readouterr().out

    monkeypatch.setattr(update_cmd, "_post_update_sqlite_runtime_status", lambda: (True, None))
    assert update_cmd_maint._print_verified_update_completion("✓ Update complete!") is True
    assert f"=== hermes-update completed {action_id} ===" in capsys.readouterr().out


def test_unavailable_sqlite_probe_retains_existing_nonblocking_behavior(monkeypatch, capsys):
    monkeypatch.setattr(update_cmd, "_branch_head_suffix", lambda: "")
    monkeypatch.setattr(update_cmd, "_post_update_sqlite_runtime_status", lambda: (False, None))

    assert update_cmd_maint._print_verified_update_completion("✓ Update complete!") is True
    assert "✓ Update complete!" in capsys.readouterr().out


def test_maintenance_returns_sqlite_verdict_without_frontend_flags(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=tmp_path))
    monkeypatch.setattr("hermes_cli.macos_tcc_anchor.ensure_tcc_anchor", lambda: None)
    monkeypatch.setattr("hermes_cli.model_catalog.seed_cache_from_checkout", lambda root: False)
    monkeypatch.setattr(update_cmd, "_check_and_apply_config_migration", lambda **kwargs: None)
    for name in (
        "_verify_and_restore_state_dbs_post_update", "_print_bundled_skills_sync_report",
        "_sync_profiles_after_update", "_print_post_update_notices_and_self_heals",
    ):
        monkeypatch.setattr(update_cmd_maint, name, lambda: None)
    monkeypatch.setattr(
        update_cmd, "_post_update_sqlite_runtime_status",
        lambda: (False, SimpleNamespace(sqlite_version_string="3.46.1")),
    )

    complete = update_cmd_maint._run_post_update_maintenance(
        assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
        had_desktop_app_before_update=False, pre_update_version=None,
    )

    assert complete is False
    assert "Update complete" not in capsys.readouterr().out
    monkeypatch.setattr(update_cmd, "_post_update_sqlite_runtime_status", lambda: (True, None))
    monkeypatch.setattr(update_cmd, "_branch_head_suffix", lambda: "")
    assert update_cmd_maint._run_post_update_maintenance(
        assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
        had_desktop_app_before_update=False, pre_update_version=None,
    ) is True
    assert "✓ Update complete!" in capsys.readouterr().out


@pytest.mark.parametrize("already_restarted_units", [None, {"hermes-serve"}])
def test_dashboard_refresh_reloads_then_preserves_restart_bookkeeping(
    already_restarted_units, monkeypatch, capsys,
):
    order = []
    monkeypatch.setattr(update_cmd, "_reload_process_scan_modules", lambda: order.append("reload"))

    def kill(**kwargs):
        order.append(kwargs)
        return {"unrecovered": [1234]}

    monkeypatch.setattr(
        update_cmd, "_m", lambda: SimpleNamespace(_kill_stale_dashboard_processes=kill),
    )
    update_cmd_maint._refresh_dashboard_after_update(already_restarted_units=already_restarted_units)

    assert order == ["reload", {
        "restart_managed": True, "already_restarted_units": already_restarted_units,
    }]
    assert "could not be auto-restarted" in capsys.readouterr().out
