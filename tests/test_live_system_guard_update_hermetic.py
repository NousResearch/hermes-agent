"""Synthetic updater regressions for the live-system guard."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import config as hermes_config
from hermes_cli import gateway
from hermes_cli import gateway_windows
from hermes_cli import main as hermes_main
from gateway import status as gateway_status


SYNTHETIC_PID = 987_654_321
NONEXISTENT_GATEWAY_EXECUTABLE = "/definitely-not-a-real-hermes-gateway"


def _guard_module():
    expected = Path(__file__).with_name("conftest.py").resolve()
    for module in tuple(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if module_file and Path(module_file).resolve() == expected:
            return module
    raise AssertionError("pytest did not load tests/conftest.py")


def _recording(name, calls, result):
    def _fake(*args, **kwargs):
        calls.append((name, args, kwargs))
        return result

    return _fake


def _prepare_update(monkeypatch, tmp_path, calls):
    """Make ``cmd_update`` deterministic without starting any subprocess."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_install_hangup_protection", lambda **_kw: None)
    monkeypatch.setattr(hermes_main, "_finalize_update_output", lambda _state: None)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda _args: None)
    monkeypatch.setattr(hermes_main, "_discard_lockfile_churn", lambda *_args: None)
    monkeypatch.setattr(hermes_main, "_get_origin_url", lambda *_args: "")
    monkeypatch.setattr(hermes_main, "_refresh_active_lazy_features", lambda: None)
    monkeypatch.setattr(hermes_main, "_kill_stale_dashboard_processes", lambda: None)
    monkeypatch.setattr(hermes_main, "_is_fork", lambda _url: False)
    monkeypatch.setattr(hermes_main, "_is_termux_env", lambda env=None: False)
    monkeypatch.setattr(hermes_main, "_load_installable_optional_extras", lambda *_a: [])
    monkeypatch.setattr(hermes_main._time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hermes_config, "detect_install_method", lambda *_a: "git")
    monkeypatch.setattr(hermes_config, "is_managed", lambda: False)
    monkeypatch.setattr(hermes_config, "is_unsupported_install_method", lambda _m: False)
    monkeypatch.setattr(hermes_config, "get_missing_env_vars", lambda **_kw: [])
    monkeypatch.setattr(hermes_config, "get_missing_config_fields", lambda: [])
    monkeypatch.setattr(hermes_config, "check_config_version", lambda: (5, 5))
    monkeypatch.setattr(hermes_config, "migrate_config", lambda **_kw: {})
    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"updates": {"refresh_cua_driver": False}},
    )
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: "/synthetic/uv")
    monkeypatch.setattr("hermes_cli.managed_uv.update_managed_uv", lambda: None)
    monkeypatch.setattr(
        "tools.skills_sync.sync_skills",
        lambda **_kw: {"copied": [], "updated": [], "user_modified": []},
    )
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", lambda: [])

    def fake_run(command, **kwargs):
        calls.append(("subprocess.run", command, kwargs))
        if "rev-parse" in command:
            return SimpleNamespace(stdout="main\n", stderr="", returncode=0)
        if "rev-list" in command:
            return SimpleNamespace(stdout="1\n", stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(
        hermes_main.subprocess,
        "Popen",
        _recording("subprocess.Popen", calls, SimpleNamespace(pid=SYNTHETIC_PID)),
    )
    monkeypatch.setattr(
        gateway,
        "_prepare_profile_gateway_update_restart",
        _recording("prepare_profile_restart", calls, None),
    )
    monkeypatch.setattr(
        gateway,
        "_graceful_restart_via_sigusr1",
        _recording("graceful_restart", calls, False),
    )
    monkeypatch.setattr(
        gateway,
        "_wait_for_gateway_exit",
        _recording("wait_for_gateway_exit", calls, None),
    )


def _run_update():
    hermes_main.cmd_update(
        SimpleNamespace(
            branch=None,
            check=False,
            force=True,
            force_venv=True,
            gateway=False,
            yes=True,
        )
    )


def test_cmd_update_keeps_discovery_and_restart_boundaries_inert(monkeypatch, tmp_path):
    """The updater must not reach discovery or restart boundaries in tests."""
    conftest = _guard_module()
    calls = []
    _prepare_update(monkeypatch, tmp_path, calls)

    guarded_targets = [
        (gateway, "find_gateway_pids", []),
        (gateway, "_scan_gateway_pids", []),
        (gateway, "_get_service_pids", set()),
        (gateway, "find_profile_gateway_processes", []),
        (gateway, "supports_systemd_services", False),
        (gateway, "is_macos", False),
        (gateway_windows, "is_installed", False),
        (gateway, "launch_detached_profile_gateway_restart", False),
        (gateway, "launch_detached_gateway_restart_by_cmdline", False),
        (gateway_windows, "_spawn_detached", None),
        (gateway_status, "terminate_pid", None),
        (gateway, "refresh_systemd_unit_if_needed", False),
        (gateway, "refresh_launchd_plist_if_needed", False),
    ]
    for module, name, result in guarded_targets:
        guarded = getattr(module, name)
        assert getattr(guarded, "_live_system_guard_inert", False), name
        monkeypatch.setattr(guarded, "__wrapped__", _recording(name, calls, result))

    _run_update()

    guarded_names = {name for _module, name, _result in guarded_targets}
    assert not [call for call in calls if call[0] in guarded_names]
    assert not [
        call
        for call in calls
        if call[0]
        in {
            "subprocess.Popen",
            "prepare_profile_restart",
            "graceful_restart",
            "wait_for_gateway_exit",
            "terminate_pid",
        }
    ]


def test_updater_terminate_canary_fails_when_only_its_wrapper_is_removed(monkeypatch, tmp_path):
    """The retained terminate wrapper is the only thing keeping its recorder idle."""
    calls = []
    _prepare_update(monkeypatch, tmp_path, calls)

    guarded_terminate = gateway_status.terminate_pid
    assert getattr(guarded_terminate, "_live_system_guard_inert", False)
    monkeypatch.setattr(
        guarded_terminate, "__wrapped__", _recording("terminate_pid", calls, None)
    )

    def _exercise_termination_boundary():
        gateway_status.terminate_pid(SYNTHETIC_PID, force=True)
        return None

    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", _exercise_termination_boundary)
    _run_update()
    assert not [call for call in calls if call[0] == "terminate_pid"]

    with monkeypatch.context() as reverted:
        reverted.setattr(
            gateway_status,
            "terminate_pid",
            guarded_terminate.__wrapped__,
        )
        _run_update()
        with pytest.raises(AssertionError, match="unexpected live-system boundary"):
            assert not [call for call in calls if call[0] == "terminate_pid"], (
                "unexpected live-system boundary"
            )


def test_updater_detached_spawn_canary_fails_when_only_its_wrapper_is_removed(monkeypatch, tmp_path):
    """The retained detached-spawn wrapper is the only thing keeping its recorder idle."""
    calls = []
    _prepare_update(monkeypatch, tmp_path, calls)

    guarded_spawn = gateway.launch_detached_gateway_restart_by_cmdline
    assert getattr(guarded_spawn, "_live_system_guard_inert", False)
    monkeypatch.setattr(
        guarded_spawn, "__wrapped__", _recording("detached_spawn", calls, False)
    )

    def _exercise_detached_spawn_boundary(_token):
        return gateway.launch_detached_gateway_restart_by_cmdline(
            SYNTHETIC_PID,
            [NONEXISTENT_GATEWAY_EXECUTABLE, "gateway", "run"],
        )

    monkeypatch.setattr(hermes_main, "_resume_windows_gateways_after_update", _exercise_detached_spawn_boundary)
    _run_update()
    assert not [call for call in calls if call[0] == "detached_spawn"]

    with monkeypatch.context() as reverted:
        reverted.setattr(
            gateway,
            "launch_detached_gateway_restart_by_cmdline",
            guarded_spawn.__wrapped__,
        )
        _run_update()
        with pytest.raises(AssertionError, match="unexpected live-system boundary"):
            assert not [call for call in calls if call[0] == "detached_spawn"], (
                "unexpected live-system boundary"
            )


@pytest.mark.parametrize(
    "refresh_name",
    ["refresh_systemd_unit_if_needed", "refresh_launchd_plist_if_needed"],
)
def test_updater_unit_write_canary_fails_closed_before_real_path_write(
    monkeypatch, tmp_path, refresh_name
):
    """A real unit path is rejected before the recording write delegate can run."""
    calls = []
    _prepare_update(monkeypatch, tmp_path, calls)
    guarded_refresh = getattr(gateway, refresh_name)
    assert getattr(guarded_refresh, "_live_system_guard_inert", False)
    monkeypatch.setattr(
        guarded_refresh, "__wrapped__", _recording("unit_write", calls, False)
    )

    def _exercise_unit_write_boundary():
        try:
            getattr(gateway, refresh_name)()
        except RuntimeError as exc:
            assert "non-test path" in str(exc)
        return None

    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", _exercise_unit_write_boundary)
    _run_update()
    assert not [call for call in calls if call[0] == "unit_write"]

    with monkeypatch.context() as reverted:
        reverted.setattr(gateway, refresh_name, guarded_refresh.__wrapped__)
        _run_update()
        with pytest.raises(AssertionError, match="unexpected live-system boundary"):
            assert not [call for call in calls if call[0] == "unit_write"], (
                "unexpected live-system boundary"
            )
