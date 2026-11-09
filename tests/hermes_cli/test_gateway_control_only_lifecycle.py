"""Process lifecycle diagnostics must remain owned by serving gateway boots."""

from __future__ import annotations

import atexit
import json
import sys


def test_control_only_run_preserves_exit_diagnostics_and_serving_run_appends_them(tmp_path, monkeypatch):
    from hermes_cli import gateway
    from gateway import run_exit
    from hermes_cli.observability import shared_metrics_process as process_metrics
    from hermes_cli.observability import shared_metrics_setup as setup_metrics
    from hermes_cli.observability import shared_metrics_update as update_metrics
    from hermes_cli.observability import shared_metrics_install_run as install_metrics

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": False}}},
    )
    atexit_callbacks = []
    monkeypatch.setattr(
        atexit, "register",
        lambda callback, *args, **kwargs: atexit_callbacks.append((callback, args)),
    )
    monkeypatch.setattr(sys, "excepthook", lambda *args: None)
    diag_path = tmp_path / "logs" / "gateway-exit-diag.log"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    original = '{"tag":"prior-serving-exit"}\n'
    diag_path.write_text(original, encoding="utf-8")
    marker_paths = [
        process_metrics.markers_dir(tmp_path) / "gateway-prior.json",
        setup_metrics.markers_dir(tmp_path) / "provider-prior.json",
        tmp_path / "telemetry" / "shared_metrics" / "desktop_onboarding" / "provider.completed",
        update_metrics.pending_updates_dir(tmp_path) / "update-prior.json",
        install_metrics.pending_installs_dir(tmp_path) / "install-prior.json",
    ]
    for index, path in enumerate(marker_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"prior-metrics-artifact-{index}\n".encode())
    metric_artifacts = {path: path.read_bytes() for path in marker_paths}
    prior_atexit_callbacks = list(atexit_callbacks)
    prior_metrics_callbacks = [
        item for item in prior_atexit_callbacks if item[0] is process_metrics.stamp_exit
    ]
    monkeypatch.setattr(gateway, "_guard_official_docker_root_gateway", lambda: None)
    monkeypatch.setattr(gateway, "_attach_to_host_gateway_or_guard", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_guard_supervised_gateway_conflict", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_guard_existing_gateway_process_conflict", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_stdin_is_tty", lambda: False)
    monkeypatch.setattr(gateway, "_windows_console_window_attached", lambda: False)
    monkeypatch.setattr(gateway, "_windows_gateway_breakaway_state", lambda: False)
    monkeypatch.setattr(gateway, "_windows_gateway_should_absorb_console_controls", lambda: False)
    monkeypatch.setattr(gateway, "_gateway_detached_env", lambda: False)
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway.asyncio, "run", lambda coroutine: (coroutine.close(), True)[1])
    finalizers = []
    monkeypatch.setattr(run_exit, "exit_after_graceful_shutdown",
                        lambda code, *, mark_lifecycle=True: finalizers.append((code, mark_lifecycle)))
    monkeypatch.setattr(gateway, "_apply_startup_watchdog_config",
                        lambda: (_ for _ in ()).throw(AssertionError("watchdog must stay unarmed")))
    monkeypatch.setattr(gateway, "_respawn_storm_backoff",
                        lambda: (_ for _ in ()).throw(AssertionError("control-only boot must not back off")))

    gateway.run_gateway(control_only=True)
    assert diag_path.read_text(encoding="utf-8") == original
    assert {path: path.read_bytes() for path in marker_paths} == metric_artifacts
    assert finalizers == [(0, False)]
    gateway.run_gateway(control_only=True)
    assert diag_path.read_text(encoding="utf-8") == original
    assert {path: path.read_bytes() for path in marker_paths} == metric_artifacts
    assert finalizers == [(0, False), (0, False)]
    assert [item for item in atexit_callbacks if item[0] is process_metrics.stamp_exit] == prior_metrics_callbacks

    monkeypatch.setattr(gateway, "_apply_startup_watchdog_config", lambda: None)
    monkeypatch.setattr(gateway, "_respawn_storm_backoff", lambda: None)
    gateway.run_gateway()
    records = [json.loads(line) for line in diag_path.read_text(encoding="utf-8").splitlines()]
    assert [record["tag"] for record in records] == [
        "prior-serving-exit", "gateway.start", "asyncio.run.returned", "gateway.exit_clean"]
    assert finalizers == [(0, False), (0, False), (0, True)]
    assert [item for item in atexit_callbacks if item[0] is process_metrics.stamp_exit] == prior_metrics_callbacks


def test_control_only_process_finalizer_preserves_running_marker_and_skips_metrics_flush(
    tmp_path, monkeypatch
):
    from gateway import run_exit
    import gateway.run as gateway_run
    from hermes_cli.observability import shared_metrics_process as process_metrics

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    marker = process_metrics.markers_dir(tmp_path) / "gateway-123.json"
    marker.parent.mkdir(parents=True)
    marker_bytes = b'{"kind":"gateway","pid":123,"state":"running"}\n'
    marker.write_bytes(marker_bytes)
    exits = []
    monkeypatch.setattr(gateway_run.os, "_exit", exits.append)
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_process.flush_before_hard_exit",
        lambda: (_ for _ in ()).throw(AssertionError("control-only must not flush serving metrics")),
    )

    run_exit.exit_after_graceful_shutdown(0, mark_lifecycle=False)

    assert exits == [0]
    assert marker.read_bytes() == marker_bytes


def test_normal_serving_run_updates_metrics_marker(tmp_path, monkeypatch):
    from hermes_cli import gateway
    import gateway.run as gateway_run
    from hermes_cli.observability import shared_metrics_process as process_metrics

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": True}}},
    )
    monkeypatch.setattr(process_metrics, "_STATE", {})
    atexit_callbacks = []
    monkeypatch.setattr(
        atexit, "register",
        lambda callback, *args, **kwargs: atexit_callbacks.append((callback, args)),
    )
    monkeypatch.setattr(sys, "excepthook", lambda *args: None)
    monkeypatch.setattr(gateway, "_guard_official_docker_root_gateway", lambda: None)
    monkeypatch.setattr(gateway, "_attach_to_host_gateway_or_guard", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_guard_supervised_gateway_conflict", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_guard_existing_gateway_process_conflict", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_stdin_is_tty", lambda: False)
    monkeypatch.setattr(gateway, "_windows_console_window_attached", lambda: False)
    monkeypatch.setattr(gateway, "_windows_gateway_breakaway_state", lambda: False)
    monkeypatch.setattr(gateway, "_windows_gateway_should_absorb_console_controls", lambda: False)
    monkeypatch.setattr(gateway, "_gateway_detached_env", lambda: False)
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway.asyncio, "run", lambda coroutine: (coroutine.close(), True)[1])
    exits = []
    monkeypatch.setattr(gateway_run.os, "_exit", exits.append)
    monkeypatch.setattr(gateway, "_apply_startup_watchdog_config", lambda: None)
    monkeypatch.setattr(gateway, "_respawn_storm_backoff", lambda: None)

    gateway.run_gateway()

    marker = process_metrics.markers_dir(tmp_path) / f"gateway-{process_metrics.os.getpid()}.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["state"] == "clean"
    assert exits == [0]
    assert (process_metrics.stamp_exit, ("clean",)) in atexit_callbacks


def test_control_only_run_and_later_serving_run_preserve_explicit_control_only_unit(tmp_path, monkeypatch):
    from hermes_cli import gateway
    from gateway import run_exit

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    unit_path = tmp_path / "systemd" / "hermes-gateway.service"
    unit_path.parent.mkdir()
    custom_unit = (
        "[Unit]\nDescription=operator gateway\n\n[Service]\n"
        "ExecStart=/custom/hermes gateway run --control-only\nRestart=always\n"
    )
    unit_path.write_text(custom_unit, encoding="utf-8")
    systemctl_calls = []
    monkeypatch.setattr(gateway, "_guard_official_docker_root_gateway", lambda: None)
    monkeypatch.setattr(gateway, "_attach_to_host_gateway_or_guard", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_guard_supervised_gateway_conflict", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_guard_existing_gateway_process_conflict", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_stdin_is_tty", lambda: False)
    monkeypatch.setattr(gateway, "_windows_console_window_attached", lambda: False)
    monkeypatch.setattr(gateway, "_windows_gateway_breakaway_state", lambda: False)
    monkeypatch.setattr(gateway, "_windows_gateway_should_absorb_console_controls", lambda: False)
    monkeypatch.setattr(gateway, "_gateway_detached_env", lambda: False)
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: unit_path)
    monkeypatch.setattr(gateway, "systemd_unit_is_current", lambda system=False: False)
    monkeypatch.setattr(
        "hermes_cli.gateway_service_owner.unit_belongs_to_caller",
        lambda *args: True,
    )
    monkeypatch.setattr(gateway, "_prepare_service_launcher", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_run_systemctl", lambda args, **kwargs: systemctl_calls.append(args))
    monkeypatch.setattr(gateway, "_apply_startup_watchdog_config", lambda: None)
    monkeypatch.setattr(gateway, "_respawn_storm_backoff", lambda: None)
    monkeypatch.setattr(gateway.asyncio, "run", lambda coroutine: (coroutine.close(), True)[1])
    monkeypatch.setattr(run_exit, "exit_after_graceful_shutdown", lambda code, *, mark_lifecycle=True: None)

    gateway.run_gateway(control_only=True)
    gateway.run_gateway(control_only=True)
    assert unit_path.read_text(encoding="utf-8") == custom_unit
    assert systemctl_calls == []

    gateway.run_gateway()
    assert unit_path.read_text(encoding="utf-8") == custom_unit
    assert systemctl_calls == []


def test_startup_watchdog_fast_path_skips_control_only_command():
    from hermes_cli._startup_fast import argv_is_gateway_run

    assert argv_is_gateway_run(["-p", "default", "gateway", "run"])
    assert not argv_is_gateway_run(["-p", "default", "gateway", "run", "--control-only"])
