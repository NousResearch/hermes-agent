from types import SimpleNamespace

import hermes_cli.gateway as gateway_cli
from hermes_cli.gateway import (
    GatewayRuntimeSnapshot,
    get_gateway_runtime_health,
    _runtime_health_lines,
)


def _platform(slug: str):
    return SimpleNamespace(value=slug)


def _config(*platforms: str):
    return SimpleNamespace(
        get_connected_platforms=lambda: [_platform(platform) for platform in platforms]
    )


def _snapshot(*, running: bool = True, service_installed: bool = False, service_running: bool = False):
    return GatewayRuntimeSnapshot(
        manager="manual process",
        service_installed=service_installed,
        service_running=service_running,
        gateway_pids=(1234,) if running and not service_running else (),
    )


def _patch_runtime_health_deps(monkeypatch, *, snapshot=None, config=None, status=None):
    monkeypatch.setattr(
        gateway_cli,
        "get_gateway_runtime_snapshot",
        lambda system=False: snapshot or _snapshot(),
    )
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: config or _config(),
    )
    monkeypatch.setattr("gateway.status.read_runtime_status", lambda: status)
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)


def test_gateway_runtime_health_handles_missing_status(monkeypatch):
    _patch_runtime_health_deps(
        monkeypatch,
        snapshot=_snapshot(running=True),
        config=_config("telegram"),
        status=None,
    )

    health = get_gateway_runtime_health()

    assert health.runtime_status_available is False
    assert health.gateway_state is None
    assert health.platforms == {}


def test_gateway_runtime_health_treats_unparseable_status_as_unavailable(monkeypatch):
    _patch_runtime_health_deps(
        monkeypatch,
        snapshot=_snapshot(running=True),
        config=_config("telegram"),
        status=None,
    )

    health = get_gateway_runtime_health()

    assert health.runtime_status_available is False
    assert health.gateway_state is None


def test_gateway_runtime_health_filters_configured_platforms(monkeypatch):
    _patch_runtime_health_deps(
        monkeypatch,
        snapshot=_snapshot(running=True),
        config=_config("telegram", "discord"),
        status={
            "gateway_state": "running",
            "updated_at": "2026-04-23T00:00:00+00:00",
            "platforms": {
                "telegram": {"state": "connected"},
                "discord": {"state": "connecting"},
                "slack": {"state": "connected"},
            },
        },
    )

    health = get_gateway_runtime_health()

    assert health.runtime_status_available is True
    assert health.gateway_state == "running"
    assert health.updated_at == "2026-04-23T00:00:00+00:00"
    assert set(health.platforms) == {"telegram", "discord"}


def test_gateway_runtime_health_drops_stale_platforms_when_not_running(monkeypatch):
    _patch_runtime_health_deps(
        monkeypatch,
        snapshot=_snapshot(running=False),
        config=_config("telegram"),
        status={
            "gateway_state": "running",
            "platforms": {"telegram": {"state": "connected"}},
        },
    )

    health = get_gateway_runtime_health()

    assert health.gateway_state == "stopped"
    assert health.platforms == {}


def test_gateway_runtime_health_exposes_systemd_properties(monkeypatch):
    _patch_runtime_health_deps(
        monkeypatch,
        snapshot=_snapshot(running=False, service_installed=True),
        config=_config("telegram"),
        status={"gateway_state": "stopped", "platforms": {}},
    )
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(
        gateway_cli,
        "_read_systemd_unit_properties",
        lambda system=False: {
            "ActiveState": "activating",
            "SubState": "auto-restart",
        },
    )

    health = get_gateway_runtime_health()

    assert health.systemd_unit == {
        "ActiveState": "activating",
        "SubState": "auto-restart",
    }


def test_runtime_health_lines_include_fatal_platform_and_startup_reason(monkeypatch):
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "startup_failed",
            "exit_reason": "telegram conflict",
            "platforms": {
                "telegram": {
                    "state": "fatal",
                    "error_message": "another poller is active",
                }
            },
        },
    )

    lines = _runtime_health_lines()

    assert "⚠ telegram: another poller is active" in lines
    assert "⚠ Last startup issue: telegram conflict" in lines


def test_runtime_status_running_pid_validates_live_gateway_record(monkeypatch):
    from gateway import status as status_mod

    runtime = {
        "pid": 12345,
        "kind": "hermes-gateway",
        "argv": ["/opt/hermes/hermes_cli/main.py", "gateway", "run", "--replace"],
        "start_time": None,
        "gateway_state": "running",
    }
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: pid == 12345)
    monkeypatch.setattr(status_mod, "_get_process_start_time", lambda pid: None)
    monkeypatch.setattr(status_mod, "_looks_like_gateway_process", lambda pid: False)

    assert status_mod.get_runtime_status_running_pid(runtime) == 12345


def test_runtime_status_running_pid_rejects_stopped_record(monkeypatch):
    from gateway import status as status_mod

    runtime = {
        "pid": 12345,
        "kind": "hermes-gateway",
        "argv": ["/opt/hermes/hermes_cli/main.py", "gateway", "run", "--replace"],
        "gateway_state": "stopped",
    }
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: True)

    assert status_mod.get_runtime_status_running_pid(runtime) is None
