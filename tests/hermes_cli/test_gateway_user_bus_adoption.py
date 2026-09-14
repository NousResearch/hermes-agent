"""#104893: a gateway started by a system-level systemd unit adopts its own user bus at boot."""

import os

import pytest

import hermes_cli.gateway as gw


class _BootReached(Exception):
    """Stands in for the first boot step past the adoption call."""


def _fake_user_bus(monkeypatch, *, present: bool) -> str:
    """Fake the on-disk state ``loginctl enable-linger`` leaves behind (or its absence)."""
    runtime_dir = f"/run/user/{os.getuid()}"
    monkeypatch.setattr(
        gw, "_runtime_dir_is_ours", lambda d: present and str(d) == runtime_dir)
    monkeypatch.setattr(
        gw, "_path_exists_safe", lambda p: present and str(p) == f"{runtime_dir}/bus")
    return runtime_dir


def _service_manager_env(monkeypatch) -> None:
    """The environment systemd hands a system-level unit: no bus, but INVOCATION_ID."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.setenv("INVOCATION_ID", "system-unit-104893")


@pytest.mark.linux_only
def test_service_gateway_boot_adopts_its_user_bus(monkeypatch):
    runtime_dir = _fake_user_bus(monkeypatch, present=True)
    _service_manager_env(monkeypatch)
    for guard in (
        "_guard_official_docker_root_gateway",
        "_guard_named_profile_under_multiplexer",
        "_guard_supervised_gateway_conflict",
        "_guard_existing_gateway_process_conflict",
        "_apply_startup_watchdog_config",
    ):
        monkeypatch.setattr(gw, guard, lambda *_a, **_kw: None)

    def _stop_here() -> bool:
        raise _BootReached

    # First boot step after the adoption — stop there instead of starting a gateway.
    monkeypatch.setattr(gw, "supports_systemd_services", _stop_here)

    with pytest.raises(_BootReached):
        gw.run_gateway()

    assert os.environ["XDG_RUNTIME_DIR"] == runtime_dir
    assert os.environ["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={runtime_dir}/bus"

    # Worker environments are snapshots of os.environ taken at various points in the
    # dispatch paths, so the bus address must also survive the secret scrubber they all
    # go through — otherwise systemd-run still cannot connect and the fix is a no-op.
    from tools.environments.local import build_subprocess_env

    worker_env = build_subprocess_env(scrub_secrets=True)
    assert worker_env["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={runtime_dir}/bus"


@pytest.mark.linux_only
def test_absent_user_bus_is_never_fabricated(monkeypatch):
    """Fail closed, never invent. With no user manager the env must stay bare so the
    scope probe keeps reaching its honest "unavailable" verdict — a bus address pointing
    at nothing would turn a clear dispatch refusal into a confusing connect failure."""
    _fake_user_bus(monkeypatch, present=False)
    _service_manager_env(monkeypatch)

    gw._ensure_user_systemd_env()

    assert "XDG_RUNTIME_DIR" not in os.environ
    assert "DBUS_SESSION_BUS_ADDRESS" not in os.environ
