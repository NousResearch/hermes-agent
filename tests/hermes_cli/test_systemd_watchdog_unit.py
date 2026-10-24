"""Generated service behavior for the opt-in systemd watchdog."""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig
from hermes_cli import gateway as gateway_cli


@pytest.fixture(autouse=True)
def _isolated_python_resolution(monkeypatch):
    """Keep unit generation off the real Hermes home.

    ``generate_systemd_unit`` resolves the store python by reading
    ``<hermes home>/manifest.json``. Without a sandboxed HERMES_HOME that is the
    production home, which the test suite's home-io guard refuses. The symbol is
    imported INSIDE the generator, so patch its definition module.
    """
    import hermes_cli._launchers as launchers

    monkeypatch.setattr(
        launchers, "resolve_store_python", lambda *a, **k: "/usr/bin/python3", raising=False)


def test_positive_watchdog_config_generates_notify_unit(monkeypatch):
    monkeypatch.setattr(
        gateway_cli,
        "load_gateway_config",
        lambda: GatewayConfig.from_dict({"systemd_watchdog_seconds": 120}),
        raising=False,
    )

    unit = gateway_cli.generate_systemd_unit(system=False)

    assert "Type=notify" in unit
    assert "NotifyAccess=main" in unit
    assert "WatchdogSec=120s" in unit


def test_notify_unit_carries_a_generous_timeout_start_sec(monkeypatch):
    """Type=notify waits for READY=1 under TimeoutStartSec (systemd user-unit default 90s).

    This gateway's real startup budget is larger (startup watchdog allows 300s; MCP
    discovery alone waits 120s -- observed 2026-10-23: ready at +300s+). Without a
    generous TimeoutStartSec, systemd kills the gateway mid-boot and restart-loops it.
    """
    monkeypatch.setattr(
        gateway_cli,
        "load_gateway_config",
        lambda: GatewayConfig.from_dict({"systemd_watchdog_seconds": 180}),
        raising=False,
    )

    unit = gateway_cli.generate_systemd_unit(system=False)

    assert "TimeoutStartSec=" in unit
    start_sec = int(
        unit.split("TimeoutStartSec=", 1)[1].split("s", 1)[0].splitlines()[0].strip()
    )
    assert start_sec >= 300, (
        f"TimeoutStartSec={start_sec}s is below the gateway's ~300s startup budget"
    )
    assert start_sec >= 2 * 180, "must exceed the watchdog interval by a wide margin"


def test_timeout_start_sec_honours_the_env_override(monkeypatch):
    """HERMES_SYSTEMD_TIMEOUT_START_SEC must actually reach the generated unit.

    A host with a small state.db does not need the multi-minute default boot window, and
    an operator needs an escape hatch that does not require a code change.
    """
    monkeypatch.setattr(
        gateway_cli,
        "load_gateway_config",
        lambda: GatewayConfig.from_dict({"systemd_watchdog_seconds": 180}),
        raising=False,
    )
    monkeypatch.setenv("HERMES_SYSTEMD_TIMEOUT_START_SEC", "900")

    unit = gateway_cli.generate_systemd_unit(system=False)

    assert "TimeoutStartSec=900s" in unit


def test_timeout_start_sec_rejects_garbage(monkeypatch):
    """A bad override falls back to the default instead of emitting an invalid unit."""
    monkeypatch.setattr(
        gateway_cli,
        "load_gateway_config",
        lambda: GatewayConfig.from_dict({"systemd_watchdog_seconds": 180}),
        raising=False,
    )
    monkeypatch.setenv("HERMES_SYSTEMD_TIMEOUT_START_SEC", "not-a-number")

    unit = gateway_cli.generate_systemd_unit(system=False)

    assert f"TimeoutStartSec={gateway_cli._SYSTEMD_WATCHDOG_TIMEOUT_START_SEC}s" in unit






def test_system_unit_reads_watchdog_from_target_home(tmp_path, monkeypatch):
    caller_home = tmp_path / "caller"
    target_home = tmp_path / "target"
    caller_home.mkdir()
    target_home.mkdir()
    (caller_home / "config.yaml").write_text(
        "gateway:\n  systemd_watchdog_seconds: 0\n",
        encoding="utf-8",
    )
    (target_home / "config.yaml").write_text(
        "gateway:\n  systemd_watchdog_seconds: 75\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(caller_home))
    monkeypatch.setattr(
        gateway_cli,
        "_system_service_identity",
        lambda _user: ("service", "service", str(tmp_path / "account"), 1001),
    )
    monkeypatch.setattr(
        gateway_cli,
        "_hermes_home_for_target_user",
        lambda _home: str(target_home),
    )

    unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="service")

    assert "Type=notify" in unit
    assert "WatchdogSec=75s" in unit


