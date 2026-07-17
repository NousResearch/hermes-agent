"""Generated service behavior for the opt-in systemd watchdog."""

from __future__ import annotations

from hermes_cli import gateway as gateway_cli


def test_default_user_unit_keeps_simple_service_without_watchdog(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_SYSTEMD_WATCHDOG_SECONDS", raising=False)
    monkeypatch.setattr(gateway_cli, "read_raw_config", lambda: {})

    unit = gateway_cli.generate_systemd_unit(system=False)

    assert "Type=simple" in unit
    assert "Type=notify" not in unit
    assert "NotifyAccess=" not in unit
    assert "WatchdogSec=" not in unit


def test_positive_watchdog_config_generates_notify_unit(monkeypatch):
    monkeypatch.setattr(
        gateway_cli,
        "read_raw_config",
        lambda: {"gateway": {"systemd_watchdog_seconds": 120}},
    )

    unit = gateway_cli.generate_systemd_unit(system=False)

    assert "Type=notify" in unit
    assert "Type=simple" not in unit
    assert "NotifyAccess=main" in unit
    assert "WatchdogSec=120s" in unit


def test_positive_watchdog_config_generates_notify_system_unit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gateway_cli,
        "read_raw_config",
        lambda: {"gateway": {"systemd_watchdog_seconds": 30}},
    )

    monkeypatch.setattr(
        gateway_cli,
        "_system_service_identity",
        lambda _user: ("hermes", "hermes", str(tmp_path)),
    )
    unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="hermes")

    assert "Type=notify" in unit
    assert "NotifyAccess=main" in unit
    assert "WatchdogSec=30s" in unit


def test_environment_does_not_override_watchdog_config(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_SYSTEMD_WATCHDOG_SECONDS", "45")
    monkeypatch.setattr(
        gateway_cli,
        "read_raw_config",
        lambda: {"gateway": {"systemd_watchdog_seconds": 120}},
    )

    unit = gateway_cli.generate_systemd_unit(system=False)

    assert "WatchdogSec=120s" in unit
    assert "WatchdogSec=45s" not in unit
