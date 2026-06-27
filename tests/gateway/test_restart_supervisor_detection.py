"""Gateway restart supervisor detection."""

import gateway.run as gateway_run


def test_service_manager_detection_systemd(monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "systemd-unit-id")
    monkeypatch.setattr(gateway_run.sys, "platform", "linux")
    monkeypatch.setattr(gateway_run.os, "getppid", lambda: 1234)

    assert gateway_run._is_running_under_service_manager() is True


def test_service_manager_detection_launchd_on_macos(monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setattr(gateway_run.sys, "platform", "darwin")
    monkeypatch.setattr(gateway_run.os, "getppid", lambda: 1)

    assert gateway_run._is_running_under_service_manager() is True


def test_service_manager_detection_foreground_macos(monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setattr(gateway_run.sys, "platform", "darwin")
    monkeypatch.setattr(gateway_run.os, "getppid", lambda: 4321)

    assert gateway_run._is_running_under_service_manager() is False
