"""Tests for the desktop-backend gateway recovery routine (issue #83683).

``web_server._ensure_desktop_gateway_running`` relaunches a missing *supervised*
messaging gateway on desktop (re)start so WeChat/QQ/Telegram don't go silently
offline.  It must: no-op when a gateway is already running, relaunch via the
platform-native path when a supervisor is installed but the gateway is down,
skip when no supervisor is installed, honour the opt-out env var / config flag,
clear stale planned-stop markers, and never raise.
"""

import os

import pytest

import hermes_cli.config as config_mod
import hermes_cli.gateway as gateway_mod
import hermes_cli.gateway_windows as gw_win
from gateway import status as gateway_status
from hermes_cli import web_server


class _FakePath:
    def __init__(self, exists: bool):
        self._exists = exists

    def exists(self) -> bool:
        return self._exists


@pytest.fixture
def recovery_mocks(monkeypatch):
    calls = []

    def _record(name):
        calls.append(name)

    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    monkeypatch.setattr(gateway_mod, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway_mod, "is_macos", lambda: False)
    monkeypatch.setattr(gateway_mod, "is_windows", lambda: False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **k: [])
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda: None)
    monkeypatch.setattr(gateway_status, "clear_planned_stop_marker", lambda: None)
    monkeypatch.setattr(
        gateway_mod, "get_systemd_unit_path", lambda system=False: _FakePath(False)
    )
    monkeypatch.setattr(gateway_mod, "get_launchd_plist_path", lambda: _FakePath(False))
    monkeypatch.setattr(gateway_mod, "systemd_start", lambda *a, **k: _record("systemd_start"))
    monkeypatch.setattr(gateway_mod, "launchd_start", lambda *a, **k: _record("launchd_start"))
    monkeypatch.setattr(gw_win, "is_installed", lambda: False)
    monkeypatch.setattr(gw_win, "start", lambda: _record("windows_start"))
    return calls


class TestEnsureDesktopGatewayRunning:
    def test_no_relaunch_when_already_running(self, monkeypatch, recovery_mocks):
        monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **k: [123])
        web_server._ensure_desktop_gateway_running()
        assert recovery_mocks == []

    def test_relaunch_when_windows_scheduled_task_installed(self, monkeypatch, recovery_mocks):
        monkeypatch.setattr(gateway_mod, "is_windows", lambda: True)
        monkeypatch.setattr(gw_win, "is_installed", lambda: True)
        web_server._ensure_desktop_gateway_running()
        assert recovery_mocks == ["windows_start"]

    def test_relaunch_when_systemd_unit_installed(self, monkeypatch, recovery_mocks):
        monkeypatch.setattr(gateway_mod, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(
            gateway_mod, "get_systemd_unit_path", lambda system=False: _FakePath(True)
        )
        web_server._ensure_desktop_gateway_running()
        assert recovery_mocks == ["systemd_start"]

    def test_relaunch_when_launchd_installed(self, monkeypatch, recovery_mocks):
        monkeypatch.setattr(gateway_mod, "is_macos", lambda: True)
        monkeypatch.setattr(gateway_mod, "get_launchd_plist_path", lambda: _FakePath(True))
        web_server._ensure_desktop_gateway_running()
        assert recovery_mocks == ["launchd_start"]

    def test_no_relaunch_when_no_supervisor(self, monkeypatch, recovery_mocks):
        # default fixtures: not running, no supervisor installed
        web_server._ensure_desktop_gateway_running()
        assert recovery_mocks == []

    def test_env_var_disables_relaunch(self, monkeypatch, recovery_mocks):
        monkeypatch.setenv("HERMES_DESKTOP_NO_GATEWAY_RELAUNCH", "1")
        monkeypatch.setattr(gateway_mod, "is_windows", lambda: True)
        monkeypatch.setattr(gw_win, "is_installed", lambda: True)
        web_server._ensure_desktop_gateway_running()
        assert recovery_mocks == []

    def test_config_false_disables_relaunch(self, monkeypatch, recovery_mocks):
        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda: {"gateway": {"relaunch_gateway_on_desktop_start": False}},
        )
        monkeypatch.setattr(gateway_mod, "is_windows", lambda: True)
        monkeypatch.setattr(gw_win, "is_installed", lambda: True)
        web_server._ensure_desktop_gateway_running()
        assert recovery_mocks == []

    def test_recovery_never_raises(self, monkeypatch, recovery_mocks):
        # Even if every helper throws, the function must swallow it.
        def _boom(**k):
            raise RuntimeError("boom")

        monkeypatch.setattr(gateway_mod, "find_gateway_pids", _boom)
        # Should not raise.
        web_server._ensure_desktop_gateway_running()
