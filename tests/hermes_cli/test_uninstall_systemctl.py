"""Regression tests for uninstall_gateway_service() subprocess safety.

Verify that every systemctl subprocess.run call in
``hermes_cli.uninstall.uninstall_gateway_service()`` receives
``timeout=30`` and ``stdin=subprocess.DEVNULL`` — the follow-up
hardening from PR #64535 against hung/stalled subprocesses.
"""

import subprocess
import sys
import pytest


@pytest.fixture(autouse=True)
def _linux_platform(monkeypatch):
    """Force platform.system() == "Linux" so the systemd branch is taken."""
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setenv("PREFIX", "")
    monkeypatch.delenv("TERMUX_VERSION", raising=False)


def _mock_systemctl_calls(monkeypatch):
    """Patch subprocess.run and return a recording list of (args, kwargs)."""
    calls = []

    def recorder(cmd, **kwargs):
        calls.append({"args": list(cmd), "kwargs": kwargs})
        ns = type("Completed", (), {"returncode": 0, "stdout": b"", "stderr": b""})
        return ns

    monkeypatch.setattr("subprocess.run", recorder, raising=True)
    return calls


def _stub_gateway_imports(monkeypatch):
    """Provide no-op stand-ins for the gateway helper imports."""

    class DummyPath:
        def exists(self):
            return True
        def unlink(self):
            pass

    dummy_unit = DummyPath()

    def get_systemd_unit_path(system=False):
        return dummy_unit

    def get_service_name():
        return "hermes-gateway"

    def _systemctl_cmd(is_system=False):
        if is_system:
            return ["sudo", "systemctl"]
        return ["systemctl"]

    monkeypatch.setattr("hermes_cli.gateway.get_systemd_unit_path", get_systemd_unit_path)
    monkeypatch.setattr("hermes_cli.gateway.get_service_name", get_service_name)
    monkeypatch.setattr("hermes_cli.gateway._systemctl_cmd", _systemctl_cmd)


class TestUninstallGatewaySystemctlTimeout:
    """Every systemctl call in uninstall_gateway_service must carry the
    timeout and stdin=DEVNULL guard."""

    @pytest.mark.skipif(sys.platform == "win32", reason="systemd is Linux-only")
    def test_user_service_stop(self, monkeypatch):
        calls = _mock_systemctl_calls(monkeypatch)
        _stub_gateway_imports(monkeypatch)
        monkeypatch.setattr("os.geteuid", lambda: 1000)
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])

        from hermes_cli.uninstall import uninstall_gateway_service
        uninstall_gateway_service()

        stop_call = next(c for c in calls if "stop" in c["args"])
        assert stop_call["kwargs"].get("timeout") == 30
        assert stop_call["kwargs"].get("stdin") is subprocess.DEVNULL

    @pytest.mark.skipif(sys.platform == "win32", reason="systemd is Linux-only")
    def test_user_service_disable(self, monkeypatch):
        calls = _mock_systemctl_calls(monkeypatch)
        _stub_gateway_imports(monkeypatch)
        monkeypatch.setattr("os.geteuid", lambda: 1000)
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])

        from hermes_cli.uninstall import uninstall_gateway_service
        uninstall_gateway_service()

        disable_call = next(c for c in calls if "disable" in c["args"])
        assert disable_call["kwargs"].get("timeout") == 30
        assert disable_call["kwargs"].get("stdin") is subprocess.DEVNULL

    @pytest.mark.skipif(sys.platform == "win32", reason="systemd is Linux-only")
    def test_daemon_reload(self, monkeypatch):
        calls = _mock_systemctl_calls(monkeypatch)
        _stub_gateway_imports(monkeypatch)
        monkeypatch.setattr("os.geteuid", lambda: 1000)
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])

        from hermes_cli.uninstall import uninstall_gateway_service
        uninstall_gateway_service()

        reload_call = next(c for c in calls if "daemon-reload" in c["args"])
        assert reload_call["kwargs"].get("timeout") == 30
        assert reload_call["kwargs"].get("stdin") is subprocess.DEVNULL

    @pytest.mark.skipif(sys.platform == "win32", reason="systemd is Linux-only")
    def test_system_service_calls_also_guarded(self, monkeypatch):
        """When run as root, system-scoped systemctl calls must also carry
        the timeout and stdin guards."""
        calls = _mock_systemctl_calls(monkeypatch)
        _stub_gateway_imports(monkeypatch)
        monkeypatch.setattr("os.geteuid", lambda: 0)  # root
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])

        from hermes_cli.uninstall import uninstall_gateway_service
        uninstall_gateway_service()

        for c in calls:
            assert c["kwargs"].get("timeout") == 30, (
                f"systemd call {c['args']} missing timeout=30"
            )
            assert c["kwargs"].get("stdin") is subprocess.DEVNULL, (
                f"systemd call {c['args']} missing stdin=subprocess.DEVNULL"
            )
