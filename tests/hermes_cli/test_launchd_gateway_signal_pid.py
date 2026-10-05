"""Signal only the gateway child, not launchd/JXA/stderr launchers."""
from types import SimpleNamespace

import pytest

from hermes_cli.gateway_launchd import launchd_gateway_signal_pid


class Process:
    def __init__(self, pid, argv, children=()):
        self.pid = pid
        self.argv = argv
        self.descendants = list(children)

    def cmdline(self):
        return self.argv

    def children(self, recursive=False):
        assert recursive
        return self.descendants


def topology(monkeypatch, runtimes=1):
    import psutil

    gateway = [Process(300 + i, ["python", "-m", "hermes_cli.main", "gateway", "run", "--external-supervisor"]) for i in range(runtimes)]
    timestamp = Process(200, ["python", "-m", "hermes_cli.stderr_timestamp", "--", "hermes", "gateway", "run"])
    launcher = Process(100, ["osascript", "-e", "JXA launcher"], [timestamp, *gateway])
    monkeypatch.setattr(psutil, "Process", lambda pid: launcher)
    return gateway


def test_signal_pid_skips_stderr_launcher(monkeypatch):
    topology(monkeypatch)
    assert launchd_gateway_signal_pid(100) == 300


def test_signal_pid_skips_bootstrap_stderr_launcher(monkeypatch):
    import psutil
    gateway = Process(300, ["hermes", "gateway", "run"])
    timestamp = Process(200, ["python", "-I", "-c", "import runpy; runpy.run_module('hermes_cli.main', run_name='__main__')", "--run-module", "hermes_cli.stderr_timestamp", "--", "hermes", "gateway", "run"])
    launcher = Process(100, ["osascript"], [timestamp, gateway])
    monkeypatch.setattr(psutil, "Process", lambda pid: launcher)
    assert launchd_gateway_signal_pid(100) == 300


@pytest.mark.parametrize("runtimes", [0, 2])
def test_signal_pid_fails_closed_without_unique_runtime(monkeypatch, runtimes):
    topology(monkeypatch, runtimes)
    assert launchd_gateway_signal_pid(100) is None


@pytest.mark.platforms("macos")
def test_service_pid_inventory_protects_gateway_child(monkeypatch):
    from hermes_cli import gateway
    topology(monkeypatch)
    monkeypatch.setattr(gateway, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gateway, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="pid = 100\n"))
    assert gateway._get_service_pids() == {100, 200, 300}


def test_bounded_stop_escalates_actual_child(monkeypatch):
    import psutil
    from hermes_cli.gateway_launchd import stop_launchd_gateway_runtime
    calls = []

    class Runtime(Process):
        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")

        def wait(self, timeout):
            calls.append(("wait", timeout))
            if timeout == 30:
                raise psutil.TimeoutExpired(timeout, self.pid)

    runtime = Runtime(300, ["hermes", "gateway", "run"])
    monkeypatch.setattr(psutil, "Process", lambda pid: runtime)
    assert stop_launchd_gateway_runtime(300)
    assert calls == ["terminate", ("wait", 30), "kill", ("wait", 5)]


def test_stop_refuses_unrelated_process(monkeypatch):
    import psutil
    from hermes_cli.gateway_launchd import stop_launchd_gateway_runtime
    monkeypatch.setattr(psutil, "Process", lambda pid: Process(pid, ["python", "unrelated.py"]))
    assert not stop_launchd_gateway_runtime(300)
