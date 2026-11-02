"""A stop or an update ends the Windows launcher before it ends the gateway.

The launcher restarts any exit but 0 and 78 after a minute. A socket pause drains to 75 and a force-kill is
nonzero, so a launcher left running relaunched the gateway in the middle of the update that paused it.
"""
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli import gateway_windows, update_cmd_windows
from hermes_cli import main as hermes_main

HOME = r"C:\Users\Ünï\AppData\Local\hermes"


class _Process:
    def __init__(self, name, *args, refuse=False):
        self.info = {"name": name, "cmdline": [name, *args]}
        self.refuse = refuse
        self.killed = False

    def kill(self):
        if self.refuse:
            raise psutil.AccessDenied()
        self.killed = True


def _table(monkeypatch, *processes):
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: list(processes))
    return processes


@pytest.mark.parametrize("args", [
    ["//B", "//Nologo", HOME + r"\gateway-service\Hermes_Gateway.vbs"],  # the task's action
    [HOME + r"\gateway-service\Hermes_Gateway.vbs"],  # the Startup entry's chain
    [HOME.upper() + r"\GATEWAY-SERVICE\HERMES_GATEWAY.VBS"],  # Windows paths compare without case
])
def test_ends_the_host_of_this_homes_launcher(monkeypatch, args):
    host, = _table(monkeypatch, _Process("wscript.exe", *args))

    assert gateway_windows.end_launchers([HOME]) == 1
    assert host.killed


@pytest.mark.parametrize("process", [
    _Process("wscript.exe", "//B", "//Nologo", r"C:\Other\hermes\gateway-service\Hermes_Gateway.vbs"),  # another home
    _Process("wscript.exe", HOME + r"\gateway-service-old\Hermes_Gateway.vbs"),  # a prefix, not the folder
    _Process("wscript.exe", HOME + r"\gateway-service\nested\Hermes_Gateway.vbs"),
    _Process("wscript.exe", r"C:\Tools\backup.vbs", HOME + r"\gateway-service\notes.txt"),
    _Process("notepad.exe", HOME + r"\gateway-service\Hermes_Gateway.vbs"),  # someone reading it
    _Process("python.exe", "-m", "hermes_cli.main", "gateway", "run"),
])
def test_leaves_every_other_process_alone(monkeypatch, process):
    process.killed = False
    _table(monkeypatch, process)

    assert gateway_windows.end_launchers([HOME]) == 0
    assert not process.killed


def test_a_refused_kill_is_not_counted_and_the_rest_still_end(monkeypatch):
    script = HOME + r"\gateway-service\Hermes_Gateway.vbs"
    refused, ended = _table(monkeypatch, _Process("wscript.exe", script, refuse=True), _Process("wscript.exe", script))

    assert gateway_windows.end_launchers([HOME]) == 1
    assert ended.killed and not refused.killed


def _arrange_stop(monkeypatch, *, pid, ended, order):
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "_clear_start_attestation", lambda: None)
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda *a, **k: [pid] if pid else [])
    monkeypatch.setattr(gateway_windows, "_windows_stop_drain_timeout", lambda: 1.0)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
    monkeypatch.setattr(gateway_windows, "end_launchers", lambda: order.append("launcher") or ended)
    monkeypatch.setattr(gateway_windows, "_drain_gateway_pid", lambda *a: order.append("drain") or True)
    monkeypatch.setattr(gateway_windows, "_force_terminate_known_gateway_pids", lambda identities: 0)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: pid)


def test_stop_ends_the_launcher_before_it_drains_the_gateway(monkeypatch):
    order = []
    _arrange_stop(monkeypatch, pid=4242, ended=1, order=order)

    gateway_windows.stop()

    assert order == ["launcher", "drain"]


def test_stop_during_the_restart_delay_is_a_stop_not_nothing_running(monkeypatch, capsys):
    """No gateway exists to find, and the launcher would have started one within the minute."""
    _arrange_stop(monkeypatch, pid=None, ended=1, order=[])

    gateway_windows.stop()

    out = capsys.readouterr().out
    assert "Gateway stopped" in out and "No gateway was running" not in out


@pytest.fixture
def pause_seams(monkeypatch):
    """The real pause producer and durable record; the enumeration and the stop are the seams."""
    order = []
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)
    monkeypatch.setattr(gateway_windows, "end_launchers", lambda homes=None: order.append(("launcher", list(homes))) or 1)
    monkeypatch.setattr(update_cmd_windows, "_stop_windows_gateways", lambda *a, **k: order.append(("stop", a[0])) or {})
    monkeypatch.setattr(update_cmd_windows, "_record_attested_cold_start_profiles", lambda *a: None)
    monkeypatch.setattr(update_cmd_windows, "_pause_windows_gateway_services", lambda service_gateways, token, *a: token)
    return order


def test_an_update_ends_the_launcher_before_it_asks_the_gateway_to_drain(pause_seams, monkeypatch):
    process = SimpleNamespace(pid=4242, profile="default", path=str(gateway_windows._hermes_home()), create_time=1.0)
    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", lambda: ({4242: process}, [], set(), [4242]))

    update_cmd_windows._pause_windows_gateways_for_update()

    assert [step for step, _ in pause_seams] == ["launcher", "stop"]
    assert gateway_windows._hermes_home() in pause_seams[0][1]


def test_an_update_ends_a_launcher_in_its_restart_delay(pause_seams, monkeypatch):
    """With no gateway running there is nothing to pause, and this one would wake mid-update."""
    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", lambda: ({}, [], set(), []))
    monkeypatch.setattr(update_cmd_windows, "_windows_cold_start_plan", lambda: None)

    update_cmd_windows._pause_windows_gateways_for_update()

    assert [step for step, _ in pause_seams] == ["launcher"]
