"""An s6 stop must reach the selected profile's planned-shutdown reader."""

import os
import subprocess

import pytest

from gateway import status
from hermes_cli.service_manager import S6ServiceManager


@pytest.mark.parametrize("caller,target", [("default", "worker"), ("worker", "default")])
def test_stop_marks_target_before_signal_without_overwriting_caller(
    tmp_path, monkeypatch, caller, target
):
    root = tmp_path / "hermes"
    homes = {"default": root, "worker": root / "profiles" / "worker"}
    for home in homes.values():
        home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(homes[caller]))
    assert status.write_planned_stop_marker(os.getpid() + 1)
    caller_marker = homes[caller] / ".gateway-planned-stop.json"
    original = caller_marker.read_bytes()
    scandir = tmp_path / "services"
    service = scandir / f"gateway-{target}"
    service.mkdir(parents=True)
    commands = []

    def s6_run(command, *args, **kwargs):
        commands.append(command)
        if command == "s6-svstat":
            return subprocess.CompletedProcess([], 0, f"up (pid {os.getpid()}) 10 seconds", "")
        assert command == "s6-svc"
        assert args == ("-d", str(service))
        # Simulate the target receiving SIGTERM, using the real marker reader.
        with monkeypatch.context() as target_scope:
            target_scope.setenv("HERMES_HOME", str(homes[target]))
            assert status.consume_planned_stop_marker_for_self()
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr("hermes_cli.service_manager._s6_run", s6_run)
    S6ServiceManager(scandir=scandir).stop(service.name)

    assert commands == ["s6-svstat", "s6-svc"]
    assert caller_marker.read_bytes() == original
