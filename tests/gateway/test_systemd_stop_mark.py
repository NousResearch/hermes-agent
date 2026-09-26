"""Supervisor-ordered stops exit cleanly instead of logging a failure.

Raw ``systemctl restart/stop`` sends SIGTERM with no marker, so the gateway
used to classify it as an unexpected kill (exit 1, ``Failed with result
exit-code``). The generated unit now marks the stop via ``ExecStop=`` first.
"""

from __future__ import annotations
from gateway import systemd_lifecycle
from gateway import systemd_unit_render

import pytest

from gateway import systemd_stop_mark


def test_stop_mark_writes_planned_stop_for_supervised_pid(monkeypatch):
    """MAINPID (or an explicit argv PID) is marked; the helper owns identity."""
    marked: list[int] = []
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: marked.append(pid) or True,
    )
    monkeypatch.setenv("MAINPID", "4242")
    assert systemd_stop_mark.main(["systemd_stop_mark"]) == 0
    assert marked == [4242]

    marked.clear()
    monkeypatch.delenv("MAINPID")
    assert systemd_stop_mark.main(["systemd_stop_mark", "7777"]) == 0
    assert marked == [7777]


def test_stop_mark_never_blocks_restart_and_unit_wires_exec_stop(monkeypatch):
    """Unknown PIDs mark nothing but still exit 0; the unit runs the marker."""
    marked: list[int] = []
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: marked.append(pid) or True,
    )
    for argv, env in ((["m"], None), (["m", "0"], None), (["m", "abc"], None)):
        if env is None:
            monkeypatch.delenv("MAINPID", raising=False)
        assert systemd_stop_mark.main(argv) == 0
    assert marked == []

    def _boom(_pid):
        raise OSError("disk gone")

    monkeypatch.setattr("gateway.status.write_planned_stop_marker", _boom)
    monkeypatch.setenv("MAINPID", "4242")
    assert systemd_stop_mark.main(["m"]) == 0

    from hermes_cli import gateway as gateway_cli

    unit = systemd_unit_render.generate_systemd_unit(system=False)
    assert "ExecStop=-" in unit and "-m gateway.systemd_stop_mark" in unit
    # The cgroup reaper still runs after the main process exits.
    assert "-m gateway.cgroup_cleanup" in unit
    assert unit.index("systemd_stop_mark") < unit.index("cgroup_cleanup")
