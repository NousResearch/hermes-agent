"""Tests for planned-stop hooks in generated systemd units."""

from __future__ import annotations

import hermes_cli.gateway as gateway_cli


_MARKER_COMMAND = (
    "ExecStop=-/venv/bin/python -m gateway.systemd_planned_stop $MAINPID"
)


def _assert_planned_stop_hook(unit: str) -> None:
    assert _MARKER_COMMAND in unit
    assert unit.index(_MARKER_COMMAND) < unit.index("ExecStopPost=")


def test_user_unit_marks_direct_systemd_stop_as_planned(monkeypatch):
    monkeypatch.setattr(gateway_cli, "get_python_path", lambda: "/venv/bin/python")

    _assert_planned_stop_hook(gateway_cli.generate_systemd_unit(system=False))


def test_system_unit_marks_direct_systemd_stop_as_planned(monkeypatch):
    monkeypatch.setattr(gateway_cli, "get_python_path", lambda: "/venv/bin/python")
    monkeypatch.setattr(
        gateway_cli,
        "_system_service_identity",
        lambda run_as_user=None: ("alice", "alice", "/home/alice"),
    )

    _assert_planned_stop_hook(
        gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")
    )
