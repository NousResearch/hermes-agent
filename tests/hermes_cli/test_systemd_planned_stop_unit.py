"""Tests for planned-stop hooks in generated systemd units."""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.gateway as gateway_cli


def _assert_planned_stop_hook(unit: str) -> None:
    command = next(line for line in unit.splitlines() if line.startswith("ExecStop="))
    assert command.startswith("ExecStop=-")
    assert "hermes_systemd_planned_stop" in command
    assert "gateway.systemd_stop_mark" not in command
    # Keep the manager-provided PID outside generic argv escaping: quoting
    # through _systemd_command would turn it into the literal $$MAINPID.
    assert command.endswith(" $MAINPID")
    assert "$$MAINPID" not in command
    assert unit.index(command) < unit.index("ExecStopPost=")


def _runtime_owner(monkeypatch, managed_runtime: bool) -> None:
    monkeypatch.setattr(
        "hermes_cli._launchers.resolve_store_python",
        lambda repo_root, **kwargs: Path("/store/python") if managed_runtime else None,
    )
    monkeypatch.setattr(gateway_cli, "get_python_path", lambda: "/venv/bin/python")


@pytest.mark.parametrize("managed_runtime", [False, True])
def test_user_unit_marks_direct_systemd_stop_as_planned(monkeypatch, managed_runtime):
    _runtime_owner(monkeypatch, managed_runtime)
    _assert_planned_stop_hook(gateway_cli.generate_systemd_unit(system=False))


@pytest.mark.parametrize("managed_runtime", [False, True])
def test_system_unit_marks_direct_systemd_stop_as_planned(monkeypatch, managed_runtime):
    _runtime_owner(monkeypatch, managed_runtime)
    monkeypatch.setattr(
        gateway_cli,
        "_system_service_identity",
        lambda run_as_user=None: ("alice", "alice", "/home/alice", 1000),
    )
    _assert_planned_stop_hook(
        gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")
    )
