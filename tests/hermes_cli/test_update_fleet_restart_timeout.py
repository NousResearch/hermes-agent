"""Regression tests for isolated systemd fleet restarts (#68523)."""

import subprocess

from hermes_cli import main as hermes_main


def test_per_unit_timeout_does_not_skip_later_gateways(monkeypatch, capsys):
    units = [
        "hermes-gateway-one",
        "hermes-gateway-two",
        "hermes-gateway-three",
    ]
    calls = []

    def fake_run(command, **_kwargs):
        unit = command[-1]
        calls.append(unit)
        if unit == "hermes-gateway-two":
            raise subprocess.TimeoutExpired(command, timeout=15)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    failures = {}
    restarted = []

    for unit in units:
        result = hermes_main._run_update_systemctl(
            ["systemctl", "--user", "restart", unit],
            scope="user",
            unit=unit,
            timeout=15,
            failures=failures,
        )
        if result is not None:
            restarted.append(unit)

    assert calls == units
    assert restarted == ["hermes-gateway-one", "hermes-gateway-three"]
    assert list(failures) == [("user", "hermes-gateway-two")]
    assert "continuing with remaining gateways" in capsys.readouterr().out


def test_reconciliation_clears_completed_timeout_and_keeps_stale_pid(monkeypatch):
    unit = "hermes-gateway-shared"
    discovered = {
        ("user", unit): (["systemctl", "--user"], 101),
        ("system", unit): (["systemctl"], 201),
    }
    failures = {
        ("user", unit): "systemctl timed out",
        ("system", unit): "systemctl timed out",
    }

    def fake_run(command, **_kwargs):
        is_user = "--user" in command
        pid = 102 if is_user else 201
        stdout = f"ActiveState=active\nMainPID={pid}\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    hermes_main._reconcile_updated_systemd_gateways(discovered, failures)

    assert ("user", unit) not in failures
    assert failures[("system", unit)] == "still running pre-update PID 201"


def test_reconciliation_keeps_unverifiable_timeout_without_initial_pid(monkeypatch):
    key = ("user", "hermes-gateway-unknown")
    failures = {key: "systemctl timed out"}

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="ActiveState=active\nMainPID=999\n",
            stderr="",
        )

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    hermes_main._reconcile_updated_systemd_gateways(
        {key: (["systemctl", "--user"], 0)},
        failures,
    )

    assert failures[key] == "systemctl timed out"
