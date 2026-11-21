"""Diagnostic readers distinguish host service support from active supervision."""

from __future__ import annotations

import pytest

from hermes_cli import dump, gateway, status


@pytest.mark.parametrize(
    "manager,scope,installed,supervised,pids,expected_mode,expected_state",
    [
        ("launchd", "launchd", False, False, (24680,), "manual", "running"),
        ("launchd", "launchd", True, False, (24680,), "manual", "running"),
        ("launchd", "launchd", True, True, (24680,), "launchd", "running"),
        ("launchd", "launchd", True, True, (), "launchd", "running"),
        ("launchd", "launchd", False, True, (24680,), "launchd", "running"),
        ("launchd", "launchd", True, False, (), "launchd", "stopped"),
        ("launchd", "launchd", False, False, (), "launchd", "stopped"),
        ("systemd (user)", "user", False, False, (24680,), "manual", "running"),
        ("systemd (system)", "system", False, False, (24680,), "manual", "running"),
        ("systemd (user)", "user", True, True, (24680,), "systemd (user)", "running"),
        (
            "s6 (container supervisor)",
            "s6",
            True,
            True,
            (),
            "s6 (container supervisor)",
            "running",
        ),
        (
            "s6 (container supervisor)",
            "s6",
            False,
            False,
            (24680,),
            "manual",
            "running",
        ),
        ("launchd", None, True, False, (24680,), "manual", "running"),
        ("manual process", None, False, False, (24680,), "manual process", "running"),
        ("manual process", None, False, False, (), "manual process", "stopped"),
        (
            "Termux / manual process",
            None,
            False,
            False,
            (24680,),
            "Termux / manual process",
            "running",
        ),
        (
            "docker (foreground)",
            None,
            False,
            False,
            (24680,),
            "docker (foreground)",
            "running",
        ),
    ],
)
def test_diagnostic_readers_label_observed_supervision(
    monkeypatch,
    capsys,
    manager,
    scope,
    installed,
    supervised,
    pids,
    expected_mode,
    expected_state,
):
    # These are the snapshot API's data states, not simulated host operating systems.
    snapshot = gateway.GatewayRuntimeSnapshot(
        manager, installed, supervised, pids, scope
    )
    monkeypatch.setattr(gateway, "get_gateway_runtime_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        gateway, "named_profile_served_by_running_multiplexer", lambda: False
    )
    monkeypatch.setattr(
        "hermes_cli.gateway_multiplex_served.multiplexer_served_secondaries", list
    )

    pid_text = ", pid 24680" if pids else ""
    assert dump._gateway_status() == f"{expected_state} ({expected_mode}{pid_text})"

    status._render_gateway(None)
    output = capsys.readouterr().out
    manager_line = next(
        line for line in output.splitlines() if line.strip().startswith("Manager:")
    )
    assert manager_line.split(":", 1)[1].strip() == expected_mode
    status_line = next(
        line for line in output.splitlines() if line.strip().startswith("Status:")
    )
    assert status_line.rstrip().endswith(expected_state)
    if installed and pids and not supervised:
        assert "installed but not managing the current running gateway" in output
    elif installed and not supervised:
        assert "installed but stopped" in output


def test_served_profile_status_still_reports_its_live_multiplexer(monkeypatch, capsys):
    snapshot = gateway.GatewayRuntimeSnapshot(
        manager="launchd", service_scope="launchd"
    )
    monkeypatch.setattr(gateway, "get_gateway_runtime_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        gateway, "named_profile_served_by_running_multiplexer", lambda: True
    )

    status._render_gateway(None)

    output = capsys.readouterr().out
    assert "running (via the default-profile multiplexer)" in output
    assert "stopped" not in output
    assert "Manager:" not in output
