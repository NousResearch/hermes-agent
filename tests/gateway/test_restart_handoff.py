"""Regression coverage for approved gateway restart handoff."""

import signal

import pytest

import gateway.restart as restart


@pytest.mark.parametrize(
    "command",
    [
        "hermes gateway restart",
        "systemctl --user restart hermes-gateway",
        "launchctl kickstart gui/501/ai.hermes.gateway",
    ],
)
def test_recognizes_standalone_gateway_restart_commands(command):
    assert restart.is_gateway_restart_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "systemctl stop hermes-gateway",
        "pkill -f hermes.*gateway",
        "systemctl restart nginx",
        "systemctl restart hermes-gateway; rm -rf /",
        "systemctl restart hermes-gateway-backup",
        "launchctl kickstart gui/501/ai.hermes.gateway-backup",
    ],
)
def test_rejects_non_restart_or_compound_commands(command):
    assert not restart.is_gateway_restart_command(command)


def test_systemd_handoff_signals_gateway_process(monkeypatch):
    calls = []
    monkeypatch.setattr(restart, "_is_cron_session", lambda: False)
    monkeypatch.setattr(restart.sys, "platform", "linux")
    monkeypatch.setenv("INVOCATION_ID", "unit-invocation")
    monkeypatch.setattr(restart, "_is_current_supervisor", lambda: True)
    monkeypatch.setattr(restart.os, "kill", lambda pid, sig: calls.append((pid, sig)))

    accepted, message = restart.request_approved_gateway_restart_handoff()

    assert accepted is True
    assert "request delivered" in message
    assert calls == [(restart.os.getpid(), signal.SIGUSR1)]


def test_launchd_handoff_signals_gateway_process(monkeypatch):
    calls = []
    monkeypatch.setattr(restart, "_is_cron_session", lambda: False)
    monkeypatch.setattr(restart.sys, "platform", "darwin")
    monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway")
    monkeypatch.setattr(restart, "_is_current_supervisor", lambda: True)
    monkeypatch.setattr(restart.os, "kill", lambda pid, sig: calls.append((pid, sig)))

    accepted, _message = restart.request_approved_gateway_restart_handoff()

    assert accepted is True
    assert calls == [(restart.os.getpid(), signal.SIGUSR1)]


def test_manual_gateway_fails_closed_without_supervisor(monkeypatch):
    monkeypatch.setattr(restart, "_is_cron_session", lambda: False)
    monkeypatch.setattr(restart, "_is_current_supervisor", lambda: False)
    monkeypatch.setattr(restart.sys, "platform", "linux")
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setattr(restart.os, "kill", lambda *_args: pytest.fail("must not signal"))

    accepted, message = restart.request_approved_gateway_restart_handoff()

    assert accepted is False
    assert "active launchd or systemd service" in message


def test_cron_session_cannot_schedule_restart(monkeypatch):
    monkeypatch.setattr(restart, "_is_cron_session", lambda: True)
    monkeypatch.setattr(restart.os, "kill", lambda *_args: pytest.fail("must not signal"))

    accepted, message = restart.request_approved_gateway_restart_handoff()

    assert accepted is False
    assert "cron or unknown session context" in message
def test_unknown_session_context_cannot_schedule_restart(monkeypatch):
    monkeypatch.setattr(restart, "_is_cron_session", lambda: None)
    monkeypatch.setattr(restart.os, "kill", lambda *_args: pytest.fail("must not signal"))

    accepted, message = restart.request_approved_gateway_restart_handoff()

    assert accepted is False
    assert "cron or unknown session context" in message
