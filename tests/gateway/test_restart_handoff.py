"""Regression coverage for bounded autonomous gateway recovery restarts."""

import json
import signal

import pytest

import gateway.restart as restart


@pytest.mark.parametrize("command", ["hermes gateway restart", "  hermes gateway restart  "])
def test_recognizes_only_canonical_gateway_restart_command(command):
    assert restart.is_gateway_restart_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "systemctl --user restart hermes-gateway",
        "launchctl kickstart gui/501/ai.hermes.gateway",
        "hermes gateway restart --force",
        "hermes gateway 'restart'",
        "hermes gateway re\\start",
        "hermes gateway restart\n",
        "echo hermes gateway restart",
        "hermes gateway restart; rm -rf /",
        "pkill -f hermes.*gateway",
    ],
)
def test_rejects_raw_commands_and_source_text(command):
    assert not restart.is_gateway_restart_command(command)


def _enable_systemd_handoff(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(restart, "_is_cron_session", lambda: False)
    monkeypatch.setattr(restart.sys, "platform", "linux")
    monkeypatch.setattr(restart, "_is_current_supervisor", lambda: True)
    monkeypatch.setattr(restart, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(restart.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    return calls


def test_autonomous_handoff_records_origin_and_signals_gateway(monkeypatch, tmp_path):
    calls = _enable_systemd_handoff(monkeypatch, tmp_path)

    accepted, message = restart.request_gateway_recovery_restart(
        session_id="telegram:chat:42", task_id="turn-7"
    )

    assert accepted is True
    assert "accepted and scheduled" in message
    assert calls == [(restart.os.getpid(), signal.SIGUSR1)]
    state = json.loads((tmp_path / ".gateway_recovery_restart.json").read_text())
    assert state["reason"] == "agent_recovery"
    assert state["session_id"] == "telegram:chat:42"
    assert state["task_id"] == "turn-7"
    assert isinstance(state["attempted_at"], float)


def test_launchd_handoff_signals_gateway_process(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(restart, "_is_cron_session", lambda: False)
    monkeypatch.setattr(restart.sys, "platform", "darwin")
    monkeypatch.setattr(restart, "_is_current_supervisor", lambda: True)
    monkeypatch.setattr(restart, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(restart.os, "kill", lambda pid, sig: calls.append((pid, sig)))

    accepted, _message = restart.request_gateway_recovery_restart(
        session_id="session", task_id=None
    )

    assert accepted is True
    assert calls == [(restart.os.getpid(), signal.SIGUSR1)]


def test_cooldown_blocks_recursive_recovery_restart(monkeypatch, tmp_path):
    calls = _enable_systemd_handoff(monkeypatch, tmp_path)
    monkeypatch.setattr(restart.time, "time", lambda: 1000.0)

    assert restart.request_gateway_recovery_restart(session_id="session", task_id="turn")[0]
    accepted, message = restart.request_gateway_recovery_restart(session_id="session", task_id="resumed-turn")

    assert accepted is False
    assert "cooldown" in message
    assert len(calls) == 1


def test_different_session_is_not_blocked_by_prior_recovery(monkeypatch, tmp_path):
    calls = _enable_systemd_handoff(monkeypatch, tmp_path)
    monkeypatch.setattr(restart.time, "time", lambda: 1000.0)

    assert restart.request_gateway_recovery_restart(session_id="first", task_id=None)[0]
    assert restart.request_gateway_recovery_restart(session_id="second", task_id=None)[0]
    assert len(calls) == 2


def test_cron_session_cannot_schedule_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(restart, "_is_cron_session", lambda: True)
    monkeypatch.setattr(restart, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(restart.os, "kill", lambda *_args: pytest.fail("must not signal"))

    accepted, message = restart.request_gateway_recovery_restart(session_id="session", task_id=None)

    assert accepted is False
    assert "cron or unknown session context" in message


def test_manual_gateway_fails_closed_without_supervisor(monkeypatch, tmp_path):
    monkeypatch.setattr(restart, "_is_cron_session", lambda: False)
    monkeypatch.setattr(restart, "_is_current_supervisor", lambda: False)
    monkeypatch.setattr(restart, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(restart.os, "kill", lambda *_args: pytest.fail("must not signal"))

    accepted, message = restart.request_gateway_recovery_restart(session_id="session", task_id=None)

    assert accepted is False
    assert "active launchd or systemd service" in message


def test_signal_failure_is_reported_truthfully_without_consuming_cooldown(monkeypatch, tmp_path):
    _enable_systemd_handoff(monkeypatch, tmp_path)
    monkeypatch.setattr(restart.os, "kill", lambda *_args: (_ for _ in ()).throw(OSError("denied")))

    accepted, message = restart.request_gateway_recovery_restart(session_id="session", task_id=None)

    assert accepted is False
    assert "Could not deliver" in message

    calls = []
    monkeypatch.setattr(restart.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    accepted, _message = restart.request_gateway_recovery_restart(session_id="session", task_id=None)

    assert accepted is True
    assert calls == [(restart.os.getpid(), signal.SIGUSR1)]
