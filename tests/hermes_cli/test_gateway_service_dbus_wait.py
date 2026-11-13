"""``_wait_for_user_dbus_socket``: readiness found on poll_until's final check still exports the env."""

import pytest

pytest.importorskip("pwd")  # POSIX systemd helper, same gate as test_gateway_service

import hermes_cli.gateway as gateway_cli


def test_ready_only_on_final_check_exports_env_once(monkeypatch):
    calls, slept = [], []
    checks = iter([False, True])
    # Deadline at 1.0. After the first miss the clock reads 0.9, so the last sleep is cut to 0.1
    # and the second check is the final one, made at the deadline.
    clock = iter([0.0, 0.9, 1.0])
    monkeypatch.setattr(gateway_cli, "_ensure_user_systemd_env", lambda: calls.append("env"))
    monkeypatch.setattr(gateway_cli, "_user_systemd_socket_ready", lambda: next(checks))
    monkeypatch.setattr("agent.deadline.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("agent.deadline.time.sleep", slept.append)

    assert gateway_cli._wait_for_user_dbus_socket(timeout=1.0) is True
    assert slept == [pytest.approx(0.1)]
    assert calls == ["env"]
