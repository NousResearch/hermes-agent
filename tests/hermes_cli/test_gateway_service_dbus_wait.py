"""``_wait_for_user_dbus_socket``: readiness found on poll_until's final check still exports the env."""

import pytest

pytest.importorskip("pwd")  # POSIX systemd helper, same gate as test_gateway_service

import hermes_cli.gateway as gateway_cli


def test_ready_only_on_final_check_exports_env_once(monkeypatch):
    calls = []
    checks = iter([False, False, True])
    clock = iter([0.0, 0.0, 0.5, 1.5])  # deadline at 1.0: the third check runs after it
    monkeypatch.setattr(gateway_cli, "_ensure_user_systemd_env", lambda: calls.append("env"))
    monkeypatch.setattr(gateway_cli, "_user_systemd_socket_ready", lambda: next(checks))
    monkeypatch.setattr("agent.deadline.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("agent.deadline.time.sleep", lambda _s: None)

    assert gateway_cli._wait_for_user_dbus_socket(timeout=1.0) is True
    assert calls == ["env"]
