from __future__ import annotations

import types

from hermes_cli import exit_watchdog


def _capture_watchdog(monkeypatch, *, timeout_s=None, exit_code=0):
    captured = {}

    class ImmediateThread:
        def __init__(self, *, target, daemon, name):
            captured.update(target=target, daemon=daemon, name=name)

        def start(self):
            captured["started"] = True

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(exit_watchdog.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(exit_watchdog.time, "sleep", lambda seconds: captured.update(sleep=seconds))
    monkeypatch.setattr(exit_watchdog.logging, "shutdown", lambda: None)
    monkeypatch.setattr(exit_watchdog.os, "_exit", lambda code: captured.update(exit_code=code))
    exit_watchdog.arm_exit_watchdog(timeout_s=timeout_s, exit_code=exit_code)
    return captured


def test_watchdog_default_cli_exit_code_remains_zero(monkeypatch):
    captured = _capture_watchdog(monkeypatch, timeout_s=0.01)
    captured["target"]()
    assert captured["exit_code"] == 0


def test_watchdog_accepts_explicit_oneshot_exit_code(monkeypatch):
    captured = _capture_watchdog(monkeypatch, timeout_s=0.01, exit_code=70)
    captured["target"]()
    assert captured["exit_code"] == 70


def test_watchdog_invalid_timeout_uses_existing_default(monkeypatch):
    monkeypatch.setenv("HERMES_EXIT_WATCHDOG_S", "invalid")
    captured = _capture_watchdog(monkeypatch, timeout_s=None)
    captured["target"]()
    assert captured["sleep"] == 30.0


def test_watchdog_zero_timeout_disables_arming(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    started = []
    monkeypatch.setattr(
        exit_watchdog.threading,
        "Thread",
        lambda **_kwargs: types.SimpleNamespace(start=lambda: started.append(True)),
    )
    exit_watchdog.arm_exit_watchdog(timeout_s=0)
    assert started == []


def test_watchdog_is_disabled_under_pytest(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "active")
    started = []
    monkeypatch.setattr(
        exit_watchdog.threading,
        "Thread",
        lambda **_kwargs: types.SimpleNamespace(start=lambda: started.append(True)),
    )
    exit_watchdog.arm_exit_watchdog(timeout_s=0.01)
    assert started == []
