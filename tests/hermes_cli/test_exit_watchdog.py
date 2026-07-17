from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

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


def test_watchdog_start_failure_propagates_and_oneshot_owner_can_retry(monkeypatch):
    from hermes_cli.oneshot import _OneshotResources

    starts = []

    class FailingThenStartingThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            starts.append(self.kwargs)
            if len(starts) == 1:
                raise RuntimeError("thread start failed")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(exit_watchdog.threading, "Thread", FailingThenStartingThread)
    resources = _OneshotResources()

    with pytest.raises(RuntimeError, match="thread start failed"):
        resources._arm_watchdog_once()
    resources._arm_watchdog_once()
    resources._arm_watchdog_once()

    assert len(starts) == 2


def test_watchdog_timer_bypasses_blocking_logging_handler_and_exits_bounded():
    script = textwrap.dedent(
        """
        import logging
        import os
        import threading

        os.environ.pop("PYTEST_CURRENT_TEST", None)
        from hermes_cli.exit_watchdog import arm_exit_watchdog

        class BlockingHandler(logging.Handler):
            def emit(self, record):
                threading.Event().wait()

        root = logging.getLogger()
        root.handlers[:] = [BlockingHandler()]
        root.setLevel(logging.DEBUG)
        arm_exit_watchdog(timeout_s=0.2, exit_code=70)
        threading.Event().wait()
        """
    )
    env = os.environ.copy()
    env.pop("PYTEST_CURRENT_TEST", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        timeout=3,
    )

    assert completed.returncode == 70
    assert completed.stdout == b""
    assert completed.stderr == b""
