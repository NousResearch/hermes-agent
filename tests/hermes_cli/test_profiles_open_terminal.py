"""Tests for the profile open-terminal endpoint (macOS osascript path).

Regression: the darwin branch previously fire-and-forgot ``osascript`` via
``subprocess.Popen`` with no returncode / timeout handling — when Terminal
could not be activated (missing Automation permission, osascript error), the
API still returned ``{"ok": true}`` while nothing opened on the user's screen.
"""

import asyncio
import subprocess

import pytest
from fastapi import HTTPException

from hermes_cli.web_routers import profiles as profiles_router


def _invoke_endpoint(monkeypatch, *, subprocess_result=None, exc=None, returncode=0):
    """Call the open-terminal endpoint with a stubbed osascript backend."""
    command = "hermes setup"
    calls = {}

    def fake_run(argv, **kwargs):
        calls["argv"] = argv
        calls["kwargs"] = kwargs
        if exc is not None:
            raise exc
        return subprocess_result

    monkeypatch.setattr(profiles_router.sys, "platform", "darwin")
    monkeypatch.setattr(profiles_router.subprocess, "run", fake_run)
    monkeypatch.setattr(profiles_router, "_profile_setup_command", lambda name: command)
    # Async endpoint — drive it synchronously.
    return asyncio.run(profiles_router.open_profile_terminal_endpoint("default")), calls


def test_darwin_osascript_success_returns_ok(monkeypatch):
    result = subprocess.CompletedProcess(["osascript"], 0, stdout="ok", stderr="")
    response, calls = _invoke_endpoint(monkeypatch, subprocess_result=result)
    assert response == {"ok": True, "command": "hermes setup"}
    # Must be a blocking run (not fire-and-forget Popen) with a timeout.
    assert calls["kwargs"]["capture_output"] is True
    assert calls["kwargs"]["timeout"] > 0
    # A successful activation is distinguishable from a silent no-op.
    assert 'return "ok"' in calls["argv"][2]


def test_darwin_osascript_failure_raises_500(monkeypatch, caplog):
    result = subprocess.CompletedProcess(
        ["osascript"], 1, stdout="", stderr="execution error: Not authorized"
    )
    with caplog.at_level("WARNING", logger="hermes_cli.web_server"):
        with pytest.raises(HTTPException) as excinfo:
            _invoke_endpoint(monkeypatch, subprocess_result=result)
    assert excinfo.value.status_code == 500
    assert "Automation" in excinfo.value.detail
    # The osascript stderr must reach the log for diagnosability.
    assert "Not authorized" in caplog.text


def test_darwin_osascript_timeout_raises_504(monkeypatch):
    with pytest.raises(HTTPException) as excinfo:
        _invoke_endpoint(monkeypatch, exc=subprocess.TimeoutExpired("osascript", 15))
    assert excinfo.value.status_code == 504
    assert "Automation" in excinfo.value.detail


def test_darwin_osascript_missing_raises_500(monkeypatch):
    with pytest.raises(HTTPException) as excinfo:
        _invoke_endpoint(monkeypatch, exc=FileNotFoundError("osascript"))
    assert excinfo.value.status_code == 500
    assert "osascript" in excinfo.value.detail


def test_darwin_osascript_never_fire_and_forget(monkeypatch):
    """Guard against the endpoint regressing to ``subprocess.Popen``.

    Behavior test: if the darwin branch accidentally calls ``subprocess.Popen``,
    the stub raises RuntimeError and the test fails — regardless of source text.
    """
    called_popen = []

    def fake_popen(*args, **kwargs):
        called_popen.append((args, kwargs))
        raise RuntimeError("Popen must not be called in the darwin path")

    monkeypatch.setattr(profiles_router.sys, "platform", "darwin")
    monkeypatch.setattr(profiles_router.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        profiles_router, "_profile_setup_command", lambda name: "hermes setup"
    )
    # A normal successful run should complete without touching Popen.
    result = subprocess.CompletedProcess(["osascript"], 0, stdout="ok", stderr="")
    monkeypatch.setattr(profiles_router.subprocess, "run", lambda *a, **k: result)
    try:
        asyncio.run(profiles_router.open_profile_terminal_endpoint("default"))
    except Exception:
        pass  # We only care about whether Popen was called
    assert called_popen == [], (
        f"darwin path called subprocess.Popen {len(called_popen)} time(s); "
        "should use subprocess.run for synchronous osascript execution"
    )
