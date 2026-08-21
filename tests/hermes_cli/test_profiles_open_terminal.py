"""Tests for the profile open-terminal endpoint (macOS osascript path).

Regression: the darwin branch previously fire-and-forgot ``osascript`` via
``subprocess.Popen`` with no returncode / timeout handling — when Terminal
could not be activated (missing Automation permission, osascript error), the
API still returned ``{"ok": true}`` while nothing opened on the user's screen.
"""

import subprocess
import sys
from unittest.mock import Mock

import pytest

from hermes_cli.web_routers import profiles as profiles_router


def _invoke_endpoint(monkeypatch, *, subprocess_result=None, exc=None):
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
    monkeypatch.setattr(
        profiles_router, "_profile_setup_command", lambda name: command
    )
    # Async endpoint — drive it synchronously via asyncio.
    import asyncio
    return asyncio.run(
        profiles_router.open_profile_terminal_endpoint("default")
    ), calls


def test_darwin_osascript_success_returns_ok(monkeypatch):
    result = subprocess.CompletedProcess(["osascript"], 0, stdout="ok", stderr="")
    response, calls = _invoke_endpoint(monkeypatch, subprocess_result=result)
    assert response == {"ok": True, "command": "hermes setup"}
    # Must be a blocking run (not fire-and-forget Popen) with a timeout.
    assert "timeout" in calls["kwargs"]
    assert calls["kwargs"]["timeout"] > 0
    assert "capture_output" in calls["kwargs"]


def test_darwin_osascript_failure_raises_500(monkeypatch):
    result = subprocess.CompletedProcess(
        ["osascript"], 1, stdout="", stderr="execution error: Not authorized"
    )
    with pytest.raises(Exception) as excinfo:
        _invoke_endpoint(monkeypatch, subprocess_result=result)
    from fastapi import HTTPException
    assert isinstance(excinfo.value, HTTPException)
    assert excinfo.value.status_code == 500
    assert "Automation" in excinfo.value.detail


def test_darwin_osascript_timeout_raises_504(monkeypatch):
    with pytest.raises(Exception) as excinfo:
        _invoke_endpoint(monkeypatch, exc=subprocess.TimeoutExpired("osascript", 15))
    from fastapi import HTTPException
    assert isinstance(excinfo.value, HTTPException)
    assert excinfo.value.status_code == 504


def test_darwin_osascript_missing_raises_500(monkeypatch):
    with pytest.raises(Exception) as excinfo:
        _invoke_endpoint(monkeypatch, exc=FileNotFoundError("osascript"))
    from fastapi import HTTPException
    assert isinstance(excinfo.value, HTTPException)
    assert excinfo.value.status_code == 500
