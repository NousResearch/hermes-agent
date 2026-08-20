"""Tests for cua-driver child-env policy in tools/computer_use/cua_backend.py.

Covers the telemetry opt-out injection and — the #64917 regression — the
Windows AppCompat cache bypass (``=::=::\\``) that keeps cua-driver from
tripping the ``ahcache.sys`` ``NtApphelpCacheControl`` kernel crash path.
"""

import sys

import pytest


def test_child_env_injects_telemetry_opt_out(monkeypatch):
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend, "_cua_telemetry_disabled", lambda: True)
    env = cua_backend.cua_driver_child_env({"PATH": "/usr/bin"})
    assert env["CUA_DRIVER_RS_TELEMETRY_ENABLED"] == "0"
    assert env["PATH"] == "/usr/bin"


def test_child_env_passes_base_env_through(monkeypatch):
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend, "_cua_telemetry_disabled", lambda: False)
    env = cua_backend.cua_driver_child_env({"FOO": "bar", "PATH": "/bin"})
    assert env["FOO"] == "bar"
    # Telemetry left untouched when opted in.
    assert "CUA_DRIVER_RS_TELEMETRY_ENABLED" not in env


def test_windows_injects_appcompat_cache_bypass(monkeypatch):
    """#64917 regression: on win32 the env must carry ``=::=::\\`` so the
    AppCompat shim cache lookup (NtApphelpCacheControl / ahcache.sys) is
    skipped for the cua-driver process tree — the call that traces to the
    reported BSOD."""
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend, "_cua_telemetry_disabled", lambda: True)
    monkeypatch.setattr(sys, "platform", "win32")
    env = cua_backend.cua_driver_child_env({"PATH": "C:\\Windows"})
    assert "=::=::\\" in env
    assert env["=::=::\\"] == ""
    assert env["CUA_DRIVER_RS_TELEMETRY_ENABLED"] == "0"


def test_non_windows_does_not_inject_appcompat_bypass(monkeypatch):
    """On non-Windows platforms the guard must NOT appear (it is meaningless
    and would just be dead noise in the child env)."""
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend, "_cua_telemetry_disabled", lambda: True)
    for plat in ("linux", "darwin"):
        monkeypatch.setattr(sys, "platform", plat)
        env = cua_backend.cua_driver_child_env({})
        assert "=::=::\\" not in env


def test_appcompat_bypass_not_overwritten(monkeypatch):
    """If a caller already set ``=::=::\\`` (defensive / explicit), our
    setdefault must not clobber it."""
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend, "_cua_telemetry_disabled", lambda: True)
    monkeypatch.setattr(sys, "platform", "win32")
    env = cua_backend.cua_driver_child_env({"=::=::\\": "preexisting"})
    assert env["=::=::\\"] == "preexisting"
