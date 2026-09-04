"""Tests: `hermes gateway service` verbs + the updater's SCM restart branch.

Hermetic — pywin32/SCM calls stubbed; the CONTRACTS are what's tested:
dispatch, config persistence, the sealed+running gate for the updater's
Restart-Service path, and the fall-through to ordinary restarts on every
other machine shape.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import hermes_cli.gateway as gw


def _fake_scm(monkeypatch, *, installed=True, running=True, calls=None):
    """Stub the pywin32 SCM surface; record calls into `calls`."""
    import sys
    import types

    if calls is None:
        calls = []
    mod = types.ModuleType("win32service")
    mod.SERVICE_AUTO_START = 2
    mod.SERVICE_DEMAND_START = 3
    util_mod = types.ModuleType("win32serviceutil")
    util_mod.QueryServiceStatus = lambda name: (None, 4 if running else 1) if installed else (_ for _ in ()).throw(OSError("not installed"))
    util_mod.ChangeServiceConfig = lambda name, starttype: calls.append(("config", starttype))
    util_mod.StartService = lambda name: calls.append(("start",))
    util_mod.StopService = lambda name: calls.append(("stop",))
    util_mod.RestartService = lambda name: calls.append(("restart",))
    monkeypatch.setitem(sys.modules, "win32service", mod)
    monkeypatch.setitem(sys.modules, "win32serviceutil", util_mod)
    return calls


def test_service_status_not_installed(monkeypatch, capfd):
    monkeypatch.setattr(gw.sys, "platform", "win32")
    import sys as _sys

    _sys.modules.pop("win32service", None)
    _sys.modules.pop("win32serviceutil", None)

    import types

    mod = types.ModuleType("win32service")
    mod.SERVICE_AUTO_START = 2
    mod.SERVICE_DEMAND_START = 3
    util_mod = types.ModuleType("win32serviceutil")

    def boom(name):
        raise OSError("not installed")

    util_mod.QueryServiceStatus = boom
    monkeypatch.setitem(_sys.modules, "win32service", mod)
    monkeypatch.setitem(_sys.modules, "win32serviceutil", util_mod)

    gw._windows_scm_service_command(None)
    out = capfd.readouterr().out
    assert "not installed" in out


def test_service_on_configures_starts_persists(monkeypatch, capfd):
    monkeypatch.setattr(gw.sys, "platform", "win32")
    calls = _fake_scm(monkeypatch, installed=True, running=True)
    saved = {}
    monkeypatch.setattr("cli.save_config_value", lambda k, v: saved.update({k: v}))

    gw._windows_scm_service_command("on")
    assert ("config", 2) in calls  # SERVICE_AUTO_START
    assert ("start",) in calls
    assert saved == {"gateway.service": True}
    assert "automatic at logon" in capfd.readouterr().out


def test_service_off_demand_stops_persists(monkeypatch, capfd):
    monkeypatch.setattr(gw.sys, "platform", "win32")
    calls = _fake_scm(monkeypatch, installed=True, running=True)
    saved = {}
    monkeypatch.setattr("cli.save_config_value", lambda k, v: saved.update({k: v}))

    gw._windows_scm_service_command("off")
    assert ("config", 3) in calls  # SERVICE_DEMAND_START
    assert ("stop",) in calls
    assert saved == {"gateway.service": False}


def test_service_unknown_action_is_an_error(monkeypatch, capfd):
    monkeypatch.setattr(gw.sys, "platform", "win32")
    calls = _fake_scm(monkeypatch, installed=True, running=True)
    monkeypatch.setattr("cli.save_config_value", lambda k, v: None)
    gw._windows_scm_service_command("dance")
    assert "Unknown service action" in capfd.readouterr().out


# ── the updater's SCM restart branch ─────────────────────────────────


def test_updater_scm_restart_gated_on_running_service(monkeypatch):
    """Sealed + HermesGateway RUNNING → Restart-Service, True (skip the
    detached watcher); any other shape → False (ordinary paths)."""
    monkeypatch.setattr(gw.sys, "platform", "win32")
    calls = _fake_scm(monkeypatch, installed=True, running=True)

    import importlib
    import sys

    sys.modules.pop("pm.ensure", None)
    pm_ensure = importlib.import_module("pm.ensure")

    monkeypatch.setattr(pm_ensure, "sealed", lambda: True)

    assert gw._try_scm_service_restart() is True
    assert ("restart",) in calls


def test_updater_scm_restart_false_when_not_sealed(monkeypatch):
    monkeypatch.setattr(gw.sys, "platform", "win32")
    calls = _fake_scm(monkeypatch, installed=True, running=True)
    import importlib

    pm_ensure = importlib.import_module("pm.ensure")
    monkeypatch.setattr(pm_ensure, "sealed", lambda: False)
    assert gw._try_scm_service_restart() is False
    assert calls == []  # no SCM touch


def test_updater_scm_restart_false_when_service_stopped(monkeypatch):
    monkeypatch.setattr(gw.sys, "platform", "win32")
    calls = _fake_scm(monkeypatch, installed=True, running=False)
    import importlib

    pm_ensure = importlib.import_module("pm.ensure")
    monkeypatch.setattr(pm_ensure, "sealed", lambda: True)
    assert gw._try_scm_service_restart() is False


def test_updater_scm_restart_false_on_posix(monkeypatch):
    monkeypatch.setattr(gw.sys, "platform", "linux")
    assert gw._try_scm_service_restart() is False
