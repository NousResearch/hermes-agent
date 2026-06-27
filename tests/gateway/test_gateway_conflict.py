"""Tests for the desktop/foreign gateway conflict handling (issue #168).

Covers:
- ``gateway.status``: structured ``error_detail`` round-trip, the top-level
  ``gateway_conflict`` marker, and ``classify_port_conflict`` owner resolution.
- ``feishu._classify_address_in_use``: EADDRINUSE detection through a wrapped
  exception chain (incl. the Windows WSAEADDRINUSE code).
- ``hermes_cli.gateway`` restart path: a desktop-managed restart that finds a
  foreign Windows gateway service declines (records a conflict) unless an
  explicit force-takeover is requested; non-desktop restarts are unchanged.
"""

import errno
import os
import sys
import types
from unittest import mock

import pytest


@pytest.fixture()
def runtime_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_RUNTIME_DIR", str(tmp_path))
    # Reimport-safe: status reads the env each call, so just clear caches via env.
    return tmp_path


# --------------------------------------------------------------------------- #
# gateway.status
# --------------------------------------------------------------------------- #
def test_error_detail_round_trip_and_clear(runtime_dir):
    import gateway.status as gs

    gs.write_runtime_status(
        platform="feishu",
        platform_state="fatal",
        error_code="gateway_conflict_port",
        error_message="boom",
        error_detail={"kind": "port", "port": 8765, "can_takeover": False},
    )
    feishu = gs.read_runtime_status()["platforms"]["feishu"]
    assert feishu["error_code"] == "gateway_conflict_port"
    assert feishu["error_detail"] == {"kind": "port", "port": 8765, "can_takeover": False}

    # Passing None clears the structured detail without touching other fields.
    gs.write_runtime_status(platform="feishu", error_detail=None)
    feishu = gs.read_runtime_status()["platforms"]["feishu"]
    assert "error_detail" not in feishu
    assert feishu["error_code"] == "gateway_conflict_port"


def test_set_gateway_conflict_set_and_clear(runtime_dir):
    import gateway.status as gs

    gs.set_gateway_conflict({"kind": "service", "can_takeover": True, "message": "svc"})
    rec = gs.read_runtime_status()
    assert rec["gateway_conflict"]["kind"] == "service"
    assert rec["gateway_conflict"]["can_takeover"] is True
    assert "updated_at" in rec["gateway_conflict"]

    gs.set_gateway_conflict(None)
    assert "gateway_conflict" not in gs.read_runtime_status()


def test_classify_port_conflict_no_psutil(runtime_dir, monkeypatch):
    import gateway.status as gs

    # Force the ImportError path → safe, non-takeover-able default.
    monkeypatch.setitem(sys.modules, "psutil", None)
    detail = gs.classify_port_conflict(8765)
    assert detail == {
        "kind": "port",
        "port": 8765,
        "can_takeover": False,
        "owner_pid": None,
        "owner_home": None,
    }


def test_classify_port_conflict_local_owner(runtime_dir, monkeypatch):
    import gateway.status as gs

    fake_conn = types.SimpleNamespace(
        laddr=types.SimpleNamespace(port=8765),
        status="LISTEN",
        pid=4321,
    )

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def environ(self):
            return {"HERMES_HOME": "/home/u/.hermes"}

    fake_psutil = types.SimpleNamespace(
        CONN_LISTEN="LISTEN",
        CONN_NONE="NONE",
        net_connections=lambda kind="inet": [fake_conn],
        Process=FakeProc,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    detail = gs.classify_port_conflict(8765)
    assert detail["can_takeover"] is True
    assert detail["owner_pid"] == 4321
    assert detail["owner_home"] == "/home/u/.hermes"


# --------------------------------------------------------------------------- #
# feishu._classify_address_in_use
# --------------------------------------------------------------------------- #
def test_classify_address_in_use_detects_eaddrinuse(runtime_dir):
    from plugins.platforms.feishu.adapter import _classify_address_in_use

    wrapped = RuntimeError("startup failed")
    wrapped.__cause__ = OSError(errno.EADDRINUSE, "Address already in use")
    detail = _classify_address_in_use(wrapped, 8765)
    assert detail is not None
    assert detail["kind"] == "port"
    assert detail["port"] == 8765


def test_classify_address_in_use_windows_code(runtime_dir):
    from plugins.platforms.feishu.adapter import _classify_address_in_use

    win = OSError()
    win.errno = 10048  # WSAEADDRINUSE
    assert _classify_address_in_use(win, 8765) is not None


def test_classify_address_in_use_ignores_other_errors(runtime_dir):
    from plugins.platforms.feishu.adapter import _classify_address_in_use

    assert _classify_address_in_use(ValueError("nope"), 8765) is None


# --------------------------------------------------------------------------- #
# hermes_cli.gateway restart path
# --------------------------------------------------------------------------- #
def _restart_args():
    return types.SimpleNamespace(gateway_command="restart", system=False, all=False)


@pytest.fixture()
def windows_restart_env(monkeypatch):
    """Drive the restart branch to the Windows path, desktop-managed."""
    import hermes_cli.gateway as gw

    monkeypatch.setenv("HERMES_DESKTOP_MANAGED", "1")
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    monkeypatch.setattr(gw, "is_windows", lambda: True)
    monkeypatch.setattr(gw, "is_macos", lambda: False)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda _x: False)
    monkeypatch.setattr(gw, "_dispatch_all_via_service_manager_if_s6", lambda _x: False)
    return gw


def test_desktop_managed_foreign_service_declines(runtime_dir, windows_restart_env, monkeypatch):
    gw = windows_restart_env
    monkeypatch.delenv("HERMES_GATEWAY_FORCE_TAKEOVER", raising=False)

    fake_win = types.SimpleNamespace(
        is_installed=lambda: True,
        restart=mock.Mock(),
        stop=mock.Mock(),
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.gateway_windows", fake_win)
    run_gateway = mock.Mock()
    monkeypatch.setattr(gw, "run_gateway", run_gateway)

    with pytest.raises(SystemExit) as exc:
        gw._gateway_command_inner(_restart_args())
    assert exc.value.code == 2

    # Declined: did not restart the foreign service, did not run our own,
    # and recorded a takeover-able conflict for the desktop.
    fake_win.restart.assert_not_called()
    run_gateway.assert_not_called()
    import gateway.status as gs

    assert gs.read_runtime_status()["gateway_conflict"]["kind"] == "service"


def test_desktop_managed_force_takeover(runtime_dir, windows_restart_env, monkeypatch):
    gw = windows_restart_env
    monkeypatch.setenv("HERMES_GATEWAY_FORCE_TAKEOVER", "1")

    fake_win = types.SimpleNamespace(
        is_installed=lambda: True,
        restart=mock.Mock(),
        stop=mock.Mock(),
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.gateway_windows", fake_win)
    monkeypatch.setattr(gw, "_free_conflicting_local_gateways", lambda: 0)
    monkeypatch.setattr(gw, "stop_profile_gateway", lambda: False)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **k: None)
    run_gateway = mock.Mock()
    monkeypatch.setattr(gw, "run_gateway", run_gateway)

    gw._gateway_command_inner(_restart_args())

    # Took over: stopped the foreign service and ran a desktop-managed gateway
    # with --replace.
    fake_win.stop.assert_called_once()
    fake_win.restart.assert_not_called()
    run_gateway.assert_called_once()
    assert run_gateway.call_args.kwargs.get("replace") is True


def test_non_desktop_windows_uses_service_restart(runtime_dir, windows_restart_env, monkeypatch):
    gw = windows_restart_env
    monkeypatch.delenv("HERMES_DESKTOP_MANAGED", raising=False)  # not desktop-managed
    monkeypatch.delenv("HERMES_GATEWAY_FORCE_TAKEOVER", raising=False)

    fake_win = types.SimpleNamespace(
        is_installed=lambda: True,
        restart=mock.Mock(),
        stop=mock.Mock(),
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.gateway_windows", fake_win)
    run_gateway = mock.Mock()
    monkeypatch.setattr(gw, "run_gateway", run_gateway)

    gw._gateway_command_inner(_restart_args())

    # Unchanged legacy behavior: restart the installed service, never run our own.
    fake_win.restart.assert_called_once()
    run_gateway.assert_not_called()
