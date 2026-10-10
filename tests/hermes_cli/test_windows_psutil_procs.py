"""#121719: Windows process scans must not spawn PowerShell."""

from __future__ import annotations

import sys
import types


def _load_gateway():
    import hermes_cli.gateway as gw

    return gw


def test_windows_process_pairs_uses_psutil(monkeypatch):
    gw = _load_gateway()

    class FakeProc:
        def __init__(self, pid, cmdline):
            self.info = {"pid": pid, "cmdline": cmdline}

    fake_psutil = types.SimpleNamespace(
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        AccessDenied=type("AccessDenied", (Exception,), {}),
        ZombieProcess=type("ZombieProcess", (Exception,), {}),
        process_iter=lambda attrs: iter(
            [FakeProc(123, ["C:\\x\\python.exe", "hermes", "gateway", "run"]), FakeProc(456, ["notepad.exe"])]
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    pairs = list(gw._windows_process_pairs())
    assert (123, "C:\\x\\python.exe hermes gateway run") in pairs
    assert any(pid == 456 for pid, _ in pairs)


def test_windows_process_listing_has_no_powershell_fallback():
    import inspect

    gw = _load_gateway()
    src = inspect.getsource(gw._windows_process_listing)
    assert "Get-CimInstance" not in src
    assert "powershell" not in src.lower()


def test_scan_gateway_pids_prefers_pairs_over_listing(monkeypatch):
    gw = _load_gateway()
    monkeypatch.setattr(gw, "is_windows", lambda: True)
    seen = {}

    def fake_pairs():
        seen["used"] = True
        yield 777, "hermes gateway run --profile default"

    monkeypatch.setattr(gw, "_windows_process_pairs", fake_pairs)
    monkeypatch.setattr(
        gw, "_windows_process_listing", lambda: (_ for _ in ()).throw(AssertionError("must not be called"))
    )
    pids = gw._scan_gateway_pids(set(), all_profiles=True)
    assert seen.get("used") is True
    assert pids == [777]


def test_claw_node_lookup_prefers_psutil(monkeypatch):
    import hermes_cli.claw as claw_mod

    class FakeProc:
        def __init__(self, pid, name, cmdline):
            self.info = {"pid": pid, "name": name, "cmdline": cmdline}

    fake_psutil = types.SimpleNamespace(
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        AccessDenied=type("AccessDenied", (Exception,), {}),
        ZombieProcess=type("ZombieProcess", (Exception,), {}),
        process_iter=lambda attrs: iter([FakeProc(999, "node.exe", ["node", "openclaw-gateway.js"])]),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert claw_mod._node_pid_with_cmdline_match("openclaw|clawd") == 999
    assert claw_mod._node_pid_with_cmdline_match("nothing-here") is None
