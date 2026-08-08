"""Behavioral coverage: WhatsApp adapter helper spawns hide the Windows console.

The gateway commonly runs under pythonw.exe.  A synchronous console app
spawned from that console-less process flashes a visible window unless it
receives CREATE_NO_WINDOW via creationflags.  All subprocess.run calls in
the adapter route through _run_hidden(), which injects the flag by default
so new call sites inherit the fix (prior regressions in this class:
#53282, #56747, #63698, #68457).

These tests exercise the real call paths with a mocked subprocess.run and
assert the captured kwargs — no source/AST inspection.  Only paths that are
actually reachable on Windows are tested; the lsof/ss port probes are
POSIX-only and are covered by the _run_hidden contract test instead.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from plugins.platforms.whatsapp import adapter as whatsapp_adapter
from tests.gateway.test_whatsapp_connect import _make_adapter

_CREATE_NO_WINDOW = 0x08000000


def _capture_run(monkeypatch):
    """Monkeypatch Windows + hide flags + subprocess.run; return captured list."""
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd) if cmd is not None else cmd, dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(whatsapp_adapter, "_IS_WINDOWS", True)
    monkeypatch.setattr(whatsapp_adapter, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(whatsapp_adapter.subprocess, "run", fake_run)
    return captured


def test_run_hidden_injects_hide_flag_by_default(monkeypatch):
    """_run_hidden must inject creationflags=windows_hide_flags() when absent.

    This is the recurrence guard: every subprocess.run in the adapter goes
    through _run_hidden, so a new spawn site added without any flags still
    gets CREATE_NO_WINDOW on Windows.
    """
    captured = _capture_run(monkeypatch)

    whatsapp_adapter._run_hidden(["some-helper"], capture_output=True, timeout=5)

    assert len(captured) == 1
    _, kwargs = captured[0]
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW


def test_run_hidden_respects_explicit_creationflags(monkeypatch):
    """An explicitly passed creationflags value must win over the default."""
    captured = _capture_run(monkeypatch)

    whatsapp_adapter._run_hidden(["some-helper"], creationflags=0x123)

    _, kwargs = captured[0]
    assert kwargs["creationflags"] == 0x123


def test_check_whatsapp_requirements_probe_hides_console_window(monkeypatch):
    """The node --version probe must carry CREATE_NO_WINDOW on Windows.

    This is the worst offender in practice: the channel monitor re-probes
    every ~5 minutes while the bridge is down, flashing a console window
    on a permanent cycle.
    """
    captured = _capture_run(monkeypatch)
    monkeypatch.setattr(whatsapp_adapter, "find_node_executable", lambda _name: "node")

    assert whatsapp_adapter.check_whatsapp_requirements() is True

    node_spawns = [(cmd, kw) for cmd, kw in captured if cmd and cmd[-1] == "--version"]
    assert node_spawns, f"no node --version probe captured: {captured}"
    _, kwargs = node_spawns[0]
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW
    assert kwargs["capture_output"] is True


def test_terminate_bridge_process_taskkill_hides_console_window(monkeypatch):
    """Bridge-termination taskkill must carry CREATE_NO_WINDOW on Windows."""
    captured = _capture_run(monkeypatch)
    proc = SimpleNamespace(pid=1234)

    whatsapp_adapter._terminate_bridge_process(proc, force=True)

    taskkills = [(cmd, kw) for cmd, kw in captured if cmd and "taskkill" in cmd]
    assert taskkills, f"no taskkill spawn captured: {captured}"
    _, kwargs = taskkills[0]
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW


def test_kill_port_process_netstat_and_taskkill_hide_console_window(monkeypatch):
    """Stale-bridge cleanup (netstat + taskkill) must hide both spawns on Windows."""
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((list(cmd), dict(kwargs)))
        if cmd[0] == "netstat":
            return SimpleNamespace(
                returncode=0,
                stdout="  TCP    0.0.0.0:19876    0.0.0.0:0    LISTENING    4321\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(whatsapp_adapter, "_IS_WINDOWS", True)
    monkeypatch.setattr(whatsapp_adapter, "windows_hide_flags", lambda: _CREATE_NO_WINDOW)
    monkeypatch.setattr(whatsapp_adapter.subprocess, "run", fake_run)

    whatsapp_adapter._kill_port_process(19876)

    spawned = [cmd[0] for cmd, _ in captured]
    assert "netstat" in spawned, f"no netstat spawn captured: {captured}"
    assert "taskkill" in spawned, f"no taskkill spawn captured: {captured}"
    for cmd, kwargs in captured:
        assert kwargs["creationflags"] == _CREATE_NO_WINDOW, f"{cmd[0]} spawned unhidden"


@pytest.mark.asyncio
async def test_connect_npm_install_hides_console_window(tmp_path, monkeypatch):
    """npm install during connect must carry CREATE_NO_WINDOW on Windows."""
    bridge_dir = tmp_path / "whatsapp-bridge"
    bridge_dir.mkdir()
    (bridge_dir / "bridge.js").write_text("// bridge\n", encoding="utf-8")
    (bridge_dir / "package.json").write_text('{"name":"bridge"}\n', encoding="utf-8")
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / "creds.json").write_text("{}", encoding="utf-8")

    adapter = _make_adapter()
    adapter._bridge_script = str(bridge_dir / "bridge.js")
    adapter._session_path = session_path

    captured = _capture_run(monkeypatch)
    monkeypatch.setattr(whatsapp_adapter, "check_whatsapp_requirements", lambda: True)
    monkeypatch.setattr(whatsapp_adapter, "find_node_executable", lambda _name: "npm")
    monkeypatch.setattr(whatsapp_adapter, "with_hermes_node_path", lambda: {"PATH": "x"})

    # Fail the install after capture so connect() returns early without
    # needing aiohttp / Popen plumbing for the rest of the bootstrap.
    def fake_run(cmd, **kwargs):
        captured.append((list(cmd) if cmd is not None else cmd, dict(kwargs)))
        return SimpleNamespace(returncode=1, stdout="", stderr="fail")

    monkeypatch.setattr(whatsapp_adapter.subprocess, "run", fake_run)

    with patch.object(adapter, "_acquire_platform_lock", return_value=True), \
         patch.object(adapter, "_release_platform_lock"):
        result = await adapter.connect()

    assert result is False
    npm_spawns = [
        (cmd, kw)
        for cmd, kw in captured
        if cmd and len(cmd) >= 2 and cmd[1] == "install"
    ]
    assert npm_spawns, f"no npm install spawn captured: {captured}"
    _, kwargs = npm_spawns[0]
    assert kwargs["creationflags"] == _CREATE_NO_WINDOW
