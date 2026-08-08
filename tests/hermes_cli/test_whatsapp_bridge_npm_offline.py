"""Measured-work pins for npm --prefer-offline at the WhatsApp bridge sites.

perf(cli) #39267/#39399 adopted ``--prefer-offline`` for the update/web
npm installs so npm reuses its local cache instead of re-fetching
metadata. The WhatsApp bridge installs (CLI setup and dashboard surface)
were sibling sites that missed the flag; these pins hold the propagation
in place.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def bridge_dir(tmp_path: Path) -> Path:
    d = tmp_path / "whatsapp-bridge"
    d.mkdir(parents=True)
    (d / "bridge.js").write_text("// bridge")
    return d


def test_cli_whatsapp_bridge_install_prefers_offline(bridge_dir: Path, monkeypatch) -> None:
    """``cmd_whatsapp``'s bridge install passes ``--prefer-offline``.

    The setup wizard runs ``npm install`` in the bridge dir when
    node_modules is missing. It must reuse npm's local cache (same flag
    as the update path) rather than re-fetching metadata.
    """
    import hermes_cli.main as cli_main

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    import hermes_constants

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    # The helper imports these from hermes_constants inside the function,
    # so patch the source module.
    monkeypatch.setattr(
        hermes_constants, "find_node_executable", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", lambda: {})

    ok = cli_main._install_whatsapp_bridge_deps(bridge_dir)

    assert ok, "install helper should report success"
    assert calls, "npm install should have run"
    cmd = calls[0][0]
    assert cmd[0] == "/usr/bin/npm" and cmd[1] == "install"
    assert "--prefer-offline" in cmd, (
        f"WhatsApp bridge npm install should pass --prefer-offline "
        f"(same as update path, perf(cli) #39267), got: {cmd}"
    )


def test_dashboard_whatsapp_bridge_install_prefers_offline(bridge_dir: Path, monkeypatch) -> None:
    """``_ensure_whatsapp_bridge_dependencies`` passes ``--prefer-offline``.

    The dashboard-initiated bridge install is the second sibling npm
    install site (web_server.py). Without the flag a dashboard-triggered
    reinstall re-fetches registry metadata.
    """
    import hermes_cli.web_server as ws

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    import hermes_constants
    import utils

    monkeypatch.setattr(ws.subprocess, "run", fake_run)
    # The function imports these from the source modules inside its body,
    # so patch the modules themselves.
    monkeypatch.setattr(
        hermes_constants, "find_node_executable", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", lambda: {})
    monkeypatch.setattr(utils, "env_int", lambda _k, _d: 300)
    monkeypatch.setattr(ws, "windows_hide_flags", lambda: 0)

    ws._ensure_whatsapp_bridge_dependencies(bridge_dir)

    assert calls, "npm install should have run"
    cmd = calls[0][0]
    assert cmd[0] == "/usr/bin/npm" and cmd[1] == "install"
    assert "--prefer-offline" in cmd, (
        f"Dashboard WhatsApp bridge npm install should pass --prefer-offline, got: {cmd}"
    )
