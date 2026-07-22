"""Tests for incomplete Baileys package detection during ``hermes whatsapp``.

Regression coverage for #52713: a prior failed or interrupted ``npm install``
can leave ``node_modules/`` present but ``@whiskeysockets/baileys`` empty,
causing the bridge to crash with ``ERR_MODULE_NOT_FOUND`` at startup.  The
CLI must detect this and trigger a reinstall instead of printing
"✓ Bridge dependencies already installed" and proceeding.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def bridge_env(tmp_path, monkeypatch):
    """Set up a fake bridge directory with bridge.js and package.json."""
    home = tmp_path / "home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    for key in list(os.environ):
        if key.startswith("WHATSAPP_"):
            monkeypatch.delenv(key, raising=False)

    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    (bridge_dir / "bridge.js").write_text("// bridge\n")
    (bridge_dir / "package.json").write_text(
        '{"dependencies": {"@whiskeysockets/baileys": "git+ssh://test"}}\n'
    )
    return bridge_dir


def _patch_cmd_deps(bridge_dir, monkeypatch, npm_calls=None):
    """Stub all external dependencies for cmd_whatsapp."""
    if npm_calls is None:
        npm_calls = []

    import gateway.platforms.whatsapp_common as wa_common
    monkeypatch.setattr(wa_common, "resolve_whatsapp_bridge_dir", lambda: bridge_dir)

    def fake_subprocess_run(cmd, **kwargs):
        npm_calls.append(cmd)
        cwd = kwargs.get("cwd", "")
        if "npm" in str(cmd) and str(bridge_dir) in str(cwd):
            baileys_dir = bridge_dir / "node_modules" / "@whiskeysockets" / "baileys"
            baileys_dir.mkdir(parents=True, exist_ok=True)
            (baileys_dir / "package.json").write_text("{}")
        return MagicMock(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("subprocess.run", fake_subprocess_run)

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda _n: "/usr/bin/npm")
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", lambda: os.environ.copy())


def _run_cmd_whatsapp(monkeypatch):
    """Run cmd_whatsapp with enough inputs to reach the dependency step."""
    from hermes_cli.main import cmd_whatsapp

    # Provide inputs for: mode choice, allowed users, phone number, session prompts
    inputs = iter(["1", "15551234567", "n"])

    def fake_input(_prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            return "n"

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("hermes_cli.main._require_tty", lambda *_a, **_kw: None)

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            cmd_whatsapp(MagicMock())
        except (KeyboardInterrupt, StopIteration):
            pass

    return buf.getvalue()


def test_incomplete_baileys_triggers_reinstall(bridge_env, monkeypatch):
    """When node_modules exists but Baileys package.json is missing, reinstall."""
    bridge_dir = bridge_env
    node_modules = bridge_dir / "node_modules"
    node_modules.mkdir()
    (node_modules / "@whiskeysockets").mkdir(parents=True)
    # Empty baileys dir — no package.json

    npm_calls = []
    _patch_cmd_deps(bridge_dir, monkeypatch, npm_calls)

    output = _run_cmd_whatsapp(monkeypatch)
    assert "incomplete" in output.lower() or "reinstall" in output.lower(), (
        f"Expected reinstall message but got: {output}"
    )
    assert len(npm_calls) >= 1, "Expected npm install to be called"


def test_valid_baileys_skips_reinstall(bridge_env, monkeypatch):
    """When node_modules and Baileys package.json both exist, skip install."""
    bridge_dir = bridge_env
    node_modules = bridge_dir / "node_modules"
    node_modules.mkdir()
    baileys_dir = node_modules / "@whiskeysockets" / "baileys"
    baileys_dir.mkdir(parents=True)
    (baileys_dir / "package.json").write_text("{}")

    npm_calls = []
    _patch_cmd_deps(bridge_dir, monkeypatch, npm_calls)

    output = _run_cmd_whatsapp(monkeypatch)
    assert "already installed" in output, (
        f"Expected 'already installed' but got: {output}"
    )
    # Filter out bridge.js calls — only check for npm install calls
    npm_install_calls = [c for c in npm_calls if "npm" in str(c) and "bridge.js" not in str(c)]
    assert len(npm_install_calls) == 0, "npm install should not have been called"


def test_missing_node_modules_installs_normally(bridge_env, monkeypatch):
    """When node_modules is entirely absent, install from scratch."""
    bridge_dir = bridge_env
    # No node_modules at all

    npm_calls = []
    _patch_cmd_deps(bridge_dir, monkeypatch, npm_calls)

    output = _run_cmd_whatsapp(monkeypatch)
    assert "Installing" in output or "Dependencies" in output, (
        f"Expected install message but got: {output}"
    )
    assert len(npm_calls) >= 1, "Expected npm install to be called"
