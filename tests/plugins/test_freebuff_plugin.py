"""Tests for the Freebuff Hermes plugin."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.freebuff import core


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def test_binary_name_windows(monkeypatch):
    monkeypatch.setattr(core.os, "name", "nt")
    assert core.binary_name() == "freebuff.exe"


def test_binary_name_posix(monkeypatch):
    monkeypatch.setattr(core.os, "name", "posix")
    assert core.binary_name() == "freebuff"


def test_launch_command_prefers_binary(tmp_path, monkeypatch):
    binary = tmp_path / ".config" / "manicode" / "freebuff.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fake")
    monkeypatch.setattr(core, "default_binary_path", lambda: binary)
    monkeypatch.setattr(core, "resolve_workdir", lambda explicit=None: tmp_path)
    monkeypatch.setattr(core.shutil, "which", lambda _name: None)
    assert core.launch_command() == [str(binary)]


def test_launch_command_falls_back_to_npx(monkeypatch):
    monkeypatch.setattr(core, "default_binary_path", lambda: Path("/missing/freebuff"))
    monkeypatch.setattr(
        core.shutil,
        "which",
        lambda name: "/usr/bin/npx" if name == "npx" else None,
    )
    assert core.launch_command() == ["/usr/bin/npx", "--yes", "freebuff"]


def test_read_metadata_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "METADATA_PATH", tmp_path / "freebuff-metadata.json")
    assert core._read_metadata() == {}


def test_read_metadata_valid(tmp_path, monkeypatch):
    meta = tmp_path / "freebuff-metadata.json"
    meta.write_text(json.dumps({"version": "0.0.115", "target": "win32-x64"}), encoding="utf-8")
    monkeypatch.setattr(core, "METADATA_PATH", meta)
    payload = core._read_metadata()
    assert payload["version"] == "0.0.115"


def test_doctor_flags_missing_node(monkeypatch):
    monkeypatch.setattr(
        core,
        "status",
        lambda: {
            "node": {},
            "binary": {"exists": False},
            "cli_on_path": None,
            "upstream_token_set": True,
            "provider_registered": True,
            "model_provider": "freebuff",
            "proxy": {"installed": True, "running": True, "probe": {"ok": True}},
        },
    )
    monkeypatch.setattr(core, "_plugin_enabled", lambda config=None: False)
    monkeypatch.setattr(core, "resolve_workdir", lambda explicit=None: Path("."))
    payload = core.doctor()
    assert payload["ok"] is False
    assert any("Node.js" in issue for issue in payload["issues"])


def test_setup_enables_plugin(hermes_home):
    config_path = hermes_home / "config.yaml"
    config_path.write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    with patch.object(core, "doctor", return_value={"ok": True, "issues": [], "warnings": []}):
        payload = core.setup()
    assert payload["ok"] is True
    assert "plugins.enabled+=freebuff" in payload["changed"]
    saved = config_path.read_text(encoding="utf-8")
    assert "freebuff" in saved


def test_handle_launch_dry_run_default():
    with patch.object(core, "run", return_value={"ok": True, "dry_run": True, "command": ["freebuff"]}):
        raw = core.handle_launch({}, task_id="t1")
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert "hint" in payload


def test_load_auth_token_from_credentials(tmp_path, monkeypatch):
    from plugins.freebuff import token as token_mod

    creds = tmp_path / "credentials.json"
    creds.write_text(
        json.dumps({"default": {"authToken": "test-token-abc123"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(token_mod, "credentials_path", lambda: creds)
    assert token_mod.load_auth_token_from_credentials() == "test-token-abc123"


def test_connect_applies_model(hermes_home, monkeypatch):
    config_path = hermes_home / "config.yaml"
    config_path.write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    monkeypatch.setattr(
        core.token_bridge,
        "sync_upstream_token_to_env",
        lambda force=False: {"ok": True},
    )
    monkeypatch.setattr(
        core.token_bridge,
        "ensure_proxy_api_key",
        lambda force=False: {"ok": True},
    )
    monkeypatch.setattr(
        core.proxy_mgr,
        "proxy_status",
        lambda: {"installed": True, "running": True, "probe": {"ok": True}},
    )
    monkeypatch.setattr(
        core.proxy_mgr,
        "start_proxy",
        lambda force_restart=False: {"ok": True, "base_url": "http://127.0.0.1:8765/v1"},
    )
    with patch.object(core, "doctor", return_value={"ok": True, "issues": [], "warnings": []}):
        payload = core.connect(start_proxy=True, install_proxy=False)
    assert payload["ok"] is True
    saved = config_path.read_text(encoding="utf-8")
    assert "provider: freebuff" in saved or "provider: 'freebuff'" in saved or "freebuff" in saved
