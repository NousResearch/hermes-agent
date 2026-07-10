"""Tests for notebooklm-mcp-cli bridge integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plugins.notebooklm import bridge, core, mcp_stack


def test_resolve_nlm_command_prefers_path(monkeypatch):
    monkeypatch.setattr(bridge.shutil, "which", lambda name: "/usr/bin/nlm" if name == "nlm" else None)
    assert bridge.resolve_nlm_command() == ("nlm", [])


def test_resolve_nlm_command_falls_back_to_uvx(monkeypatch):
    def fake_which(name: str):
        if name == "uvx":
            return "/usr/bin/uvx"
        return None

    monkeypatch.setattr(bridge.shutil, "which", fake_which)
    monkeypatch.setenv("NOTEBOOKLM_MCP_CLI_REF", "b9cf0e2")
    resolved = bridge.resolve_nlm_command()
    assert resolved is not None
    assert resolved[0] == "uvx"
    assert "git+https://github.com/zapabob/notebooklm-mcp-cli@b9cf0e2" in resolved[1]


def test_run_nlm_parses_json(monkeypatch):
    payload = {"authenticated": True}

    class Result:
        returncode = 0
        stdout = json.dumps(payload)
        stderr = ""

    monkeypatch.setattr(bridge, "resolve_nlm_command", lambda: ("nlm", []))
    monkeypatch.setattr(bridge.subprocess, "run", lambda *a, **k: Result())
    out = bridge.run_nlm(["login", "--check"])
    assert out["ok"] is True
    assert out["data"] == payload


def test_auth_status_detects_valid_stdout(monkeypatch):
    class Result:
        returncode = 0
        stdout = "✓ Authentication valid!\n  Notebooks found: 15"
        stderr = ""

    monkeypatch.setattr(bridge, "resolve_nlm_command", lambda: ("nlm", []))
    monkeypatch.setattr(bridge.subprocess, "run", lambda *a, **k: Result())
    out = bridge.auth_status()
    assert out["authenticated"] is True


def test_sync_source_consumer_requires_auth(tmp_path, monkeypatch):
    src = tmp_path / "source.md"
    src.write_text("# test\n", encoding="utf-8")
    monkeypatch.setattr(bridge, "cli_available", lambda: True)
    monkeypatch.setattr(
        bridge,
        "auth_status",
        lambda **_: {"ok": True, "authenticated": False},
    )
    out = core.sync_source_consumer(source_path=src, notebook_id="nb-1")
    assert out["ok"] is False
    assert "auth" in out


def test_sync_source_auto_uses_consumer_when_enterprise_missing(tmp_path, monkeypatch):
    src = tmp_path / "source.md"
    src.write_text("# test\n", encoding="utf-8")

    cfg = core.settings()
    monkeypatch.setattr(core, "settings", lambda: cfg)
    monkeypatch.setattr(core, "_enterprise_ready", lambda _cfg: False)
    monkeypatch.setattr(
        core,
        "sync_source_consumer",
        lambda **_: {"ok": True, "backend": "consumer"},
    )

    out = core.sync_source(source_path=src, mode="auto")
    assert out["ok"] is True
    assert out["backend"] == "consumer"


def test_build_mcp_server_config_from_resolved_command(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "resolve_mcp_command",
        lambda: ("uvx", ["--from", "git+https://github.com/zapabob/notebooklm-mcp-cli@b9cf0e2", "notebooklm-mcp"]),
    )
    cfg = mcp_stack.build_mcp_server_config()
    assert cfg["command"] == "uvx"
    assert cfg["enabled"] is True


def test_ensure_mcp_server_dry_run(monkeypatch):
    monkeypatch.setattr(
        mcp_stack,
        "mcp_server_status",
        lambda: {"configured": False},
    )
    monkeypatch.setattr(
        mcp_stack,
        "build_mcp_server_config",
        lambda: {"command": "notebooklm-mcp", "args": [], "enabled": True},
    )
    out = mcp_stack.ensure_mcp_server(dry_run=True)
    assert out["status"] == "would_install"


def test_status_includes_consumer_bridge(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "bridge_status",
        lambda: {"cli_available": True},
    )
    monkeypatch.setattr(
        mcp_stack,
        "mcp_server_status",
        lambda: {"configured": False},
    )
    st = core.status()
    assert "consumer_cli" in st
    assert st["consumer_cli"]["cli_available"] is True
