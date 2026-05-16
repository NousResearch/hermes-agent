from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
HTTPException = fastapi.HTTPException


MODULE_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "plugin_api.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("kb_live_dashboard_plugin_api_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dashboard_command_uses_configured_command(monkeypatch):
    module = _load_module()

    monkeypatch.setenv("HERMES_KB_DASHBOARD_COMMAND", "kb mcp call dashboard.live --arguments-json {args_json} --json")

    command = module._dashboard_command(limit=7)

    assert command[:4] == ["kb", "mcp", "call", "dashboard.live"]
    args = json.loads(command[5])
    assert args["limit"] == 7
    assert args["include_feedback"] is True


def test_dashboard_command_uses_ssh_mcp_environment(monkeypatch):
    module = _load_module()

    monkeypatch.delenv("HERMES_KB_DASHBOARD_COMMAND", raising=False)
    monkeypatch.setenv("HERMES_KB_MCP_SSH_TARGET", "helix-vpn")
    monkeypatch.setenv("HERMES_KB_WORKSPACE", "/home/abcosta/Knowledge/kb-anthony")
    monkeypatch.setenv("HERMES_KB_BIN", "/home/abcosta/.local/share/kb-engine-prod/.venv/bin/kb")

    command = module._dashboard_command(limit=5)

    assert command[:5] == ["ssh", "-T", "-o", "BatchMode=yes", "-o"]
    assert "helix-vpn" in command
    assert "dashboard.live" in command[-1]
    assert "/home/abcosta/Knowledge/kb-anthony" in command[-1]


def test_dashboard_command_refreshes_hermes_env(monkeypatch, tmp_path):
    module = _load_module()
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text(
        "\n".join(
            [
                "HERMES_KB_MCP_SSH_TARGET=helix-vpn",
                "HERMES_KB_WORKSPACE=/home/abcosta/Knowledge/kb-anthony",
                "HERMES_KB_BIN=/home/abcosta/.local/share/kb-engine-prod/.venv/bin/kb",
                "",
            ]
        )
    )
    for key in (
        "HERMES_KB_DASHBOARD_COMMAND",
        "HERMES_KB_MCP_SSH_TARGET",
        "HERMES_KB_WORKSPACE",
        "HERMES_KB_BIN",
        "HERMES_KB_PROD_BIN",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    command = module._dashboard_command(limit=6)

    assert command[0] == "ssh"
    assert "helix-vpn" in command
    assert "dashboard.live" in command[-1]


@pytest.mark.asyncio
async def test_live_dashboard_unwraps_mcp_packet(monkeypatch):
    module = _load_module()

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "structuredContent": {
                        "surface": "dashboard.live",
                        "summary": {"readiness_status": "ready", "publication_status": "clean"},
                    }
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = await module.live_dashboard(limit=5)

    assert result["ok"] is True
    assert result["payload"]["surface"] == "dashboard.live"
    assert result["payload"]["summary"]["readiness_status"] == "ready"


@pytest.mark.asyncio
async def test_live_dashboard_reports_command_failure(monkeypatch):
    module = _load_module()

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(HTTPException) as excinfo:
        await module.live_dashboard(limit=5)

    assert excinfo.value.status_code == 502
