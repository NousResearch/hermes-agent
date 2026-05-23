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


def test_dashboard_cwd_is_none_for_ssh_command(monkeypatch):
    module = _load_module()

    monkeypatch.setenv("HERMES_KB_WORKSPACE", "/remote-only/kb-anthony")

    assert module._dashboard_cwd(["ssh", "-T", "helix-vpn", "kb"]) is None


def test_dashboard_cwd_uses_existing_local_workspace(monkeypatch, tmp_path):
    module = _load_module()

    monkeypatch.setenv("HERMES_KB_WORKSPACE", str(tmp_path))

    assert module._dashboard_cwd(["kb", "mcp", "call", "dashboard.live"]) == str(tmp_path)


@pytest.mark.asyncio
async def test_standalone_dashboard_url_defaults_to_tailnet(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HERMES_KB_DASHBOARD_STANDALONE_URL", raising=False)
    monkeypatch.delenv("KB_DASHBOARD_URL", raising=False)

    result = await module.standalone_dashboard()

    assert result["ok"] is True
    assert result["url"] == "https://helix.tailca54ca.ts.net:9121"
    assert result["deprecated_plugin"] is True


@pytest.mark.asyncio
async def test_standalone_dashboard_url_uses_environment(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("HERMES_KB_DASHBOARD_STANDALONE_URL", "https://example.test/dashboard")

    result = await module.standalone_dashboard()

    assert result["url"] == "https://example.test/dashboard"


@pytest.mark.asyncio
async def test_live_dashboard_unwraps_mcp_packet(monkeypatch):
    module = _load_module()
    seen = {}

    def fake_run(*_args, **kwargs):
        seen["cwd"] = kwargs.get("cwd")
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
    assert seen["cwd"] is None


@pytest.mark.asyncio
async def test_live_dashboard_reports_command_failure(monkeypatch):
    module = _load_module()

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(HTTPException) as excinfo:
        await module.live_dashboard(limit=5)

    assert excinfo.value.status_code == 502


@pytest.mark.asyncio
async def test_html_dashboard_reads_configured_artifact(monkeypatch, tmp_path):
    module = _load_module()
    html_path = tmp_path / "DASHBOARD.html"
    html_path.write_text(
        '<html><body><h1>Dashboard</h1>'
        '<script type="application/json" id="kb-dashboard-metadata">'
        '{"source_surfaces":["attention.cockpit","dashboard.live"]}'
        "</script></body></html>"
    )
    monkeypatch.setenv("HERMES_KB_DASHBOARD_HTML_PATH", str(html_path))
    monkeypatch.delenv("HERMES_KB_DASHBOARD_HTML_COMMAND", raising=False)

    result = await module.html_dashboard()

    assert result["ok"] is True
    assert result["source"] == "kb.dashboard.html"
    assert "<html" in result["html"].lower()
    assert result["metadata"]["source_surfaces"] == ["attention.cockpit", "dashboard.live"]


@pytest.mark.asyncio
async def test_html_dashboard_reports_missing_artifact(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setenv("HERMES_KB_DASHBOARD_HTML_PATH", str(tmp_path / "missing.html"))
    monkeypatch.delenv("HERMES_KB_DASHBOARD_HTML_COMMAND", raising=False)

    with pytest.raises(HTTPException) as excinfo:
        await module.html_dashboard()

    assert excinfo.value.status_code == 503
    assert "KB dashboard HTML artifact is not available" in excinfo.value.detail


@pytest.mark.asyncio
async def test_html_dashboard_uses_configured_command(monkeypatch):
    module = _load_module()
    html = '<html><body><h1>Dashboard</h1></body></html>'

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=html, stderr="")

    monkeypatch.setenv("HERMES_KB_DASHBOARD_HTML_COMMAND", "cat /tmp/DASHBOARD.html")
    monkeypatch.delenv("HERMES_KB_DASHBOARD_HTML_PATH", raising=False)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = await module.html_dashboard()

    assert result["ok"] is True
    assert result["source"] == "kb.dashboard.html"
    assert result["metadata"] == {}
