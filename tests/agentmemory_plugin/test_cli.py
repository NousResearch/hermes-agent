"""Tests for plugins/memory/agentmemory/cli.py."""

from __future__ import annotations

from types import SimpleNamespace


def test_mcp_command_writes_agentmemory_mcp_config(monkeypatch, tmp_path, capsys):
    import plugins.memory.agentmemory.cli as agentmemory_cli

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(agentmemory_cli, "_config_path", lambda: config_path)

    agentmemory_cli.agentmemory_command(SimpleNamespace(agentmemory_command="mcp"))

    out = capsys.readouterr().out
    assert "AgentMemory MCP configured" in out
    saved = config_path.read_text()
    assert "agentmemory:" in saved
    assert "@agentmemory/mcp" in saved


def test_provider_command_sets_memory_provider(monkeypatch, tmp_path, capsys):
    import plugins.memory.agentmemory.cli as agentmemory_cli

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(agentmemory_cli, "_config_path", lambda: config_path)

    agentmemory_cli.agentmemory_command(SimpleNamespace(agentmemory_command="provider"))

    out = capsys.readouterr().out
    assert "Memory provider: agentmemory" in out
    assert "provider: agentmemory" in config_path.read_text()


def test_viewer_url_uses_agentmemory_host(monkeypatch):
    import plugins.memory.agentmemory.cli as agentmemory_cli

    monkeypatch.delenv("AGENTMEMORY_VIEWER_URL", raising=False)

    assert agentmemory_cli._viewer_url("http://127.0.0.1:65530") == "http://127.0.0.1:3113"


def test_status_reports_server_and_viewer(monkeypatch, tmp_path, capsys):
    import plugins.memory.agentmemory.cli as agentmemory_cli

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "mcp_servers:\n"
        "  agentmemory:\n"
        "    command: npx\n"
        "    args: ['-y', '@agentmemory/mcp']\n"
        "memory:\n"
        "  provider: agentmemory\n"
    )
    monkeypatch.setattr(agentmemory_cli, "_config_path", lambda: config_path)
    monkeypatch.setattr(agentmemory_cli.shutil, "which", lambda name: "/usr/bin/npx" if name == "npx" else None)
    monkeypatch.setattr(agentmemory_cli, "_get_json", lambda path, base_url=None: {"status": "healthy"} if path == "health" else {"flags": []})
    monkeypatch.setattr(agentmemory_cli, "_viewer_reachable", lambda base_url=None: True)

    agentmemory_cli.agentmemory_command(SimpleNamespace(agentmemory_command="status", url=None))

    out = capsys.readouterr().out
    assert "Server... OK" in out
    assert "Viewer... OK" in out
    assert "MCP config... OK" in out
    assert "Provider... OK" in out


def test_disable_only_clears_agentmemory_provider(monkeypatch, tmp_path, capsys):
    import plugins.memory.agentmemory.cli as agentmemory_cli

    config_path = tmp_path / "config.yaml"
    config_path.write_text("memory:\n  provider: agentmemory\n")
    monkeypatch.setattr(agentmemory_cli, "_config_path", lambda: config_path)

    agentmemory_cli.agentmemory_command(SimpleNamespace(agentmemory_command="disable"))

    out = capsys.readouterr().out
    assert "AgentMemory provider disabled" in out
    assert "provider: ''" in config_path.read_text()
