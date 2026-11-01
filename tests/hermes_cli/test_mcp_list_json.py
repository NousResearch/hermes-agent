"""`hermes mcp list --json` — machine-readable MCP server listing.

With --json, mcp list prints a JSON array of server objects with name,
transport, url, command, args, and enabled keys. Empty fleet prints [].
"""

import json
import types

import pytest


@pytest.fixture()
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _run_list(tmp_home, monkeypatch, capsys, *, servers=None):
    from hermes_cli import mcp_config as mod

    if servers is None:
        servers = {}
    monkeypatch.setattr(mod, "_get_mcp_servers", lambda config=None: servers)
    args = types.SimpleNamespace(json_output=True)
    mod.cmd_mcp_list(args)
    return capsys.readouterr().out


def test_empty_returns_empty_array(tmp_home, monkeypatch, capsys):
    out = _run_list(tmp_home, monkeypatch, capsys, servers={})
    doc = json.loads(out)
    assert doc == []


def test_stdio_server(tmp_home, monkeypatch, capsys):
    servers = {"myserver": {"command": "npx", "args": ["-y", "mcp-server"]}}
    out = _run_list(tmp_home, monkeypatch, capsys, servers=servers)
    doc = json.loads(out)
    assert len(doc) == 1
    row = doc[0]
    assert row["name"] == "myserver"
    assert row["transport"] == "stdio"
    assert row["command"] == "npx"
    assert row["args"] == ["-y", "mcp-server"]
    assert row["url"] is None


def test_http_server(tmp_home, monkeypatch, capsys):
    servers = {"remote": {"url": "https://mcp.example.com/v1"}}
    out = _run_list(tmp_home, monkeypatch, capsys, servers=servers)
    doc = json.loads(out)
    row = doc[0]
    assert row["name"] == "remote"
    assert row["transport"] == "streamable_http"
    assert row["url"] == "https://mcp.example.com/v1"
    assert row["command"] is None


def test_disabled_server(tmp_home, monkeypatch, capsys):
    servers = {"off": {"url": "https://x.example.com", "enabled": False}}
    out = _run_list(tmp_home, monkeypatch, capsys, servers=servers)
    doc = json.loads(out)
    assert doc[0]["enabled"] is False
