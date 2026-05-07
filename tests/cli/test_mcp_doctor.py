"""Tests for `hermes mcp doctor` CLI diagnostics rendering."""

import argparse
import json


def _args(**kwargs):
    defaults = {
        "mcp_action": "doctor",
        "name": None,
        "refresh": False,
        "json": False,
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_mcp_doctor_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "tools.mcp_tool.get_mcp_diagnostics",
        lambda name=None, refresh=False: [{
            "name": name or "srv",
            "configured": True,
            "enabled": True,
            "attempted": refresh,
            "connected": False,
            "transport": "stdio → npx",
            "tool_count": 0,
            "registered_tool_count": 0,
            "raw_tool_count": 0,
            "needs_auth": False,
            "tools_list_failed": False,
            "process_or_http_failure": False,
            "last_error": "",
            "next_action": "Run refresh.",
        }],
    )
    from hermes_cli.mcp_config import cmd_mcp_doctor

    cmd_mcp_doctor(_args(name="srv", refresh=True, json=True))
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data[0]["name"] == "srv"
    assert data[0]["attempted"] is True


def test_mcp_doctor_text_includes_status_and_next(monkeypatch, capsys):
    monkeypatch.setattr(
        "tools.mcp_tool.get_mcp_diagnostics",
        lambda name=None, refresh=False: [{
            "name": "remote",
            "configured": True,
            "enabled": True,
            "attempted": True,
            "connected": False,
            "transport": "http → https://example.com/mcp",
            "tool_count": 0,
            "registered_tool_count": 0,
            "raw_tool_count": 0,
            "needs_auth": True,
            "tools_list_failed": False,
            "process_or_http_failure": False,
            "last_error": "401 Unauthorized",
            "next_action": "Check auth headers/env vars for `remote`.",
        }],
    )
    from hermes_cli.mcp_config import cmd_mcp_doctor

    cmd_mcp_doctor(_args())
    out = capsys.readouterr().out
    assert "MCP diagnostics" in out
    assert "remote: needs auth" in out
    assert "401 Unauthorized" in out
    assert "Check auth" in out


def test_mcp_dispatcher_routes_doctor(monkeypatch, capsys):
    called = {}
    monkeypatch.setattr(
        "hermes_cli.mcp_config.cmd_mcp_doctor",
        lambda args: called.setdefault("action", args.mcp_action),
    )
    from hermes_cli.mcp_config import mcp_command

    mcp_command(_args(mcp_action="doctor"))

    assert called["action"] == "doctor"
