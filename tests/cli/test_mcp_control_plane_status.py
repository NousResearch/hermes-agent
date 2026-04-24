"""Wave 7 MCP control-plane status tests."""

from types import SimpleNamespace
from unittest.mock import patch


def test_mcp_control_plane_status_reports_config_connected_tools_and_auth_state():
    from tools import mcp_tool

    server = SimpleNamespace(
        name="alpha",
        session=object(),
        _tools=[SimpleNamespace(name="raw_tool")],
        _registered_tool_names=["alpha_raw_tool", "alpha_list_resources"],
        _sampling=None,
    )

    configured = {
        "alpha": {"url": "https://example.invalid/mcp", "oauth": True},
        "beta": {"command": "python", "args": ["server.py"], "enabled": False},
    }

    with (
        patch.object(mcp_tool, "_load_mcp_config", return_value=configured),
        patch.object(mcp_tool, "_servers", {"alpha": server}),
    ):
        status = mcp_tool.get_mcp_control_plane_status()

    assert status["summary"] == {
        "configured": 2,
        "enabled": 1,
        "connected": 1,
        "tools": 2,
    }
    alpha = next(item for item in status["servers"] if item["name"] == "alpha")
    beta = next(item for item in status["servers"] if item["name"] == "beta")

    assert alpha["connected"] is True
    assert alpha["auth"]["required"] is True
    assert alpha["auth"]["state"] in {"unknown", "authenticated", "expired"}
    assert alpha["tools"] == 2
    assert beta["enabled"] is False
    assert beta["connected"] is False
