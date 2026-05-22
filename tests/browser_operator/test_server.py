import json

from browser_operator.server import create_mcp_server, tool_finish, tool_start_session


def test_start_session_returns_ephemeral_policy_metadata():
    result = json.loads(tool_start_session(goal="open example", fresh=True))

    assert result["success"] is True
    assert result["session_id"].startswith("bo_")
    assert result["policy"]["session_mode"] == "ephemeral"
    assert result["policy"]["approval_mode"] == "none"
    assert result["policy"]["secrets_revealed_to_model"] is False


def test_finish_closes_session_and_returns_chat_summary(monkeypatch):
    closed = []

    monkeypatch.setattr("browser_operator.server._cleanup_browser", lambda session_id: closed.append(session_id))

    result = json.loads(tool_finish("bo_test", summary="Downloaded invoice", success=True))

    assert result == {
        "success": True,
        "session_id": "bo_test",
        "summary": "Downloaded invoice",
        "closed": True,
    }
    assert closed == ["bo_test"]


def test_create_mcp_server_exposes_core_browser_operator_tools():
    server = create_mcp_server()
    tool_names = {tool.name for tool in server._tool_manager.list_tools()}

    assert {
        "browser_start_session",
        "browser_open_url",
        "browser_observe",
        "browser_click",
        "browser_type_text",
        "browser_fill_login_from_1password",
        "browser_fill_totp_from_1password",
        "browser_finish",
    }.issubset(tool_names)
