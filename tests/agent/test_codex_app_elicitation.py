import pytest

from agent.transports.codex_app_server_session import CodexAppServerSession
from gateway.session_context import clear_session_vars, set_session_vars
from tools import approval
from tools.approval_context import reset_current_session_key, set_current_session_key


@pytest.mark.parametrize(("choice", "expected"), [("once", "accept"), ("deny", "decline")])
def test_app_tool_approval_reaches_telegram(choice, expected):
    session_key = "codex-app-approval"
    seen = []

    def notify(data):
        seen.append(data)
        assert approval.resolve_gateway_approval(session_key, choice, request_id=data["request_id"]) == 1

    tokens = set_session_vars(platform="telegram", session_key=session_key, async_delivery=False)
    key_token = set_current_session_key(session_key)
    approval.register_gateway_notify(session_key, notify)
    try:
        result = CodexAppServerSession()._respond_elicitation({
            "serverName": "codex_apps", "mode": "form", "message": "Approve this event?",
            "_meta": {"codex_approval_kind": "mcp_tool_call", "tool_title": "Create event",
                      "tool_params": {"title": "Walk", "start_time": "12:00"}},
        })
    finally:
        approval.unregister_gateway_notify(session_key)
        reset_current_session_key(key_token)
        clear_session_vars(tokens)

    assert result["action"] == expected
    assert len(seen) == 1
    assert "Walk" in seen[0]["description"]
    assert "12:00" in seen[0]["description"]


def test_unsupported_elicitation_is_declined():
    result = CodexAppServerSession()._respond_elicitation({
        "serverName": "codex_apps", "mode": "url", "message": "Visit a website",
    })
    assert result["action"] == "decline"
