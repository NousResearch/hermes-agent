import pytest

from agent.transports.codex_app_server_session import CodexAppServerSession
from gateway.session_context import clear_session_vars, set_session_vars
from tools import approval
from tools.approval_context import reset_current_session_key, set_current_session_key

SESSION_KEY = "codex-app-approval"


def _tool_call(title="Create event"):
    return {
        "serverName": "codex_apps", "mode": "form", "message": "Approve this event?",
        "_meta": {"codex_approval_kind": "mcp_tool_call", "tool_title": title,
                  "tool_params": {"title": "Walk", "start_time": "12:00"}},
    }


@pytest.fixture
def gateway_session():
    """A gateway session whose notify callback answers each prompt with the queued choice."""
    choices, seen = [], []

    def notify(data):
        seen.append(data)
        assert approval.resolve_gateway_approval(SESSION_KEY, choices.pop(0), request_id=data["request_id"]) == 1

    tokens = set_session_vars(platform="telegram", session_key=SESSION_KEY, async_delivery=False)
    key_token = set_current_session_key(SESSION_KEY)
    approval.register_gateway_notify(SESSION_KEY, notify)
    try:
        yield choices, seen
    finally:
        approval.unregister_gateway_notify(SESSION_KEY)
        approval.clear_session(SESSION_KEY)
        reset_current_session_key(key_token)
        clear_session_vars(tokens)


@pytest.mark.parametrize(("choice", "expected"), [("once", "accept"), ("deny", "decline")])
def test_mcp_tool_call_approval_routes_to_gateway(gateway_session, choice, expected):
    choices, seen = gateway_session
    choices.append(choice)

    result = CodexAppServerSession()._respond_elicitation(_tool_call())

    assert result["action"] == expected
    assert len(seen) == 1
    assert "Walk" in seen[0]["description"]
    assert "12:00" in seen[0]["description"]


def test_session_approval_covers_later_calls_to_the_same_tool(gateway_session):
    choices, seen = gateway_session
    choices.append("session")
    session = CodexAppServerSession()

    assert session._respond_elicitation(_tool_call())["action"] == "accept"
    assert session._respond_elicitation(_tool_call())["action"] == "accept"
    assert len(seen) == 1

    choices.append("deny")
    assert session._respond_elicitation(_tool_call("Delete event"))["action"] == "decline"
    assert len(seen) == 2


def test_approval_bypass_accepts_without_prompting(gateway_session):
    _, seen = gateway_session
    session = CodexAppServerSession()
    approval.enable_session_yolo(SESSION_KEY)

    assert session._respond_elicitation(_tool_call())["action"] == "accept"
    assert seen == []


def test_turning_yolo_off_mid_session_prompts_again(gateway_session):
    choices, seen = gateway_session
    approval.enable_session_yolo(SESSION_KEY)
    session = CodexAppServerSession()
    assert session._respond_elicitation(_tool_call())["action"] == "accept"

    approval.disable_session_yolo(SESSION_KEY)
    choices.append("deny")
    assert session._respond_elicitation(_tool_call())["action"] == "decline"
    assert len(seen) == 1


def test_prompt_masks_secret_arguments(gateway_session):
    choices, seen = gateway_session
    choices.append("deny")
    call = _tool_call()
    call["_meta"]["tool_params"] = {"title": "Walk", "api_key": "sk-live-abcdef0123456789abcdef"}

    CodexAppServerSession()._respond_elicitation(call)

    assert "Walk" in seen[0]["description"]
    assert "sk-live-abcdef0123456789abcdef" not in seen[0]["description"]


def test_unsupported_elicitation_is_declined():
    result = CodexAppServerSession()._respond_elicitation({
        "serverName": "codex_apps", "mode": "url", "message": "Visit a website",
    })
    assert result["action"] == "decline"
