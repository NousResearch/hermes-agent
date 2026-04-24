import json

from gateway.session_context import clear_session_vars, set_session_vars
from tools.send_message_tool import send_message_tool


def test_send_message_blocks_list_when_gateway_user_id_missing():
    tokens = set_session_vars(platform="discord", chat_id="dm-123", user_id="")
    try:
        raw = send_message_tool({"action": "list"})
        data = json.loads(raw)
        assert "error" in data
        assert "no authoritative numeric user_id" in data["error"]
    finally:
        clear_session_vars(tokens)


def test_send_message_blocks_send_when_gateway_user_id_missing():
    tokens = set_session_vars(platform="discord", chat_id="dm-123", user_id="")
    try:
        raw = send_message_tool({"action": "send", "target": "discord:123", "message": "hi"})
        data = json.loads(raw)
        assert "error" in data
        assert "no authoritative numeric user_id" in data["error"]
    finally:
        clear_session_vars(tokens)


def test_send_message_identity_guard_allows_local_context(monkeypatch):
    clear_session_vars([])
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "local")
    raw = send_message_tool({"action": "send", "target": "", "message": "hi"})
    data = json.loads(raw)
    # The identity guard did not fire; normal argument validation did.
    assert "Both 'target' and 'message' are required" in data.get("error", "")
