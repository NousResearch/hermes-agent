import json
from unittest.mock import patch

from gateway.session_context import clear_session_vars, set_session_vars
from plugins.platforms.slack.tools import _handle_current_message


class _Adapter:
    def __init__(self):
        self.calls = []

    async def react_to_current_message(self, **kwargs):
        self.calls.append(("react", kwargs))
        return True

    async def current_message_permalink(self, **kwargs):
        self.calls.append(("permalink", kwargs))
        return "https://workspace.slack.com/archives/C1/p123"


def _bound_slack():
    return set_session_vars(
        platform="slack", chat_id="C1", thread_id="111.0001",
        message_id="123.0002", scope_id="T1",
    )


def test_react_uses_only_current_context_identifiers():
    adapter = _Adapter()
    tokens = _bound_slack()
    try:
        with patch("plugins.platforms.slack.tools._current_slack_adapter", return_value=adapter):
            result = json.loads(_handle_current_message({"action": "react", "emoji": ":beer:"}))
    finally:
        clear_session_vars(tokens)
    assert result == {"success": True, "action": "react", "emoji": "beer"}
    assert adapter.calls == [("react", {
        "channel": "C1", "timestamp": "123.0002", "emoji": "beer", "team_id": "T1",
    })]


def test_permalink_uses_current_reply_not_thread_root():
    adapter = _Adapter()
    tokens = _bound_slack()
    try:
        with patch("plugins.platforms.slack.tools._current_slack_adapter", return_value=adapter):
            result = json.loads(_handle_current_message({"action": "permalink"}))
    finally:
        clear_session_vars(tokens)
    assert result == {
        "success": True, "action": "permalink",
        "permalink": "https://workspace.slack.com/archives/C1/p123",
    }
    assert adapter.calls == [("permalink", {
        "channel": "C1", "timestamp": "123.0002", "team_id": "T1",
    })]


def test_rejects_non_slack_or_missing_bound_context():
    tokens = set_session_vars(platform="discord", chat_id="C1", message_id="123", scope_id="T1")
    try:
        result = json.loads(_handle_current_message({"action": "permalink"}))
    finally:
        clear_session_vars(tokens)
    assert result["success"] is False
    assert "Slack" in result["error"]


def test_rejects_model_controlled_target_fields_at_schema_boundary():
    # The schema has additionalProperties:false, and the handler never reads a target.
    adapter = _Adapter()
    tokens = _bound_slack()
    try:
        with patch("plugins.platforms.slack.tools._current_slack_adapter", return_value=adapter):
            result = json.loads(_handle_current_message({
                "action": "react", "emoji": "beer", "target": "slack:C-ATTACKER:999",
            }))
    finally:
        clear_session_vars(tokens)
    assert result["success"] is True
    assert adapter.calls[0][1]["channel"] == "C1"
    assert adapter.calls[0][1]["timestamp"] == "123.0002"
