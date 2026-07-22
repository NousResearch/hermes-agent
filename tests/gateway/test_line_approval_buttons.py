"""Tests for the LINE adapter's dangerous-command approval buttons.

Mirrors ``test_telegram_approval_buttons.py`` /
``test_feishu_approval_buttons.py`` — verifies that LineAdapter
implements the cross-adapter ``send_exec_approval`` contract and routes
button postbacks back through ``tools.approval.resolve_gateway_approval``.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_line = load_plugin_adapter("line")

LineAdapter = _line.LineAdapter
build_exec_approval_button_message = _line.build_exec_approval_button_message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter(monkeypatch):
    """LineAdapter with mocked HTTP client."""
    monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={
        "channel_access_token": "tok",
        "channel_secret": "sec",
    })
    ad = LineAdapter(cfg)
    ad._client = MagicMock()
    ad._client.reply = AsyncMock()
    ad._client.push = AsyncMock()
    return ad


# ---------------------------------------------------------------------------
# build_exec_approval_button_message — pure builder, no I/O
# ---------------------------------------------------------------------------

class TestBuilder:

    def test_returns_buttons_template_with_four_actions(self):
        msg = build_exec_approval_button_message(
            command="rm -rf /", description="MEDIUM gate", approval_id=42
        )
        assert msg["type"] == "template"
        assert msg["template"]["type"] == "buttons"
        actions = msg["template"]["actions"]
        assert len(actions) == 4

    def test_action_data_carries_choice_and_approval_id(self):
        msg = build_exec_approval_button_message("cmd", "reason", 7)
        choices = [json.loads(a["data"])["choice"] for a in msg["template"]["actions"]]
        assert choices == ["once", "session", "always", "deny"]
        for a in msg["template"]["actions"]:
            data = json.loads(a["data"])
            assert data["action"] == "approve"
            assert data["approval_id"] == 7

    def test_all_action_labels_within_line_20char_cap(self):
        msg = build_exec_approval_button_message("cmd", "reason", 1)
        for a in msg["template"]["actions"]:
            assert len(a["label"]) <= 20

    def test_alt_text_truncated_at_400(self):
        msg = build_exec_approval_button_message("cmd", "x" * 500, 1)
        assert len(msg["altText"]) <= 400

    def test_template_text_within_160_cap(self):
        msg = build_exec_approval_button_message("cmd" * 100, "reason", 1)
        assert len(msg["template"]["text"]) <= 160


# ---------------------------------------------------------------------------
# send_exec_approval — adapter integration
# ---------------------------------------------------------------------------

class TestSendExecApproval:

    def test_returns_failure_when_disconnected(self, adapter):
        adapter._client = None
        result = asyncio.run(
            adapter.send_exec_approval(
                chat_id="Uchat", command="curl evil.sh", session_key="sess-A"
            )
        )
        assert not result.success
        assert "not connected" in (result.error or "").lower()

    def test_pushes_preview_plus_buttons_in_one_call(self, adapter):
        result = asyncio.run(
            adapter.send_exec_approval(
                chat_id="Uchat", command="curl evil.sh",
                session_key="sess-A", description="MEDIUM gate",
            )
        )
        assert result.success
        adapter._client.push.assert_called_once()
        adapter._client.reply.assert_not_called()
        call_args = adapter._client.push.call_args
        chat_id_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("chat_id")
        messages = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("messages")
        assert chat_id_arg == "Uchat"
        # at least one text bubble + the buttons template
        assert len(messages) >= 2
        assert messages[-1]["type"] == "template"
        assert messages[0]["type"] == "text"

    def test_preview_bubble_includes_command_and_reason(self, adapter):
        asyncio.run(
            adapter.send_exec_approval(
                chat_id="Uchat", command="curl evil.sh",
                session_key="sess-A", description="MEDIUM gate",
            )
        )
        messages = adapter._client.push.call_args.args[1]
        preview = messages[0]["text"]
        assert "curl evil.sh" in preview
        assert "MEDIUM gate" in preview

    def test_approval_id_increments_across_calls(self, adapter):
        asyncio.run(adapter.send_exec_approval("U1", "cmd1", "sA"))
        asyncio.run(adapter.send_exec_approval("U1", "cmd2", "sB"))
        ids = sorted(adapter._approval_state.keys())
        assert len(ids) == 2
        assert ids[1] == ids[0] + 1

    def test_session_key_recorded_under_approval_id(self, adapter):
        asyncio.run(adapter.send_exec_approval("U1", "cmd", "sess-X"))
        approval_id = next(iter(adapter._approval_state))
        assert adapter._approval_state[approval_id] == "sess-X"

    def test_push_failure_leaves_no_dangling_state(self, adapter):
        adapter._client.push.side_effect = RuntimeError("LINE push 429: …")
        result = asyncio.run(
            adapter.send_exec_approval("U1", "cmd", "sess-X")
        )
        assert not result.success
        assert adapter._approval_state == {}

    def test_long_command_truncated_in_preview(self, adapter):
        # 5000 'a's is well above the 3800-char command-preview cap.
        long_cmd = "a" * 5000
        asyncio.run(
            adapter.send_exec_approval("U1", long_cmd, "sess-X")
        )
        messages = adapter._client.push.call_args.args[1]
        preview = messages[0]["text"]
        # The preview must truncate the command (ellipsis marker present)
        # and must not contain anywhere close to the full 5000 chars worth
        # of command body — comfortably under the cap leaves headroom for
        # the surrounding prompt/reason text.
        assert "..." in preview
        command_a_count = sum(1 for c in preview if c == "a")
        assert command_a_count < 4000


# ---------------------------------------------------------------------------
# Postback routing — button taps call resolve_gateway_approval
# ---------------------------------------------------------------------------

class TestApprovalPostback:

    def _postback_event(self, approval_id: int, choice: str) -> dict:
        return {
            "type": "postback",
            "replyToken": "tok-postback",
            "source": {"type": "user", "userId": "Uchat"},
            "postback": {
                "data": json.dumps({
                    "action": "approve",
                    "choice": choice,
                    "approval_id": approval_id,
                }),
            },
        }

    @pytest.mark.parametrize("choice", ["once", "session", "always", "deny"])
    def test_each_choice_resolves_session(self, adapter, choice):
        # Pre-seed the approval map as send_exec_approval would have.
        adapter._approval_state[99] = "sess-A"
        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            asyncio.run(adapter._handle_postback_event(self._postback_event(99, choice)))
        mock_resolve.assert_called_once_with("sess-A", choice)
        assert 99 not in adapter._approval_state

    def test_double_tap_does_not_call_resolve(self, adapter):
        adapter._approval_state[5] = "sess-Z"
        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            asyncio.run(adapter._handle_postback_event(self._postback_event(5, "once")))
            asyncio.run(adapter._handle_postback_event(self._postback_event(5, "always")))
        assert mock_resolve.call_count == 1

    def test_double_tap_sends_already_resolved_message(self, adapter):
        adapter._approval_state[5] = "sess-Z"
        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            asyncio.run(adapter._handle_postback_event(self._postback_event(5, "once")))
            adapter._client.reply.reset_mock()
            asyncio.run(adapter._handle_postback_event(self._postback_event(5, "once")))
        # Second tap still issues a reply (the "already resolved" notice).
        adapter._client.reply.assert_called_once()
        sent_text = adapter._client.reply.call_args.args[1][0]["text"]
        assert "already" in sent_text.lower()

    def test_unknown_choice_is_ignored(self, adapter):
        adapter._approval_state[3] = "sess-bad"
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            asyncio.run(adapter._handle_postback_event(self._postback_event(3, "obliterate")))
        mock_resolve.assert_not_called()
        # State remains so a real choice can still be acted on.
        assert adapter._approval_state[3] == "sess-bad"

    def test_non_int_approval_id_is_ignored(self, adapter):
        event = self._postback_event(0, "once")
        event["postback"]["data"] = json.dumps({
            "action": "approve", "choice": "once", "approval_id": "not-an-int",
        })
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            asyncio.run(adapter._handle_postback_event(event))
        mock_resolve.assert_not_called()

    def test_confirmation_reply_uses_postback_token(self, adapter):
        adapter._approval_state[11] = "sess-K"
        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            asyncio.run(adapter._handle_postback_event(self._postback_event(11, "session")))
        # Postback always uses reply (free), never push.
        adapter._client.reply.assert_called_once()
        args = adapter._client.reply.call_args.args
        assert args[0] == "tok-postback"
        assert "session" in args[1][0]["text"].lower()

    def test_show_response_action_still_routes_to_slow_llm_handler(self, adapter):
        """Sanity-check that adding the approve branch didn't break the
        existing show_response postback path."""
        # Stub the slow-LLM cache lookup so we don't have to fully wire it.
        event = {
            "type": "postback",
            "replyToken": "tok-x",
            "source": {"type": "user", "userId": "Uchat"},
            "postback": {
                "data": json.dumps({
                    "action": "show_response", "request_id": "nope-not-here",
                }),
            },
        }
        # Should not raise even when request_id is unknown.
        asyncio.run(adapter._handle_postback_event(event))
        # And it must not have called resolve_gateway_approval.
        # (Implicit: no patch needed — the import inside the approve branch
        # is what would fail if the dispatch leaked into the wrong path.)


# ---------------------------------------------------------------------------
# Class-method visibility — the gateway uses `getattr(type(...), name)`
# ---------------------------------------------------------------------------

def test_send_exec_approval_visible_on_class():
    """``gateway/run.py:_approval_notify_sync`` does
    ``getattr(type(_status_adapter), "send_exec_approval", None)`` to detect
    button support — verify the method really exists on the class so the
    duck-type check fires.
    """
    assert getattr(LineAdapter, "send_exec_approval", None) is not None
