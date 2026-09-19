"""Tests for WeCom template_card approval button security.

Verifies that _handle_template_card_event correctly:
1. Accepts legitimate clicks from the expected admin chat + user
2. Rejects clicks from a wrong chat_id (forwarded card to another chat)
3. Rejects clicks from a wrong user_id (unauthorized user in same chat)
4. Preserves the task entry when rejecting (so the real admin can still approve)
5. Cleans up the task entry after a successful click
6. Handles expired/missing tasks gracefully
7. send_exec_approval routes group chats via passive reply (req_id) and DMs
   via APP_CMD_SEND
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.wecom.adapter import WeComAdapter

# Enable asyncio support for all async test methods in this file
pytestmark = pytest.mark.asyncio


def _make_adapter():
    """Create a WeComAdapter without calling __init__ (bypasses WS/HTTP setup)."""
    adapter = WeComAdapter.__new__(WeComAdapter)
    adapter._approval_tasks = {}
    adapter._APPROVAL_TASK_TTL = 600.0
    adapter._group_chat_ids = set()
    adapter._last_chat_req_ids = {}
    # build_source needs self.platform; mock it to return a simple stand-in
    adapter.platform = MagicMock()
    adapter.platform.value = "wecom"
    adapter.build_source = MagicMock(return_value=MagicMock())
    # handle_message is the base-class dispatch; mock to capture the event
    adapter.handle_message = AsyncMock()
    return adapter


def _card_payload(body_extra=None, *, task_id="task_123"):
    """Build a template_card_event payload mimicking WeCom websocket."""
    body = {
        "msgtype": "template_card_event",
        "chatid": "chat_456",
        "chattype": "dm",
        "from": {"userid": "user_789"},
        "template_card_event": {
            "event_key": "approve",
            "task_id": task_id,
        },
    }
    if body_extra:
        body.update(body_extra)
    return {"body": body}


class TestTemplateCardSecurity:
    """Security tests for _handle_template_card_event."""

    async def test_legitimate_click_dispatches_approve(self):
        adapter = _make_adapter()
        adapter._approval_tasks["task_123"] = (
            "sess_1", "chat_456", "user_789", time.monotonic(),
        )
        await adapter._handle_template_card_event(
            _card_payload()["body"], _card_payload()
        )
        # handle_message should be called with a /approve text event
        assert adapter.handle_message.called
        event = adapter.handle_message.call_args[0][0]
        assert event.text == "/approve"
        # Task cleaned up after success
        assert "task_123" not in adapter._approval_tasks

    async def test_legitimate_deny_click(self):
        adapter = _make_adapter()
        adapter._approval_tasks["task_123"] = (
            "sess_1", "chat_456", "user_789", time.monotonic(),
        )
        payload = _card_payload(
            {"template_card_event": {"event_key": "deny", "task_id": "task_123"}}
        )
        await adapter._handle_template_card_event(payload["body"], payload)
        assert adapter.handle_message.called
        assert adapter.handle_message.call_args[0][0].text == "/deny"

    async def test_chat_id_mismatch_rejected(self):
        adapter = _make_adapter()
        adapter._approval_tasks["task_123"] = (
            "sess_1", "chat_456", "user_789", time.monotonic(),
        )
        # Forwarded card into a different chat
        payload = _card_payload({"chatid": "chat_attacker"})
        await adapter._handle_template_card_event(payload["body"], payload)
        assert not adapter.handle_message.called
        # Task preserved — real admin can still approve
        assert "task_123" in adapter._approval_tasks

    async def test_user_id_mismatch_rejected(self):
        adapter = _make_adapter()
        adapter._approval_tasks["task_123"] = (
            "sess_1", "chat_456", "user_789", time.monotonic(),
        )
        # Another member of the same chat clicks
        payload = _card_payload({"from": {"userid": "user_attacker"}})
        await adapter._handle_template_card_event(payload["body"], payload)
        assert not adapter.handle_message.called
        assert "task_123" in adapter._approval_tasks

    async def test_expired_task_still_dispatches(self):
        adapter = _make_adapter()
        # Entry stored 100s ago (TTL is 600s, so not yet expired by TTL but
        # no strict expiry check in handler — main gate is chat/user match)
        adapter._approval_tasks["task_123"] = (
            "sess_1", "chat_456", "user_789", time.monotonic() - 100,
        )
        await adapter._handle_template_card_event(
            _card_payload()["body"], _card_payload()
        )
        assert adapter.handle_message.called

    async def test_unknown_event_key_preserves_task(self):
        """Unknown event_key must NOT clean up the task — an attacker could
        forward the admin's card and use an invalid key to delete the real
        task before identity validation. Task preservation is fail-closed."""
        adapter = _make_adapter()
        adapter._approval_tasks["task_123"] = (
            "sess_1", "chat_456", "user_789", time.monotonic(),
        )
        payload = _card_payload(
            {"template_card_event": {"event_key": "hack", "task_id": "task_123"}}
        )
        await adapter._handle_template_card_event(payload["body"], payload)
        assert not adapter.handle_message.called
        # Task preserved — admin can still approve (fail-closed)
        assert "task_123" in adapter._approval_tasks

    async def test_approve_session_synthesises_correct_command(self):
        adapter = _make_adapter()
        adapter._approval_tasks["task_123"] = (
            "sess_1", "chat_456", "user_789", time.monotonic(),
        )
        payload = _card_payload(
            {"template_card_event": {"event_key": "approve_session", "task_id": "task_123"}}
        )
        await adapter._handle_template_card_event(payload["body"], payload)
        assert adapter.handle_message.call_args[0][0].text == "/approve session"

    async def test_approve_always_synthesises_correct_command(self):
        adapter = _make_adapter()
        adapter._approval_tasks["task_123"] = (
            "sess_1", "chat_456", "user_789", time.monotonic(),
        )
        payload = _card_payload(
            {"template_card_event": {"event_key": "approve_always", "task_id": "task_123"}}
        )
        await adapter._handle_template_card_event(payload["body"], payload)
        assert adapter.handle_message.call_args[0][0].text == "/approve always"

    async def test_missing_card_event_data_returns_silently(self):
        adapter = _make_adapter()
        await adapter._handle_template_card_event(
            {"msgtype": "template_card_event"}, {}
        )
        assert not adapter.handle_message.called

    async def test_unknown_task_id_does_not_dispatch(self):
        """Attack: attacker guesses/forwards a task_id that is NOT in the
        approval store. With stored=None the chat/user validation is skipped —
        verify the command is NOT dispatched (no unvalidated /approve)."""
        adapter = _make_adapter()
        # No task stored — attacker sends arbitrary task_id
        payload = _card_payload(task_id="task_attacker_guess")
        await adapter._handle_template_card_event(payload["body"], payload)
        assert not adapter.handle_message.called

    async def test_attacker_forwarded_card_invalid_key_preserves_task(self):
        """Attack: group member forwards the admin's card to their own chat and
        clicks with an INVALID event_key. The event_key whitelist check pops the
        task BEFORE chat/user validation — verify the real task survives so the
        admin can still approve."""
        adapter = _make_adapter()
        adapter._approval_tasks["task_legit"] = (
            "sess_1", "chat_admin_group", "user_admin", time.monotonic(),
        )
        # Attacker forwards card to their own chat, clicks with unknown key
        payload = _card_payload(
            {
                "chatid": "chat_attacker",
                "chattype": "group",
                "from": {"userid": "user_attacker"},
                "template_card_event": {"event_key": "hack", "task_id": "task_legit"},
            }
        )
        await adapter._handle_template_card_event(payload["body"], payload)
        assert not adapter.handle_message.called
        # Real task must be preserved — admin can still approve
        assert "task_legit" in adapter._approval_tasks


class TestSendExecApproval:
    """Tests for the send_exec_approval delivery path."""

    async def test_dm_uses_APP_CMD_SEND(self):
        adapter = _make_adapter()
        adapter._send_request = AsyncMock(return_value={"errcode": 0})
        result = await adapter.send_exec_approval(
            chat_id="chat_456",
            command="rm -rf /",
            session_key="sess_1",
            description="dangerous",
            admin_user_id="user_789",
        )
        assert result.success
        # APP_CMD_SEND used for DM
        sent_body = adapter._send_request.call_args[0][1]
        assert sent_body["chatid"] == "chat_456"
        assert sent_body["msgtype"] == "template_card"
        # Task stored for callback validation
        assert adapter._approval_tasks
        _, _, uid, _ = next(iter(adapter._approval_tasks.values()))
        assert uid == "user_789"

    async def test_group_chat_uses_passive_reply(self):
        adapter = _make_adapter()
        adapter._group_chat_ids = {"chat_group"}
        adapter._last_chat_req_ids = {"chat_group": "req_abc"}
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})
        adapter._send_request = AsyncMock()
        result = await adapter.send_exec_approval(
            chat_id="chat_group",
            command="rm -rf /",
            session_key="sess_1",
            description="dangerous",
            admin_user_id="user_789",
        )
        assert result.success
        # Passive reply used for group (APP_CMD_SEND blocked in groups)
        reply_req_id, body = adapter._send_reply_request.call_args[0]
        assert reply_req_id == "req_abc"
        assert body["msgtype"] == "template_card"
        assert not adapter._send_request.called

    async def test_group_chat_no_req_id_fails_cleanly(self):
        adapter = _make_adapter()
        adapter._group_chat_ids = {"chat_group"}
        adapter._last_chat_req_ids = {}
        result = await adapter.send_exec_approval(
            chat_id="chat_group",
            command="rm -rf /",
            session_key="sess_1",
        )
        assert not result.success
        # No orphaned task entry
        assert not adapter._approval_tasks

    async def test_send_error_cleans_up_task(self):
        adapter = _make_adapter()
        adapter._send_request = AsyncMock(return_value={"errcode": 846600, "errmsg": "fail"})
        result = await adapter.send_exec_approval(
            chat_id="chat_456",
            command="rm -rf /",
            session_key="sess_1",
        )
        assert not result.success
        assert not adapter._approval_tasks


@pytest.mark.asyncio
async def test_on_message_routes_template_card_event():
    """_on_message intercepts template_card_event before text pipeline."""
    adapter = _make_adapter()
    adapter._approval_tasks["task_123"] = (
        "sess_1", "chat_456", "user_789", time.monotonic(),
    )
    payload = _card_payload()
    await adapter._on_message(payload)
    assert adapter.handle_message.called
    assert adapter.handle_message.call_args[0][0].text == "/approve"
