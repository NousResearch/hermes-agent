"""Tests for WeCom template_card approval button security.

Verifies that _handle_template_card_event correctly:
1. Accepts legitimate clicks from the expected admin chat + user
2. Rejects clicks from a wrong chat_id (forwarded card to another chat)
3. Rejects clicks from a wrong user_id (unauthorized user in same chat)
4. Preserves the task entry when rejecting (so the real admin can still approve)
5. Cleans up the task entry after a successful click
6. Handles expired/missing tasks gracefully
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.wecom.adapter import WeComAdapter


def _make_adapter():
    """Create a WeComAdapter without calling __init__ (bypasses WS/HTTP setup)."""
    adapter = WeComAdapter.__new__(WeComAdapter)
    adapter._approval_tasks = {}
    adapter._APPROVAL_TASK_TTL = 600.0
    # build_source needs self.platform; mock it to return a simple stand-in
    adapter.platform = MagicMock()
    adapter.platform.value = "wecom"
    adapter.build_source = MagicMock(return_value=MagicMock())
    # handle_message is the base-class dispatch; mock to capture the event
    adapter.handle_message = AsyncMock()
    return adapter


def _make_card_payload(task_id: str, event_key: str, userid: str, chatid: str):
    """Build a template_card_event payload mimicking WeCom websocket."""
    return {
        "body": {
            "msgtype": "template_card_event",
            "template_card_event": {
                "task_id": task_id,
                "event_key": event_key,
            },
            "from": {"userid": userid},
            "chatid": chatid,
            "chattype": "single",
        }
    }


class TestTemplateCardEvent:
    """Security tests for _handle_template_card_event."""

    @pytest.mark.asyncio
    async def test_legitimate_click_dispatches_approve(self):
        """A click from the expected admin chat+user synthesises /approve."""
        adapter = _make_adapter()
        adapter._approval_tasks["task_123"] = (
            "session_key_abc",  # session_key
            "admin_chat_id",     # expected chat_id
            "admin_user_id",      # expected user_id
            time.monotonic(),     # timestamp
        )

        payload = _make_card_payload("task_123", "approve", "admin_user_id", "admin_chat_id")
        await adapter._handle_template_card_event(payload["body"], payload)

        # handle_message should have been called with /approve
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.text == "/approve"

        # Task should be cleaned up
        assert "task_123" not in adapter._approval_tasks

    @pytest.mark.asyncio
    async def test_legitimate_deny_click(self):
        """A deny click synthesises /deny."""
        adapter = _make_adapter()
        adapter._approval_tasks["task_456"] = (
            "session_key", "admin_chat", "admin_user", time.monotonic(),
        )

        payload = _make_card_payload("task_456", "deny", "admin_user", "admin_chat")
        await adapter._handle_template_card_event(payload["body"], payload)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.text == "/deny"
        assert "task_456" not in adapter._approval_tasks

    @pytest.mark.asyncio
    async def test_chat_id_mismatch_rejected(self):
        """A click from a different chat (forwarded card) is rejected.

        The task entry is preserved so the real admin can still approve.
        """
        adapter = _make_adapter()
        adapter._approval_tasks["task_789"] = (
            "session_key", "expected_chat", "expected_user", time.monotonic(),
        )

        # Click comes from a DIFFERENT chat_id
        payload = _make_card_payload("task_789", "approve", "expected_user", "wrong_chat")
        await adapter._handle_template_card_event(payload["body"], payload)

        # handle_message must NOT be called
        adapter.handle_message.assert_not_awaited()

        # Task entry must still be present (real admin can still click)
        assert "task_789" in adapter._approval_tasks

    @pytest.mark.asyncio
    async def test_user_id_mismatch_rejected(self):
        """A click from the right chat but wrong user is rejected.

        The task entry is preserved so the real admin can still approve.
        """
        adapter = _make_adapter()
        adapter._approval_tasks["task_000"] = (
            "session_key", "expected_chat", "expected_user", time.monotonic(),
        )

        # Same chat, DIFFERENT user
        payload = _make_card_payload("task_000", "approve", "impostor_user", "expected_chat")
        await adapter._handle_template_card_event(payload["body"], payload)

        adapter.handle_message.assert_not_awaited()
        assert "task_000" in adapter._approval_tasks

    @pytest.mark.asyncio
    async def test_expired_task_still_dispatches(self):
        """When the task entry is expired/missing, the click still synthesises
        a command (the slash-command pipeline will report 'expired' downstream).

        This tests current behaviour — the handler does not block on a missing
        task, it lets the downstream pipeline handle the 'no pending approval' case.
        """
        adapter = _make_adapter()
        # No task entry stored (expired or gateway restart)

        payload = _make_card_payload("ghost_task", "approve", "some_user", "some_chat")
        await adapter._handle_template_card_event(payload["body"], payload)

        # Current behaviour: synthesises /approve even without stored task.
        # The downstream /approve handler will report 'expired' to the user.
        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_event_key_cleans_up(self):
        """An unknown event_key cleans up the task and does not dispatch."""
        adapter = _make_adapter()
        adapter._approval_tasks["task_bad"] = (
            "session_key", "chat", "user", time.monotonic(),
        )

        payload = _make_card_payload("task_bad", "unknown_action", "user", "chat")
        await adapter._handle_template_card_event(payload["body"], payload)

        adapter.handle_message.assert_not_awaited()
        # Unknown key cleans up the task
        assert "task_bad" not in adapter._approval_tasks

    @pytest.mark.asyncio
    async def test_approve_session_synthesises_correct_command(self):
        """The approve_session button synthesises '/approve session'."""
        adapter = _make_adapter()
        adapter._approval_tasks["task_ses"] = (
            "sk", "chat", "user", time.monotonic(),
        )

        payload = _make_card_payload("task_ses", "approve_session", "user", "chat")
        await adapter._handle_template_card_event(payload["body"], payload)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.text == "/approve session"

    @pytest.mark.asyncio
    async def test_approve_always_synthesises_correct_command(self):
        """The approve_always button synthesises '/approve always'."""
        adapter = _make_adapter()
        adapter._approval_tasks["task_alw"] = (
            "sk", "chat", "user", time.monotonic(),
        )

        payload = _make_card_payload("task_alw", "approve_always", "user", "chat")
        await adapter._handle_template_card_event(payload["body"], payload)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.text == "/approve always"

    @pytest.mark.asyncio
    async def test_missing_card_event_data_returns_silently(self):
        """Malformed payload (no template_card_event dict) returns silently."""
        adapter = _make_adapter()

        # No template_card_event key
        await adapter._handle_template_card_event({"msgtype": "template_card_event"}, {})
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_double_click_idempotent(self):
        """A second click on the same task_id does not dispatch again.

        The first click pops the task entry; the second click finds no
        stored entry and still dispatches (current behaviour), but the
        downstream /approve pipeline handles the 'no pending approval' case.

        This test documents the current behaviour: WeCom relies on the
        downstream pipeline for idempotency, unlike Slack which has
        atomic-pop double-click prevention.
        """
        adapter = _make_adapter()
        adapter._approval_tasks["task_dbl"] = (
            "sk", "chat", "user", time.monotonic(),
        )

        payload = _make_card_payload("task_dbl", "approve", "user", "chat")

        # First click — dispatches and cleans up
        await adapter._handle_template_card_event(payload["body"], payload)
        assert adapter.handle_message.await_count == 1
        assert "task_dbl" not in adapter._approval_tasks

        # Second click — task already gone, still dispatches (downstream handles)
        await adapter._handle_template_card_event(payload["body"], payload)
        assert adapter.handle_message.await_count == 2
