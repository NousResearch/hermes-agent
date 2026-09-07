"""Regression coverage for canonical Kanban Telegram notifications."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.kanban_watchers_notifier import _KanbanNotification


class _Event:
    kind = "completed"


def _notification():
    notification = _KanbanNotification.__new__(_KanbanNotification)
    notification.sub = {
        "chat_id": "8148316720",
        "platform": "telegram",
        "thread_id": "12345",
        "delivery_metadata": {"thread_id": "12345", "message_thread_id": "12345"},
        "notifier_profile": "orion",
    }
    notification.adapter = MagicMock()
    notification.platform_str = "telegram"
    notification.task_id = "t_test"
    notification.task = None
    notification.board_slug = "default"
    notification.runner = MagicMock()
    notification.runner._deliver_kanban_artifacts = AsyncMock()
    return notification


@pytest.mark.asyncio
async def test_operational_telegram_bypasses_notifier_profile_adapter():
    notification = _notification()

    with patch(
        "tools.operational_sender.send_operational_message",
        return_value={"ok": True},
    ) as sender:
        await notification._send_event(_Event(), "Kanban update")

    sender.assert_called_once_with("Kanban update")
    notification.adapter.send.assert_not_called()
    notification.runner._deliver_kanban_artifacts.assert_awaited_once_with(
        adapter=notification.adapter,
        chat_id="8148316720",
        metadata={},
        event_payload=None,
        task=None,
        operational=True,
    )


@pytest.mark.asyncio
async def test_operational_telegram_sender_failure_propagates_and_keeps_no_topic():
    notification = _notification()

    with patch(
        "tools.operational_sender.send_operational_message",
        side_effect=RuntimeError("identity unavailable"),
    ) as sender:
        with pytest.raises(RuntimeError, match="identity unavailable"):
            await notification._send_event(_Event(), "Kanban update")

    sender.assert_called_once_with("Kanban update")
    notification.adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_non_operational_telegram_preserves_adapter_metadata():
    notification = _notification()
    notification.sub["chat_id"] = "other-chat"
    notification.adapter.send = AsyncMock(return_value=None)

    await notification._send_event(_Event(), "Kanban update")

    notification.adapter.send.assert_awaited_once_with(
        "other-chat",
        "Kanban update",
        metadata={"thread_id": "12345", "message_thread_id": "12345"},
    )
