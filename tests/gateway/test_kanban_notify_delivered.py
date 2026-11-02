"""Delivery observers receive the actual ping anchor after its durable checkpoint."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.kanban_watchers_notifier import _KanbanNotification
from gateway.platforms.base import SendResult


@pytest.fixture
def notification(monkeypatch):
    async def present(send, **kwargs):
        await send()
        return True

    monkeypatch.setattr("gateway.warning_notifications.present_notification", present)
    runner = SimpleNamespace(_kanban_sub_op=Mock(), _kanban_rewind=Mock())
    sub = {
        "task_id": "task-example", "platform": "telegram", "chat_id": "chat-example",
        "thread_id": "thread-example", "notifier_profile": "observer",
        "delivery_metadata": {"chat_type": "group", "parent_chat_id": "parent-example"},
    }
    event = SimpleNamespace(id=12, kind="blocked", payload={"reason": "needs input"})
    n = _KanbanNotification(
        runner, {"sub": sub, "task": None, "board": "example", "events": [event], "cursor": 12},
        platform_cls=str, sub_fail_counts={},
    )
    n.adapter = SimpleNamespace(send=AsyncMock())
    return n


@pytest.mark.asyncio
@pytest.mark.parametrize(("result", "expected"), [
    (SendResult(success=True, message_id="single"), "single"),
    (SendResult(success=True, message_id="last", continuation_message_ids=("first", "last")), "first"),
    (None, None),
])
async def test_delivery_observer_receives_anchor_and_copied_metadata(notification, monkeypatch, result, expected):
    n = notification
    n.adapter.send.return_value = result
    # A legacy adapter returning None must not reuse a previous event's anchor.
    n.last_message_id = "stale"
    observed = []

    async def observe(name, **kwargs):
        n.runner._kanban_sub_op.assert_called_once_with(
            "example", "record_notify_ping", n.sub, event_id=12,
        )
        observed.append((name, kwargs))
        assert kwargs["delivery_metadata"] is not n.sub["delivery_metadata"]
        kwargs["delivery_metadata"]["observer_only"] = True

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", observe)
    assert await n._send_pings() is True
    assert len(observed) == 1
    name, payload = observed[0]
    assert name == "kanban_notify_delivered"
    assert payload == {
        "task_id": "task-example", "board": "example", "platform": "telegram",
        "chat_id": "chat-example", "thread_id": "thread-example", "event_id": 12,
        "event_kind": "blocked", "notifier_profile": "observer", "message_id": expected,
        "delivery_metadata": {
            "chat_type": "group", "parent_chat_id": "parent-example", "observer_only": True,
        },
    }
    assert "observer_only" not in n.sub["delivery_metadata"]


@pytest.mark.asyncio
async def test_observer_failure_cannot_rewind_delivered_ping(notification, monkeypatch):
    n = notification
    n.adapter.send.return_value = SendResult(success=True, message_id="delivered")
    hook = AsyncMock(side_effect=RuntimeError("observer failed"))
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", hook)
    n.sub_fail_counts[n.sub_key] = 2
    assert await n._send_pings() is True
    hook.assert_awaited_once()
    n.runner._kanban_sub_op.assert_called_once()
    n.runner._kanban_rewind.assert_not_called()
    assert n.sub_key not in n.sub_fail_counts


@pytest.mark.asyncio
async def test_failed_send_never_emits_delivery_hook(notification, monkeypatch):
    n = notification
    n.adapter.send.return_value = SendResult(success=False, error="offline")
    hook = AsyncMock()
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", hook)
    assert await n._send_pings() is False
    hook.assert_not_awaited()
    n.runner._kanban_sub_op.assert_not_called()
    n.runner._kanban_rewind.assert_called_once()
