"""Notification-scoped Graph ingress must isolate durable conversation history."""

import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.msgraph_webhook import MSGraphWebhookAdapter
from gateway.session import SessionStore


class NotificationRequest:
    remote = "127.0.0.1"
    query = {}
    content_length = None

    def __init__(self, notifications):
        self.notifications = notifications

    async def read(self):
        return json.dumps({"value": self.notifications}).encode("utf-8")


def make_adapter(**extra):
    config = GatewayConfig.from_dict({"platforms": {"msgraph_webhook": {
        "enabled": True,
        "extra": {"host": "127.0.0.1", "client_state": "test-secret", **extra},
    }}})
    return MSGraphWebhookAdapter(config.platforms[Platform.MSGRAPH_WEBHOOK])


def notification(receipt=None):
    result = {
        "subscriptionId": "one-subscription",
        "resource": "communications/onlineMeetings/same-meeting",
        "changeType": "updated",
        "clientState": "test-secret",
    }
    if receipt is not None:
        result["id"] = receipt
    return result


@pytest.mark.anyio
@pytest.mark.parametrize("explicit_receipts", [True, False])
async def test_opt_in_notifications_isolate_persisted_history(tmp_path, explicit_receipts):
    adapter = make_adapter(session_per_notification=True)
    events = []
    adapter.set_notification_scheduler(lambda payload, event: events.append(event))
    first = notification("first" if explicit_receipts else None)
    second = notification("second" if explicit_receipts else None)
    response = await adapter._handle_notification(NotificationRequest([first, second]))
    assert response.status == 202
    assert len(events) == 2
    assert all(event.internal for event in events)
    # No listener, service, model call or Graph credentials. Exercise the real
    # routing store and SQLite transcripts, not just unequal event payloads.
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    entries = []
    try:
        for i, event in enumerate(events):
            entry = store.get_or_create_session(event.source, touch_activity=False)
            store.append_to_transcript(entry.session_id, {"role": "user", "content": f"notice-{i}"})
            entries.append(entry)
        assert entries[0].session_id != entries[1].session_id
        assert [m["content"] for m in store.load_transcript(entries[0].session_id)] == ["notice-0"]
        assert [m["content"] for m in store.load_transcript(entries[1].session_id)] == ["notice-1"]
        # Reopening the routing index must keep the same isolated identities.
        reopened = SessionStore(tmp_path / "sessions", GatewayConfig())
        try:
            for event, entry in zip(events, entries):
                assert reopened.get_or_create_session(event.source).session_id == entry.session_id
        finally:
            reopened.close_all_db_handles()
    finally:
        store.close_all_db_handles()


@pytest.mark.anyio
@pytest.mark.parametrize("extra", [{}, {"session_per_notification": False}, {"session_per_notification": "false"}])
async def test_subscription_sessions_remain_default(extra):
    adapter = make_adapter(**extra)
    events = []
    adapter.set_notification_scheduler(lambda payload, event: events.append(event))
    response = await adapter._handle_notification(NotificationRequest([notification("a"), notification("b")]))
    assert response.status == 202
    assert [event.source.chat_id for event in events] == ["msgraph:one-subscription"] * 2


@pytest.mark.anyio
async def test_opt_in_preserves_auth_and_receipt_dedup():
    adapter = make_adapter(session_per_notification=True)
    events = []
    adapter.set_notification_scheduler(lambda payload, event: events.append(event))
    forged = {**notification("forged"), "clientState": "wrong"}
    assert (await adapter._handle_notification(NotificationRequest([forged]))).status == 403
    assert events == []
    accepted = notification("accepted")
    assert (await adapter._handle_notification(NotificationRequest([accepted, accepted]))).status == 202
    assert len(events) == 1
    assert events[0].message_id == "id:accepted"
