import pytest

from gateway.platforms.base import SendResult
from gateway.work_router.config import DEFAULT_PROFILES, RouterConfig
from gateway.work_router.models import CanonicalEvent
from gateway.work_router.service import WorkRouter, _confirmed_send_result


@pytest.mark.parametrize("message_id", [None, "", "   ", False, 123])
def test_success_without_transport_identity_is_not_confirmed(message_id):
    assert _confirmed_send_result(SendResult(success=True, message_id=message_id)) == (False, None)
    assert _confirmed_send_result({"success": True, "message_id": message_id}) == (False, None)


@pytest.mark.asyncio
async def test_unconfirmed_delivery_is_quarantined_without_replay(tmp_path):
    config = RouterConfig.from_mapping({
        "enabled": True,
        "channel_allowlist": ["CTEST"],
        "db_path": str(tmp_path / "router.db"),
        "bot_registry": {name: f"UBOT{i}" for i, name in enumerate(DEFAULT_PROFILES)},
    }).require_ready()
    router = WorkRouter(config)
    event = CanonicalEvent(
        event_id="EvDelivery", channel_id="CTEST", thread_ts="1700000000.1",
        text="한스야 확인해", author_user_id="UHUMAN", mentioned_user_ids=("UBOT1",),
    )
    calls = []

    def sender(**kwargs):
        calls.append(kwargs["operation_id"])
        return SendResult(success=True)

    try:
        router.ack_gate.complete(event.event_id, success=True)
        assert await router.enqueue_after_ack(event)
        result = await router.process_once(event.event_id, sender=sender)
        assert result.status == "quarantined"
        assert not await router.enqueue_after_ack(event)
        await router.process_once(event.event_id, sender=sender)
        assert len(calls) == 1
        assert _confirmed_send_result(SendResult(success=True, message_id="1700000000.2")) == (
            True, "1700000000.2"
        )
    finally:
        router.store.close()
