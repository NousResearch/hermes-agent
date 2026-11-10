"""Original-final runtime wiring; durability's primary owner is test_webhook_final_intent."""
import threading

import pytest

from gateway.platforms.webhook_delivery import validate_destination, snapshot_destination
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_webhook_finalization_recovery import settled_before_finalizer
from tests.gateway.test_webhook_final_intent import transport


def test_bot_chat_destination_roundtrip_and_closed_target():
    source = {'deliver': 'bot-chat', 'deliver_extra': {}, 'route': 'orders', 'profile': 'other'}
    assert snapshot_destination(None, source) == source
    assert validate_destination({**source, 'mirror': True}) == source  # inherent, never an extra append
    assert validate_destination({**source, 'deliver': 'bot-chat:other'})['deliver'] == 'bot-chat:other'
    for patch in ({'deliver_extra': {'chat_id': 'other'}}, {'deliver': 'bot-chat:other/session'}):
        with pytest.raises(RuntimeStoreError):
            validate_destination({**source, **patch})


@pytest.mark.asyncio
async def test_native_receipt_offloop_and_result_identity(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from gateway.platforms.webhook_bot_chat import result_id
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _, authority, row):
        receipts, calls = transport(monkeypatch)
        loop_thread = threading.get_ident()
        send = bot_live_delivery.deliver_to_live_owner
        def offloop(*args, **kwargs):
            assert threading.get_ident() != loop_thread
            return send(*args, **kwargs)
        monkeypatch.setattr(bot_live_delivery, 'deliver_to_live_owner', offloop)
        await recover_bot_chat_finals(authority)
        key = result_id(authority.db.db_path.parent, row['admission_id'])
        assert receipts[key]['status'] == 'queued'
        assert 'Webhook automation' in calls[0][2] and 'not the user' in calls[0][2]
        await recover_bot_chat_finals(authority)
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_existing_receipt_is_observed_without_discovery(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _, authority, _):
        _, calls = transport(monkeypatch, status='claimed')
        await recover_bot_chat_finals(authority)
        def forbidden(home):
            pytest.fail('a matching admitted receipt must be observed, not resent')
        monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', forbidden)
        await recover_bot_chat_finals(authority)
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_silence_and_force_redaction(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from agent import redact
    monkeypatch.setattr(redact, 'redact_sensitive_text', lambda text, force=False: '[scrubbed]' if force else text)
    async with settled_before_finalizer(deliver='bot-chat', final_response='private') as (_, _, _, _, authority, _):
        _, calls = transport(monkeypatch)
        await recover_bot_chat_finals(authority)
        assert len(calls) == 1 and 'private' not in calls[0][2] and calls[0][2].count('[scrubbed]') == 2


@pytest.mark.asyncio
async def test_settled_uses_original_committed_result_not_event_text(monkeypatch):
    from gateway.platforms.webhook_bot_chat import deliver_settled
    from gateway.session_envelope import restore_native
    from hermes_state_runtime import get_session_admission
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, runner, authority, row):
        _, calls = transport(monkeypatch)
        saved = get_session_admission(authority.db, admission_id=row['admission_id'])
        event = restore_native(saved['payload'], runner)
        event.text = 'not the committed answer'
        adapter = runner._adapter_for_source(event.source)
        result = await deliver_settled(authority, adapter, event, row['admission_id'])
        assert result.success and 'committed answer' in calls[0][2]
        assert event.text not in calls[0][2]
        assert result.raw_response['status'] == 'queued'  # success is handoff admission, not task settlement
        assert not (await deliver_settled(authority, adapter, event, 'not-an-admission')).success


@pytest.mark.asyncio
async def test_arbitrary_media_looking_send_cannot_create_bot_chat_turn():
    from gateway.platforms.webhook import WebhookAdapter
    from gateway.config import PlatformConfig
    adapter = WebhookAdapter(PlatformConfig(enabled=True))
    result = await adapter._deliver_to({'deliver': 'bot-chat:other'}, 'untrusted', 'MEDIA: /x/image.png')
    assert not result.success and 'committed' in result.error
