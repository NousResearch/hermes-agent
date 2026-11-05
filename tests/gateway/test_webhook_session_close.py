"""A late webhook finalizer cannot end a replacement route or generation."""
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.platforms.webhook import WebhookAdapter
from gateway.run import GatewayRunner
from gateway.session_authority import admission_row, initialize_session_authority


@pytest.mark.asyncio
async def test_webhook_finalization_is_exact_session_and_generation(tmp_path):
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='close-test')
    runner._session_db = authority.persistence
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={'secret': 'owned-close-secret'}))
    adapter.gateway_runner = runner
    runner.adapters[Platform.WEBHOOK] = adapter
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='webhook:close:one', user_id='webhook:close', chat_type='webhook')
    ref = authority.register(source)

    def settled(identity):
        row, _created = authority.persistence.admissions.admit(epoch=authority.epoch, principal_id='fixture',
            session_id=ref.session_id, request_id=identity, payload={'text': 'fixture'})
        claimed = authority.persistence.admissions.claim(epoch=authority.epoch, session_id=ref.session_id)
        return authority._receipt(admission_row(authority.persistence.admissions.settle(epoch=authority.epoch,
            admission_id=row.admission_id, generation=claimed.generation, outcome='completed')))

    receipt = settled('first')
    event = MessageEvent(text='fixture', source=source, message_id='first')
    event._webhook_receipt = (authority, receipt)
    successor = runner.session_store.get_or_create_session(source, force_new=True)
    assert successor.session_id != ref.session_id
    # Simulate a delayed callback after route publication. The old authority
    # receipt, not the current route, is the only admissible finalization target.
    await adapter._end_webhook_session(event, source.chat_id)
    assert authority.persistence.sessions.get(successor.session_id).ended_at is None

    authority.persistence.sessions.reopen(ref.session_id)
    newer = settled('second')
    assert newer.execution_generation > receipt.execution_generation
    await adapter._end_webhook_session(event, source.chat_id)
    assert authority.persistence.sessions.get(ref.session_id).ended_at is None
    event._webhook_receipt = (authority, newer)
    await adapter._end_webhook_session(event, source.chat_id)
    assert authority.persistence.sessions.get(ref.session_id).end_reason == 'webhook_complete'
