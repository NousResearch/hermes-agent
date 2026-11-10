"""Attributable automation finals inherit a real ingress root and its pin.

Real FIFO/admission/settlement with controlled transport, not a model E2E.
"""
import json

import pytest

from tests.gateway.test_webhook_finalization_recovery import settled_before_finalizer
from tests.gateway.test_webhook_final_intent import intent, transport


@pytest.mark.asyncio
async def test_two_transitive_completion_admissions_keep_root_pin_and_own_final(monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.platforms.webhook_ingress import finalize_webhook, recover_bot_chat_finals
    from gateway.session_envelope import restore_native
    from hermes_state_runtime import get_session_admission, claim_session_input, settle_session_input
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, runner, authority, original):
        _, calls = transport(monkeypatch)
        root = get_session_admission(authority.db, admission_id=original['admission_id'])
        native = root['payload']['native_text_v1']
        adapter = runner._adapter_for_source(restore_native(root['payload'], runner).source)
        assert finalize_webhook(authority, authority._receipt(root))
        ids = []
        for index, identity in enumerate((['completion', 'process-owned', 123.0], ['async_delegation', 'group-owned', ''])):
            event = MessageEvent(text='actual completion input', source=restore_native(root['payload'], runner).source,
                internal=True, metadata={'gateway_session_key': native['route'],
                    'gateway_session_id': original['target_session_id'],
                    **({'automation_identities': [json.dumps(identity, separators=(',', ':')),
                        '["async_delegation","group-sibling",""]']} if index else {})})
            receipt = await authority.admit_automation(adapter, event, json.dumps(identity, separators=(',', ':')))
            saved = get_session_admission(authority.db, admission_id=receipt.admission_id)
            envelope = saved['payload']['native_text_v1']
            assert envelope.get('webhook_root') == original['admission_id'], 'completion lost its actual webhook root'
            assert envelope['webhook_target'] == native['webhook_target'], 'completion recaptured a target'
            claimed = claim_session_input(authority.db, epoch=authority.epoch, session_id=original['target_session_id'])
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=receipt.admission_id,
                generation=claimed['generation'], outcome='completed',
                result={'result': {'final_response': f'COMPLETION_FINAL_{index}', 'messages': []}, 'usage': {}})
            ids.append(receipt.admission_id)
        await recover_bot_chat_finals(authority)
        for index, admission_id in enumerate(ids):
            value = intent(authority, admission_id)
            assert value and value['origin_kind'] == 'completion_final'
            assert value['origin_id'] == admission_id and value['state'] == 'queued'
            assert value['message'].endswith(f'COMPLETION_FINAL_{index}')
        assert len({call[1] for call in calls}) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('identity', [['heartbeat', 'process-owned', 'tick'], ['watch_match', 'process-owned', 'match']])
async def test_nonfinal_automation_never_becomes_a_bot_chat_origin(monkeypatch, identity):
    from gateway.platforms.event import MessageEvent
    from gateway.session_envelope import restore_native
    from hermes_state_runtime import get_session_admission
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, runner, authority, original):
        native = get_session_admission(authority.db, admission_id=original['admission_id'])['payload']
        event = MessageEvent(text='not a final', source=restore_native(native, runner).source, internal=True,
            metadata={'gateway_session_key': native['native_text_v1']['route'],
                      'gateway_session_id': original['target_session_id']})
        adapter = runner._adapter_for_source(event.source)
        receipt = await authority.admit_automation(adapter, event, json.dumps(identity, separators=(',', ':')))
        saved = get_session_admission(authority.db, admission_id=receipt.admission_id)
        assert 'webhook_delivery' not in saved['payload']['native_text_v1']


@pytest.mark.asyncio
async def test_completion_source_policy_is_frozen_before_transport_recovery(monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_envelope import restore_native
    from gateway.platforms.webhook_bot_chat import presentation
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from gateway import warning_notifications
    from hermes_state_runtime import get_session_admission, claim_session_input, settle_session_input
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, runner, authority, original):
        payload = get_session_admission(authority.db, admission_id=original['admission_id'])['payload']
        native = payload['native_text_v1']
        event = MessageEvent(text='trusted diagnostic completion', source=restore_native(payload, runner).source,
            internal=True, metadata={'gateway_session_key': native['route'],
                'gateway_session_id': original['target_session_id'], 'notification_category': 'diagnostic'})
        adapter = runner._adapter_for_source(event.source)
        receipt = await authority.admit_automation(adapter, event, '["completion","policy-process",123.0]')
        monkeypatch.setattr(warning_notifications, 'warning_notifications_enabled', lambda *a, **kw: False)
        result = {'final_response': 'nonempty diagnostic completion', 'messages': []}
        policy = presentation(result, 'completed', category='diagnostic')
        assert policy['disposition'] == 'suppressed'
        claimed = claim_session_input(authority.db, epoch=authority.epoch, session_id=original['target_session_id'])
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=receipt.admission_id,
            generation=claimed['generation'], outcome='completed',
            result={'result': result, 'usage': {}, 'webhook_presentation': policy})
        monkeypatch.setattr(warning_notifications, 'warning_notifications_enabled', lambda *a, **kw: True)
        _, calls = transport(monkeypatch)
        await recover_bot_chat_finals(authority)
        value = intent(authority, receipt.admission_id)
        assert value['state'] == 'suppressed' and value['category'] == 'diagnostic'
        assert value['source_outcome'] == 'completed' and value['message'] == ''
        assert all(not call[2].endswith('nonempty diagnostic completion') for call in calls)
