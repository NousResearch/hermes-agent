"""Missing terminal evidence and fairness at real source persistence boundaries."""
import pytest

from tests.gateway.test_webhook_final_intent import intent, transport
from tests.gateway.test_webhook_finalization_recovery import settled_before_finalizer


@pytest.mark.asyncio
async def test_missing_committed_result_is_ambiguous_not_suppressed_success(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from hermes_state_terminal import RESULT_PREFIX
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _, authority, row):
        authority.db._write_sql('DELETE FROM state_meta WHERE key=?', (RESULT_PREFIX + row['admission_id'],))
        _, calls = transport(monkeypatch)
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved and saved['state'] == 'ambiguous' and saved['reason'] == 'source_result_unavailable'
        assert not calls


@pytest.mark.asyncio
async def test_bounded_pass_rotates_past_offline_origin(monkeypatch):
    from gateway import webhook_reconcile
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from hermes_state_runtime import claim_session_input, settle_session_input
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat') as (client, body, headers, _, authority, first):
        headers = {**headers, 'X-GitHub-Delivery': 'second-offline'}
        response = await client.post('/webhooks/fixture', data=body, headers=headers)
        assert response.status == 202
        second = authority.db._read_one('SELECT * FROM session_admissions WHERE request_id=?', ('second-offline',))
        claimed = claim_session_input(authority.db, epoch=authority.epoch, session_id=second['target_session_id'])
        assert isinstance(claimed, dict)
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=claimed['admission_id'],
            generation=claimed['generation'], outcome='completed', result={'result': {'final_response': 'second final'}})
        _, calls = transport(monkeypatch)
        monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', lambda home: None)
        monkeypatch.setattr(webhook_reconcile, 'PASS_LIMIT', 1)
        await recover_bot_chat_finals(authority)
        assert sum(intent(authority, key) is not None for key in [first['admission_id'], second['admission_id']]) == 1
        await recover_bot_chat_finals(authority)
        for key in [first['admission_id'], second['admission_id']]:
            saved = intent(authority, key)
            assert saved and saved['state'] == 'pending'
        assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize('observation', ['uncertain-dial', 'pre-admission-pin'])
async def test_missing_evidence_distinguishes_uncertain_dial_from_withdrawn_native_pin(monkeypatch, observation):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from gateway.platforms.webhook_bot_chat import result_id, wrapped_message
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _, authority, row):
        receipts, calls = transport(monkeypatch)
        key = result_id(authority.db.db_path.parent, row['admission_id'])
        home = authority.db.db_path.parent
        if observation == 'uncertain-dial':
            attempts = []
            original = bot_live_delivery.deliver_to_live_owner
            def uncertain(*args, **kwargs):
                attempts.append(kwargs['delivery_id'])
                raise OSError('private socket error')
            monkeypatch.setattr(bot_live_delivery, 'deliver_to_live_owner', uncertain)
        else:
            receipts[key] = dict(status='ambiguous', delivery_id=key, profile_home=str(home),
                session_id='target-session', principal_id='uid:1003',
                message=wrapped_message('fixture', 'committed answer'))
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved and saved['state'] == 'ambiguous' and not saved['ever_admitted']
        receipts.clear()  # native observation: only the owner may withdraw a pre-admission pin
        if observation == 'uncertain-dial':
            monkeypatch.setattr(bot_live_delivery, 'deliver_to_live_owner', original)
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved
        if observation == 'uncertain-dial':
            assert saved['state'] == 'ambiguous' and not calls and attempts == [key]
        else:
            assert saved['state'] == 'queued' and calls[0][1] == key


@pytest.mark.asyncio
async def test_malformed_pre_admission_evidence_cannot_authorize_a_withdrawal_retry(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from gateway.platforms.webhook_bot_chat import result_id, wrapped_message
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _, authority, row):
        receipts, calls = transport(monkeypatch)
        key = result_id(authority.db.db_path.parent, row['admission_id'])
        receipts[key] = dict(status='ambiguous', delivery_id=key, profile_home=str(authority.db.db_path.parent),
            session_id='target-session', message=wrapped_message('fixture', 'committed answer'))
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved and saved['state'] == 'refused'
        receipts.clear()
        await recover_bot_chat_finals(authority)
        assert not calls
