"""Profile pin and presentation recovery contracts on actual admitted source rows.

Recipient transport controlled; SQLite/FIFO/profile state real. Not model E2E.
"""
import json

from types import SimpleNamespace

import pytest

from tests.gateway.test_webhook_final_intent import intent, transport
from tests.gateway.test_webhook_finalization_recovery import settled_before_finalizer


def receiver():
    from hermes_cli.profiles import get_profile_dir
    home = get_profile_dir('receiver')
    home.mkdir(parents=True)
    (home / 'config.yaml').write_text('{}')
    return home


@pytest.mark.asyncio
async def test_replaced_explicit_target_refuses_admission_time_obligation(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    target = receiver()
    async with settled_before_finalizer(deliver='bot-chat:receiver') as (_, _, _, _, authority, row):
        assert not (target / 'state.db').exists(), 'producer must not initialize the target store'
        target.rename(target.with_name('old-receiver'))
        target.mkdir()
        (target / 'config.yaml').write_text('{}')
        _, calls = transport(monkeypatch)
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved and saved['state'] == 'refused' and not calls
        assert saved['target_pin']['directory'] != [target.stat().st_dev, target.stat().st_ino]


@pytest.mark.asyncio
async def test_first_store_is_frozen_once_under_original_directory(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from hermes_state import SessionDB
    target = receiver()
    async with settled_before_finalizer(deliver='bot-chat:receiver') as (_, _, _, _, authority, row):
        assert not (target / 'state.db').exists()
        _, calls = transport(monkeypatch)
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved and saved['state'] == 'pending' and saved['target_pin']['store'] is None
        with SessionDB(db_path=target / 'state.db'):
            pass  # its own owner, not capture_target, creates the first store
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved and saved['state'] == 'queued' and len(calls) == 1
        assert calls[0][0] == str(target)
        assert calls[0][3]['expected_incarnation'] == saved['target_pin']
        assert saved['target_pin']['store']['file'] == [(target / 'state.db').stat().st_dev, (target / 'state.db').stat().st_ino]


@pytest.mark.asyncio
async def test_new_epoch_recovers_pending_intent_and_fences_previous_owner(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from gateway.session_authority import initialize_session_authority
    from gateway.webhook_final_intent import _save
    from hermes_state_runtime import RuntimeStoreError
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, runner, authority, row):
        _, calls = transport(monkeypatch)
        discover = bot_live_delivery.find_canonical_live_owner
        monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', lambda home: None)
        await recover_bot_chat_finals(authority)
        previous = intent(authority, row['admission_id'])
        assert previous and previous['state'] == 'pending'
        restarted = await initialize_session_authority(runner, profile_id=authority.profile_id,
            instance_id='new-core-owner', db=authority.db)
        restarted._schedule = lambda ref: pytest.fail('delivery retry must not schedule source inference')
        with pytest.raises(RuntimeStoreError, match='stale_epoch'):
            _save(authority, previous)
        monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', discover)
        await recover_bot_chat_finals(restarted)
        recovered = intent(restarted, row['admission_id'])
        assert recovered and recovered['state'] == 'queued'
        assert recovered['delivery_id'] == previous['delivery_id'] and len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['failed', 'interrupted'])
async def test_source_failure_stays_failure_independent_of_transport(monkeypatch, outcome):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    async with settled_before_finalizer(deliver='bot-chat', outcome=outcome, final_response='source failed') as (_, _, _, _, authority, row):
        _, calls = transport(monkeypatch)
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved and saved['source_outcome'] == outcome and saved['state'] == 'queued'
        assert saved['category'] == 'diagnostic' and calls[0][3]['notification_category'] == 'diagnostic'


@pytest.mark.asyncio
async def test_durable_source_suppression_cannot_be_resurrected_by_recovery(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    policy = dict(disposition='suppressed', category='diagnostic', text='', reason='source_policy')
    async with settled_before_finalizer(deliver='bot-chat', final_response='nonempty hidden diagnostic',
                                       presentation=policy) as (_, _, _, _, authority, row):
        _, calls = transport(monkeypatch)
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved and saved['state'] == 'suppressed' and saved['category'] == 'diagnostic'
        assert not calls and not saved['message']


@pytest.mark.asyncio
async def test_redaction_failure_is_durable_refusal_never_placeholder_success(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    def broken(*args, **kwargs):
        raise ValueError('PRIVATE EXCEPTION')
    async with settled_before_finalizer(deliver='bot-chat', final_response='PRIVATE OUTPUT') as (_, _, _, _, authority, row):
        _, calls = transport(monkeypatch)
        monkeypatch.setattr('agent.redact.redact_sensitive_text', broken)
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved and saved['state'] == 'refused' and saved['reason'] == 'redaction_failed'
        assert not calls and 'PRIVATE' not in json.dumps(saved)


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['queued', 'claimed', 'settled'])
async def test_unpinned_upgrade_adopts_only_proven_same_id_native_receipt(monkeypatch, status):
    from gateway.platforms.webhook_bot_chat import result_id, wrapped_message
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from gateway.session_bot import deliver, refresh_receipt
    from gateway.session_contract import SessionActor
    from hermes_state_runtime import claim_session_input, settle_session_input
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _, authority, row):
        key = result_id(authority.db.db_path.parent, row['admission_id'])
        actor = SessionActor('uid:1003', authority.profile_id,
            frozenset({'session:read', 'session:submit', 'session:create'}), 'native')
        from gateway.session_local import create_local_session
        from gateway.session_local_title import title_new_session
        chat = create_local_session(authority, actor, dict(request_id='upgrade-target', source='gui',
            cwd=str(authority.db.db_path.parent), model='fixture', toolsets=[]))
        title_new_session(authority, chat, 'Bot Chat')
        message = wrapped_message('fixture', 'committed answer')
        native = await deliver(SimpleNamespace(authority=authority, actor=actor),
            dict(id=key, profile='default', message=message))
        if status != 'queued':
            claimed = claim_session_input(authority.db, epoch=authority.epoch, session_id=native['session_id'])
            if status == 'settled':
                settle_session_input(authority.db, epoch=authority.epoch, admission_id=claimed['admission_id'],
                    generation=claimed['generation'], outcome='completed', result={'result': {'final_response': 'ack'}})
            await refresh_receipt(authority, authority.db.db_path.parent, key)
        payload = json.loads(row['payload_json'])
        payload['native_text_v1'].pop('webhook_target')  # exact pre-upgrade source shape, not a fabricated pin
        authority.db._write_sql('UPDATE session_admissions SET payload_json=? WHERE admission_id=?',
            (json.dumps(payload), row['admission_id']))
        def forbidden(*args, **kwargs):
            pytest.fail('an upgrade must only adopt, never redial an existing receipt')
        monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', forbidden)
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved and saved['state'] == status and saved['target_pin'] is None
        assert saved['native_admission_id'] == native['admission_id']
        assert saved['delivery_id'] == key and saved['message'] == message
