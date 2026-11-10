"""Producer privacy and authority end at the existing source-retirement boundary.

Signed ingress and real terminal admissions/deletes/prunes own lifecycle proof.
Transport observations are controlled, not recipient E2E or manufactured source work.
Ordinary private prose is intentionally not a recognized secret-redactor canary.
"""
import json
import time

import pytest

from tests.gateway.test_webhook_finalization_recovery import settled_before_finalizer
from tests.gateway.test_webhook_final_intent import intent, transport



PRIVATE = 'Ordinary private quarterly acquisition report for the board'


def delete_source(authority, session_id):
    from hermes_state_runtime import mutate_runtime_session
    snap = authority.db.get_session(session_id)
    return mutate_runtime_session(authority.db, epoch=authority.epoch, principal_id='human',
        session_id=session_id, request_id='delete-source', operation='delete', payload={},
        expected_revision=snap['runtime_revision'], expected_generation=snap['runtime_generation'])


def frozen(value):
    return {k: value.get(k) for k in ('origin_id', 'delivery_id', 'message_digest', 'target_pin',
        'native_admission_id', 'native_session_id', 'ever_admitted', 'source_outcome', 'fingerprint')}


def expire(authority, monkeypatch, at):
    with monkeypatch.context() as clock:
        clock.setattr(time, 'time', lambda: at)
        authority.db.prune_sessions(older_than_days=90)


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['pending', 'queued', 'claimed', 'settled'])
async def test_deleted_terminal_source_refuses_owed_intent_not_ended_source(monkeypatch, state, record_property):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals, finalize_webhook
    from hermes_state_runtime import get_session_admission
    from hermes_cli.webhook_inspection import delivery_states
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat', final_response=PRIVATE) as (_, _, _, _, authority, row):
        _, calls = transport(monkeypatch, status=state if state != 'pending' else 'queued')
        if state == 'pending':
            monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', lambda home: None)
        await recover_bot_chat_finals(authority)
        before = intent(authority, row['admission_id'])
        assert before['state'] == state and PRIVATE in before['message']
        # Normal one-shot ending is not loss of source authority.
        native = get_session_admission(authority.db, admission_id=row['admission_id'])
        assert finalize_webhook(authority, authority._receipt(native))
        await recover_bot_chat_finals(authority)
        assert intent(authority, row['admission_id'])['state'] == state
        sent = len(calls)
        scheduled = []
        authority._schedule = scheduled.append
        delete_source(authority, row['target_session_id'])
        tombstone = get_session_admission(authority.db, admission_id=row['admission_id'])
        assert tombstone['status'] == 'terminal' and tombstone['payload'] == {}
        await recover_bot_chat_finals(authority)
        after = intent(authority, row['admission_id'])
        assert after['state'] == ('settled' if state == 'settled' else 'refused')
        assert after['reason'] == (None if state == 'settled' else 'source_unauthorized')
        assert frozen(after) == frozen(before)
        projected, = [v for v in delivery_states()['deliveries'] if v['origin_id'] == row['admission_id']]
        assert projected['state'] == after['state'] and projected['source_outcome'] == 'completed'
        assert PRIVATE not in json.dumps(projected)
        assert len(calls) == sent and scheduled == []
        assert authority.db.get_session(row['target_session_id']) is None
        record_property('source_retirement', json.dumps(dict(before=state, after=projected,
            origin_id=after['origin_id'], delivery_id=after['delivery_id'], digest=after['message_digest'])))


@pytest.mark.asyncio
async def test_unrelated_terminal_admission_without_intent_remains_inapplicable():
    from gateway.webhook_final_intent import reconcile_final
    async with settled_before_finalizer(deliver='log') as (_, _, _, _, authority, row):
        assert await reconcile_final(authority, None, row['admission_id']) is None
        delete_source(authority, row['target_session_id'])
        assert await reconcile_final(authority, None, row['admission_id']) is None
        assert intent(authority, row['admission_id']) is None
