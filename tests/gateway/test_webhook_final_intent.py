"""Primary owner of original-final durability, evidence and physical RPC custody.

Real signed ingress, FIFO settlement and SQLite persistence; transport is controlled,
not E2E. The fake recipient supplies wire observations, never producer persistence.
"""
import asyncio
import json
import threading
from pathlib import Path

import pytest

from tests.gateway.test_webhook_finalization_recovery import settled_before_finalizer


def intent(authority, admission_id):
    row = authority.db._read_one('SELECT value FROM state_meta WHERE key=?',
                                ('webhook.final.intent.v1.' + admission_id,))
    return json.loads(row['value']) if row else None


def transport(monkeypatch, *, status='queued'):
    from tools import bot_live_delivery
    receipts, calls = {}, []
    monkeypatch.setattr(bot_live_delivery, 'read_delivery_result', lambda home, key: receipts.get(key))
    monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', lambda home: {'session_id': ''})
    def send(home, owner, message, *, delivery_id, **kwargs):
        calls.append((str(home), delivery_id, message, kwargs))
        receipts[delivery_id] = dict(status=status, delivery_id=delivery_id,
            profile_home=str(home), session_id='target-session', admission_id='native-admission', message=message,
            **({'notification_category': kwargs['notification_category']} if kwargs.get('notification_category') else {}))
        return receipts[delivery_id]
    monkeypatch.setattr(bot_live_delivery, 'deliver_to_live_owner', send)
    return receipts, calls


@pytest.mark.asyncio
async def test_offline_final_is_durable_and_active_retry_observes_not_resends(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from gateway.platforms.webhook_bot_chat import result_id
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _runner, authority, row):
        receipts, calls = transport(monkeypatch)
        monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', lambda home: None)
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved is not None, 'owed final must survive an offline recipient'
        assert saved['state'] == 'pending' and saved['source_outcome'] == 'completed'
        assert saved['delivery_id'] == result_id(Path(authority.db.db_path).parent, row['admission_id'])
        monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', lambda home: {'session_id': ''})
        await recover_bot_chat_finals(authority)
        assert intent(authority, row['admission_id'])['state'] == 'queued'
        assert len(calls) == 1
        receipts[saved['delivery_id']]['status'] = 'settled'
        await recover_bot_chat_finals(authority)
        assert intent(authority, row['admission_id'])['state'] == 'settled'
        assert len(calls) == 1
        assert len(authority.db._read_all('SELECT * FROM session_admissions')) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['queued', 'claimed', 'settled', 'ambiguous', 'failed', 'cancelled'])
async def test_native_state_and_missing_admitted_evidence_are_not_fresh_send(monkeypatch, status):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _, authority, row):
        receipts, calls = transport(monkeypatch, status=status)
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved is not None
        assert saved['state'] == ('failed' if status == 'cancelled' else status)
        assert saved['native_admission_id'] == 'native-admission' and saved['ever_admitted']
        receipts.clear()
        await recover_bot_chat_finals(authority)
        assert len(calls) == 1, 'lost evidence must never replay admitted work'
        if status in {'queued', 'claimed', 'ambiguous'}:
            assert intent(authority, row['admission_id'])['state'] == 'ambiguous'


@pytest.mark.asyncio
async def test_mismatched_receipt_refuses_without_disclosing_or_replaying(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _, authority, row):
        receipts, calls = transport(monkeypatch)
        await recover_bot_chat_finals(authority)
        key = calls[0][1]
        receipts[key]['message'] = 'FOREIGN SECRET'
        await recover_bot_chat_finals(authority)
        saved = intent(authority, row['admission_id'])
        assert saved and saved['state'] == 'refused'
        assert 'SECRET' not in json.dumps(saved) and len(calls) == 1


@pytest.mark.asyncio
async def test_route_revocation_becomes_durable_refusal(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, runner, authority, row):
        _, calls = transport(monkeypatch)
        monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', lambda home: None)
        await recover_bot_chat_finals(authority)
        next(iter(runner.adapters.values()))._routes['fixture']['prompt'] = 'revoked'
        await recover_bot_chat_finals(authority)
        assert intent(authority, row['admission_id'])['state'] == 'refused'
        assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize('retire', [False, True])
async def test_cancelled_observer_does_not_release_physical_handoff(monkeypatch, retire):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from gateway.session_runtime_workers import mutation_tasks, join_authority_work
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _, authority, row):
        _, calls = transport(monkeypatch)
        entered, release = threading.Event(), threading.Event()
        original = bot_live_delivery.deliver_to_live_owner
        def held(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)
        monkeypatch.setattr(bot_live_delivery, 'deliver_to_live_owner', held)
        observer = asyncio.create_task(recover_bot_chat_finals(authority))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert mutation_tasks(authority), 'thread still owns an authority mutation'
            with pytest.raises(TimeoutError, match='ownership retained'):
                await join_authority_work(authority, 0)
            if retire:
                from gateway.run_runtime import _retire_profile_authority
                assert await _retire_profile_authority(authority, timeout=0) is False
                assert authority.retiring and not getattr(authority, 'retired', False)
                assert mutation_tasks(authority) and not calls
        finally:
            release.set()
            results = await asyncio.gather(*mutation_tasks(authority), return_exceptions=retire)
        if retire:
            from hermes_state_runtime import RuntimeStoreError
            assert any(isinstance(result, RuntimeStoreError) and result.reason == 'runtime_retired' for result in results)
            assert intent(authority, row['admission_id'])['state'] == 'pending'
            assert await _retire_profile_authority(authority, timeout=1) is True
            assert authority.retired and not mutation_tasks(authority)
            from gateway.session_authority import initialize_session_authority
            successor = await initialize_session_authority(authority.runner, profile_id=authority.profile_id,
                instance_id='successor-after-physical-join', db=authority.db)
            await recover_bot_chat_finals(successor)
        assert intent(authority, row['admission_id'])['state'] == 'queued'
        assert len(calls) == 1
