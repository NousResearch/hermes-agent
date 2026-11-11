"""Producer payload compaction follows source retirement or the same standalone horizon."""
import asyncio
import json
import time

import pytest

from tests.gateway.test_webhook_finalization_recovery import settled_before_finalizer
from tests.gateway.test_webhook_final_intent import intent, transport
from tests.gateway.test_webhook_prepared_origins import ingress, post, records
from tests.gateway.test_webhook_source_retirement import PRIVATE, delete_source, frozen, expire


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,state', [('original', 'settled'), ('original', 'pending'), ('completion', 'pending')])
async def test_retired_result_expiry_compacts_producer_report_and_keeps_replay_identity(monkeypatch, kind, state):
    from gateway.platforms.event import MessageEvent
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from gateway.session_envelope import restore_native
    from gateway.session_results import admission_result
    from hermes_state_runtime import get_session_admission, claim_session_input, settle_session_input
    from hermes_state_mutation_retirement import RETRY_HORIZON_S
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat', final_response=PRIVATE) as (_, _, _, runner, authority, root):
        target_id = root['admission_id']
        if kind == 'completion':
            payload = get_session_admission(authority.db, admission_id=target_id)['payload']
            event = MessageEvent(text='completion input', source=restore_native(payload, runner).source, internal=True,
                metadata={'gateway_session_key': payload['native_text_v1']['route'],
                          'gateway_session_id': root['target_session_id']})
            receipt = await authority.admit_automation(runner._adapter_for_source(event.source), event,
                                                      '["completion","retention-process",123.0]')
            claimed = claim_session_input(authority.db, epoch=authority.epoch, session_id=root['target_session_id'])
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=receipt.admission_id,
                generation=claimed['generation'], outcome='completed',
                result={'result': {'final_response': PRIVATE + ' completion', 'messages': []}, 'usage': {}})
            target_id = receipt.admission_id
        _, calls = transport(monkeypatch, status='settled')
        if state == 'pending':
            monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', lambda home: None)
        await recover_bot_chat_finals(authority)
        before = intent(authority, target_id)
        assert before['state'] == state and PRIVATE in before['message']
        at = time.time()
        delete_source(authority, root['target_session_id'])
        assert admission_result(authority.db, target_id) is not None
        expire(authority, monkeypatch, at + RETRY_HORIZON_S - 1)
        assert admission_result(authority.db, target_id) is not None
        expire(authority, monkeypatch, at + RETRY_HORIZON_S + 1)
        from hermes_state_terminal import RESULT_PREFIX
        assert authority.db.get_meta(RESULT_PREFIX + target_id) is None
        after = intent(authority, target_id)
        assert after['message'] == '', 'expiry must remove ordinary private report content, not just secrets'
        assert frozen(after) == frozen(before)
        assert after['state'] == ('settled' if state == 'settled' else 'refused')
        assert PRIVATE not in ''.join(r['value'] for r in authority.db._read_all(
            "SELECT value FROM state_meta WHERE key GLOB 'webhook.*'"))
        sent = len(calls)
        scheduled = []
        authority._schedule = scheduled.append
        await recover_bot_chat_finals(authority)
        assert len(calls) == sent and not scheduled
        assert authority.db._read_all('SELECT * FROM session_admissions') == []


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['direct', 'hook', 'interrupted'])
async def test_standalone_origin_uses_same_horizon_without_rerunning_preparation(monkeypatch, kind):
    from hermes_cli.plugins import get_plugin_manager
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from hermes_state_mutation_retirement import RETRY_HORIZON_S
    calls = []
    if kind == 'hook':
        manager = get_plugin_manager()
        manager.discover_and_load()
        def consume(**event):
            calls.append('hook')
            return {'action': 'handled', 'reply': PRIVATE}
        monkeypatch.setitem(manager._hooks, 'post_gateway_admission', [consume])
    async with ingress(direct=kind != 'hook') as (client, adapter, authority):
        if kind != 'hook':
            adapter._routes['fixture']['script'] = 'owned-script'
            def script(command, payload):
                calls.append('script')
                if kind == 'interrupted':
                    raise RuntimeError('interrupted side effect')
                return True, payload
            monkeypatch.setattr(adapter._route_processor, 'run_route_script', script)
        at = time.time()
        status, first = await post(client, PRIVATE)
        assert status == (503 if kind == 'interrupted' else 202)
        if kind == 'interrupted':
            status, first = await post(client, PRIVATE)
            assert status == 202 and first['status'] == 'origin_unknown'
        before, = records(authority)
        expire(authority, monkeypatch, at + RETRY_HORIZON_S - 1)
        if kind != 'interrupted':
            assert PRIVATE in records(authority)[0]['message']
        expire(authority, monkeypatch, at + RETRY_HORIZON_S + 1)
        after, = records(authority)
        assert after['message'] == '' and after.get('payload_compacted_at') is not None
        assert frozen(after) == frozen(before)
        assert after['state'] == ('origin_unknown' if kind == 'interrupted' else 'refused')
        assert after.get('native') in (None, {})
        await recover_bot_chat_finals(authority)
        status, duplicate = await post(client, PRIVATE)
        assert status == 202 and duplicate['status'] == after['state']
        assert duplicate['origin_id'] == first['origin_id'] and duplicate['delivery_id'] == first['delivery_id']
        assert (await post(client, 'changed input'))[0] == 409
        assert len(calls) == 1
        assert authority.db._read_all('SELECT * FROM session_admissions') == []


@pytest.mark.asyncio
@pytest.mark.parametrize('preparing', [False, True])
async def test_linked_preparation_and_raw_process_copy_expire_with_source(monkeypatch, preparing):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    from hermes_state_runtime import claim_session_input, settle_session_input
    from hermes_state_mutation_retirement import RETRY_HORIZON_S
    from tools.process_registry import process_registry
    from tests.waiting import async_wait_until
    async with ingress(direct=False) as (client, adapter, authority):
        from gateway import webhook_origins
        original_state = webhook_origins._state
        def interrupted_link(authority, value, state, reason=None):
            if state == 'linked_native_admission':
                raise OSError('interrupted link publication')
            return original_state(authority, value, state, reason)
        with monkeypatch.context() as interruption:
            if preparing:
                interruption.setattr(webhook_origins, '_state', interrupted_link)
            assert (await post(client, PRIVATE))[0] == (503 if preparing else 202)
        linked, = records(authority)
        row = claim_session_input(authority.db, epoch=authority.epoch,
            session_id=authority.db._read_one('SELECT target_session_id FROM session_admissions')['target_session_id'])
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
            generation=row['generation'], outcome='completed',
            result={'result': {'final_response': PRIVATE, 'messages': []}, 'usage': {}})
        # Settlement is committed through the native store here, not its async observer.
        # Stop only this fixture's waiting finalizer, as settled_before_finalizer does.
        for task in list(adapter._background_tasks):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        native = row['payload']['native_text_v1']
        runner = authority.runner
        session = process_registry.spawn_local('printf "' + PRIVATE + '"',
            cwd=str(authority.db.db_path.parent), session_key=native['route'])
        try:
            await async_wait_until(lambda: session.exited, timeout=10, what='retention process exited')
            monkeypatch.setattr(runner, '_load_background_notifications_mode', lambda: 'result')
            watcher = dict(session_id=session.id, check_interval=0.01, platform='webhook',
                session_key=native['route'], chat_id=native['source']['chat_id'],
                parent_session_id=row['target_session_id'], notify_on_complete=False)
            await runner._watch_process(watcher)
        finally:
            if session.process is not None:
                session.process.wait(timeout=10)
        before = {v['origin_id']: v for v in records(authority)}
        assert {v['origin_kind'] for v in before.values()} == {
            'preparing' if preparing else 'linked_native_admission', 'process_final'}
        assert any(PRIVATE in v['message'] for v in before.values())
        at = time.time()
        delete_source(authority, row['target_session_id'])
        expire(authority, monkeypatch, at + RETRY_HORIZON_S + 1)
        for after in records(authority):
            assert after['message'] == '' and after.get('native') in (None, {})
            assert after.get('payload_compacted_at') is not None
            assert frozen(after) == frozen(before[after['origin_id']])
        assert PRIVATE not in ''.join(v['value'] for v in authority.db._read_all(
            "SELECT value FROM state_meta WHERE key GLOB 'webhook.*'"))
        await recover_bot_chat_finals(authority)
        status, duplicate = await post(client, PRIVATE)
        assert status == 202 and duplicate['origin_id'] == linked['origin_id']
        if not preparing:
            assert duplicate['source_admission_id'] == row['admission_id']
        else:
            assert duplicate['status'] == 'refused'
        assert authority.db._read_all('SELECT * FROM session_admissions') == []


@pytest.mark.asyncio
async def test_legacy_standalone_payload_ages_from_first_existing_maintenance_pass(monkeypatch):
    from hermes_state_mutation_retirement import RETRY_HORIZON_S
    async with ingress() as (client, _adapter, authority):
        status, first = await post(client, PRIVATE)
        assert status == 202
        # The previous persisted schema had no creation timestamp.
        authority.db._write_sql("UPDATE state_meta SET value=json_remove(value, '$.created_at') WHERE key=?",
                               ('webhook.origin.v1.' + first['origin_id'],))
        at = time.time()
        expire(authority, monkeypatch, at)
        assert records(authority)[0]['created_at'] == at
        expire(authority, monkeypatch, at + RETRY_HORIZON_S - 1)
        assert PRIVATE in records(authority)[0]['message']
        expire(authority, monkeypatch, at + RETRY_HORIZON_S + 1)
        assert records(authority)[0]['message'] == ''
        assert (await post(client, PRIVATE))[1]['status'] == 'refused'


@pytest.mark.asyncio
async def test_inflight_publication_cannot_restore_compacted_payload(monkeypatch):
    import threading
    from hermes_state_mutation_retirement import RETRY_HORIZON_S
    from tools import bot_live_delivery
    entered, release = threading.Event(), threading.Event()
    async with ingress() as (client, _adapter, authority):
        transport(monkeypatch, status='settled')
        send = bot_live_delivery.deliver_to_live_owner
        def held(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return send(*args, **kwargs)
        monkeypatch.setattr(bot_live_delivery, 'deliver_to_live_owner', held)
        request = asyncio.create_task(post(client, PRIVATE))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            before, = records(authority)
            assert PRIVATE in before['message']
            expire(authority, monkeypatch, before['created_at'] + RETRY_HORIZON_S + 1)
        finally:
            release.set()
        # A physical handoff that already started is not cancelled or claimed unsent.
        # Its stale observer cannot restore the producer payload or publish success.
        status, _ = await request
        assert status == 503
        after, = records(authority)
        assert after['message'] == '' and after['state'] == 'refused'
        assert frozen(after) == frozen(before)
