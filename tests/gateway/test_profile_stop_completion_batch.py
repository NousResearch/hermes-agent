"""Durable Stop filters every producer in its profile before completion batching."""
import asyncio
from dataclasses import replace
import pytest
from tests.gateway.test_child_notice_profile_admission import _build_rig, _for_each_leg

@pytest.mark.asyncio
@pytest.mark.parametrize('enqueue_scope', ['root', 'a', 'owner'])
async def test_consumed_result_is_removed_from_secondary_profile_batch(monkeypatch, tmp_path, request, private_db_probe_cleanup, enqueue_scope):
    from gateway.run import _profile_runtime_scope
    from tools import process_registry as module
    from tools.process_registry_results import save_completed_result
    rig = _build_rig(monkeypatch, tmp_path, request, True, 'all')
    runner = rig.runner
    async def consume(leg):
        # This is the durable result left by an earlier explicit Stop. A fresh
        # interpreter has no matching process-local _completion_consumed entry.
        with _profile_runtime_scope(rig.homes[leg.name]):
            assert not save_completed_result(replace(leg.proc, exited=True, completion_consumed=True))
            assert rig.registry.is_completion_consumed(leg.proc.id)
        assert leg.proc.id not in rig.registry._completion_consumed
        event = runner._build_process_completion_event(leg.watcher, leg.proc, leg.proc.id)
        event['output'] = 'CANCELLED_RESULT'
        companion = {**event, 'session_id': 'proc_companion' + leg.name + str(leg.index), 'output': 'LIVE_RESULT'}
        foreign = rig.homes['b' if leg.name == 'a' else 'a']
        with _profile_runtime_scope(foreign):
            assert not save_completed_result(replace(leg.proc, id=companion['session_id'], exited=True, completion_consumed=True))
        monkeypatch.setattr(module, 'process_registry', module.ProcessRegistry())
        assert not module.process_registry._completion_consumed
        async def enqueue():
            return await asyncio.gather(
                runner._enqueue_process_completion_notification('CANCELLED_RESULT', event),
                runner._enqueue_process_completion_notification('LIVE_RESULT', companion))
        if enqueue_scope != 'root':
            with _profile_runtime_scope(rig.homes[leg.name if enqueue_scope == 'owner' else enqueue_scope]):
                result = await enqueue()
        else:
            result = await enqueue()
        assert len(rig.admitted) == 1, (result, rig.admitted)
        text = rig.admitted[0][0].text
        print({'profile': leg.name, 'result': result, 'delivered_text': text})
        assert 'LIVE_RESULT' in text
        assert 'CANCELLED_RESULT' not in text, 'stopped completion contaminated another process completion'
    await _for_each_leg(rig, 'foreground', True, 'batch_child_first', consume)


@pytest.mark.asyncio
async def test_native_profile_batch_filters_before_fifo_and_preserves_human(tmp_path, monkeypatch, request, private_db_probe_cleanup):
    import shlex
    import sys
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import _profile_runtime_scope
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import initialize_session_authority, SessionAuthority
    from gateway.session_automation import completion_admission, producer_identity
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_contract import Principal
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.session_ingress_context import native_callback
    from tests.gateway.test_native_stop_reconciliation import caller, stop
    from tools import process_registry as module
    from tools.process_registry_results import load_completed_results, save_completed_result

    schedule = SessionAuthority._schedule
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    rig = _build_rig(monkeypatch, tmp_path, request, True, 'all')
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    adapters = {name: DiscordAdapter(PlatformConfig(enabled=True, token=f'fixture-{name}', typing_indicator=False))
                for name in ('default', 'a', 'b')}
    for adapter in adapters.values():
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id='fixture'))
    runner = rig.runner
    from tests.waiting import async_wait_until
    batch_ready = asyncio.Event()
    flush = runner._flush_process_completion_batch
    async def flush_after_all_producers(key):
        # Profile I/O can outlast the production coalescing window. Exercise
        # one real batch deterministically without requiring a quiet runner.
        await batch_ready.wait()
        await flush(key)
    monkeypatch.setattr(runner, '_flush_process_completion_batch', flush_after_all_producers)
    runner.config._runtime_profile_homes = tuple([('default', tmp_path), *rig.homes.items()])
    runner.adapters = {Platform.DISCORD: adapters['default']}
    runner._profile_adapters = {name: {Platform.DISCORD: adapters[name]} for name in rig.homes}
    runner._wire_adapter_handlers(adapters['default'])
    for name in rig.homes:
        runner._configure_profile_adapter(adapters[name], name, Platform.DISCORD)
    authorities = SessionAuthorities(tmp_path, multiplexed=True)
    runner.session_authorities = authorities
    for name, home in [('default', tmp_path), *rig.homes.items()]:
        (home / '.env').write_text('DISCORD_ALLOWED_USERS=42\n')
        with _profile_runtime_scope(home):
            source = adapters[name].build_source(chat_id='42', chat_type='dm', user_id='42')
            db = runner.session_store._db_for_key(runner.session_store._generate_session_key(source))
            authority = await initialize_session_authority(runner, profile_id=str(home), instance_id=name,
                                                           db=db, register=name == 'default')
            authorities.add(home, authority, name=name)
    async def admit_human(authority, event, name):
        with native_callback(runner, event, rig.homes[name], profile=name):
            return await authority.admit_native(event)
    executed = []
    async def execute(owner, ref, row):
        executed.append((get_hermes_home(), row['request_id'], row['payload']['text']))
        return 'fixture answer'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        for index, name in enumerate(('a', 'b', 'a')):
            home = rig.homes[name]
            authority = authorities.for_home(home)
            source = adapters[name].build_source(chat_id='42', chat_type='dm', user_id='42')
            assert source.profile == name
            with _profile_runtime_scope(home):
                seed = await admit_human(authority, MessageEvent(text='seed', source=source, message_id=f'seed-{index}'), name)
                actor = Principal('42', authority.profile_id, frozenset({'session:read', 'session:submit', 'session:control'}), 'fixture')
                await authority.cancel_queued(actor, seed.ref, seed.admission_id)
                registry = module.ProcessRegistry()
                monkeypatch.setattr(module, 'process_registry', registry)
                route = authority.sessions[seed.ref.session_id].route
                async def completed(label):
                    tokens = set_session_vars(session_id=seed.ref.session_id, ui_session_id=seed.ref.session_id, session_key=route)
                    try:
                        process = registry.spawn_local('exec ' + shlex.join([sys.executable, '-c', f'print({label!r})']),
                            cwd=str(home), owner_task_id=seed.ref.session_id, task_id=seed.ref.session_id, session_key=route)
                        process.notify_on_complete = True
                    finally:
                        clear_session_vars(tokens)
                    assert await asyncio.to_thread(process._completion_event.wait, 20)
                    event = dict(type='completion', session_id=process.id, session_key=route,
                        parent_session_id=seed.ref.session_id, started_at=process.started_at, platform='discord',
                        chat_type='dm', chat_id='42', user_id='42', output=process.output_buffer,
                        command=process.command, exit_code=0, completion_reason='exited')
                    return process, event
                cancelled, cancelled_event = await completed(f'CANCELLED_RESULT_{index}')
                assert 'result' in await stop(caller(authority, actor), seed.ref)
                assert registry.is_completion_consumed(cancelled.id)
                _already, already_event = await completed(f'ALREADY_RESULT_{index}')
                live, live_event = await completed(f'LIVE_RESULT_{index}')
                assert await runner._deliver_completion_notification(f'ALREADY_RESULT_{index}', already_event) is True
                already_row = completion_admission(runner, already_event)
                assert already_row is not None
                before = list_session_admissions(authority.db, session_id=seed.ref.session_id, pending_only=False)
            foreign = rig.homes['b' if name == 'a' else 'a']
            with _profile_runtime_scope(foreign):
                assert not save_completed_result(replace(live, completion_consumed=True))
            monkeypatch.setattr(module, 'process_registry', module.ProcessRegistry())
            assert not module.process_registry._completion_consumed
            batch_ready.clear()
            with _profile_runtime_scope(rig.homes['a']):
                pending = asyncio.gather(*(
                    runner._enqueue_process_completion_notification(event['output'], event)
                    for event in (cancelled_event, already_event, live_event)))
                await async_wait_until(lambda: sum(map(len, runner._completion_notification_batches.values())) == 3,
                                       timeout=20)
                batch_ready.set()
                results = await pending
            assert all(result is True for result in results), results
            with _profile_runtime_scope(home):
                rows = list_session_admissions(authority.db, session_id=seed.ref.session_id, pending_only=False)
                added = [row for row in rows if row['admission_id'] not in {prior['admission_id'] for prior in before}]
                assert len(added) == 1
                text = added[0]['payload']['text']
                assert f'LIVE_RESULT_{index}' in text and 'CANCELLED_RESULT' not in text and 'ALREADY_RESULT' not in text
                descriptor = added[0]['payload'].get('local_automation_v1') or added[0]['payload']['native_text_v1']['automation']
                assert descriptor.get('identities', [descriptor.get('identity')]) == [producer_identity(runner, live_event)]
                assert completion_admission(runner, already_event)['admission_id'] == already_row['admission_id']
                retained = load_completed_results(owns_entry=lambda row: row['id'] == cancelled.id, strict=True)[cancelled.id]
                assert retained.exit_code == 0 and f'CANCELLED_RESULT_{index}' in retained.output_buffer
                follower = await admit_human(authority, MessageEvent(text='human continuation', source=source,
                                                                       message_id=f'follow-{index}'), name)
                monkeypatch.setattr(SessionAuthority, '_schedule', schedule)
                authority._schedule(seed.ref)
                await asyncio.wait_for(authority.sessions[seed.ref.session_id].task, 20)
                monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
            leg = [item for item in executed if item[0] == home and item[1] in
                   {already_row['request_id'], added[0]['request_id'], f'follow-{index}'}]
            assert [item[1] for item in leg] == [already_row['request_id'], added[0]['request_id'], f'follow-{index}']
            assert all('CANCELLED_RESULT' not in item[2] for item in leg)
            assert follower.ref.session_id == seed.ref.session_id
    finally:
        await runner._cancel_process_completion_batch_tasks()
        for authority in authorities:
            authority.db.close()
