"""Native process output reaches durable owner admission without a model/network."""
import asyncio
import json
import shlex
import sys

import pytest

from gateway.run import GatewayRunner
from tests.waiting import async_wait_until


async def owner(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import initialize_session_authority, SessionAuthority
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from plugins.platforms.discord.adapter import DiscordAdapter

    # Real admission transaction; stop at the model scheduling boundary, no network.
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='work-results')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    event = MessageEvent(text='human', source=source, message_id='human')
    with native_callback(runner, event, get_hermes_home()):
        await authority.admit_native(event)
    key = runner.session_store._generate_session_key(source)
    sid = runner.session_store.peek_session_id(key)
    assert sid is not None
    return runner, authority, key, sid


def frame(work='build', incarnation='1', evidence='fixture-result', status='completed', **extra):
    return 'HERMES_WORK_RESULT_V1 ' + json.dumps(dict(
        work_id=work, incarnation=incarnation, status=status, evidence=evidence, **extra)) + '\n'


@pytest.mark.asyncio
async def test_live_enrolled_process_result_reaches_native_admission(monkeypatch):
    from gateway.session_context import scoped_current_session_id
    from hermes_state_runtime import list_session_admissions
    from tools.process_registry import ProcessRegistry

    runner, authority, key, sid = await owner(monkeypatch)
    registry = ProcessRegistry()
    code = 'import time; print(' + repr(frame()) + ', end="", flush=True); time.sleep(30)'
    enrollment = {'completion_work': [{'work_id': 'build', 'incarnation': '1'}]}
    with scoped_current_session_id(sid):
        source = registry.spawn_local(shlex.quote(sys.executable) + ' -c ' + shlex.quote(code),
                                      session_key=key, task_id='owner', **enrollment)
    try:
        source._watch_disabled = True
        await async_wait_until(lambda: 'fixture-result' in source.output_buffer,
                               timeout=5, what='framed result in output')
        assert 'fixture-result' in source.output_buffer
        assert not source.exited
        await runner._drain_watch_notifications(registry.completion_queue)
        rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        assert len(rows) == 2, 'explicit work result must commit while its source still runs'
        assert 'fixture-result' in rows[-1]['payload']['text']
        assert json.loads(rows[-1]['request_id'])[0] == 'process_work_result'
        await runner._drain_watch_notifications(registry.completion_queue)
        assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 2
    finally:
        registry.kill_process(source.id)
        source._reader_thread.join(timeout=3)


def enrolled(key, sid, process='proc_test', work=None):
    from tools.process_registry import ProcessRegistry, ProcessSession
    from tools.process_work_results import enroll
    registry = ProcessRegistry()
    session = ProcessSession(id=process, command='fixture', session_key=key, parent_session_id=sid)
    enroll(session, work or [{'work_id': 'build', 'incarnation': '1'}])
    return registry, session


@pytest.mark.asyncio
@pytest.mark.parametrize('after_admission', [False, True])
async def test_duplicate_is_idempotent_but_conflict_is_durable_failure(monkeypatch, after_admission):
    from tools.process_work_results import records
    from hermes_state_runtime import list_session_admissions
    runner, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    registry._ingest_output(source, frame())
    registry._ingest_output(source, frame())
    first = records(authority.db)[0]
    assert first['disposition'] == 'pending'
    if after_admission:
        await runner._drain_watch_notifications(registry.completion_queue)
    registry._ingest_output(source, frame(evidence='conflicting-result'))
    result = records(authority.db)[0]
    assert result.get('error') == 'conflicting_terminal_result'
    assert result['payload'] == first['payload']
    await runner._drain_watch_notifications(registry.completion_queue)
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == (2 if after_admission else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['stop', 'new', 'compression'])
async def test_user_boundaries_drop_but_compression_keeps_original_obligation(monkeypatch, boundary):
    from tools.process_work_results import records
    from hermes_state_runtime import list_session_admissions
    runner, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    if boundary == 'stop':
        authority.fence_route(key)
    elif boundary == 'new':
        runner.session_store.reset_session(key)
    else:
        authority.db.append_message(sid, 'user', 'human')
        child = sid + '-child'
        authority.db.publish_compression_child(parent_session_id=sid, child_session_id=child, source='discord',
            messages=[{'role': 'user', 'content': 'summary'}], require_compression_lease=False)
        assert runner.session_store.advance_compression_session(key, sid, child)
    registry._ingest_output(source, frame())
    await runner._drain_watch_notifications(registry.completion_queue)
    result = records(authority.db)[0]
    assert result['disposition'] == ('admitted' if boundary == 'compression' else 'dropped')
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == (2 if boundary == 'compression' else 1)


@pytest.mark.asyncio
async def test_stop_wins_inside_admission_transaction(monkeypatch):
    from tools.process_work_results import records
    from gateway import process_work_results as delivery
    from hermes_state_runtime import list_session_admissions
    runner, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    registry._ingest_output(source, frame())
    real = delivery.admit_automation

    async def racing_stop(*args, **kwargs):
        authority.fence_route(key)
        return await real(*args, **kwargs)
    monkeypatch.setattr(delivery, 'admit_automation', racing_stop)
    await runner._drain_watch_notifications(registry.completion_queue)
    assert records(authority.db)[0]['disposition'] == 'dropped'
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 1


@pytest.mark.asyncio
async def test_unknown_malformed_oversized_and_partial_records_never_wake(monkeypatch):
    from tools.process_work_results import records, MAX_FRAME
    from hermes_state_runtime import list_session_admissions
    runner, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    for bad in [frame(work='unknown'), frame(status='running'), frame(route='evil'),
                'HERMES_WORK_RESULT_V1 {bad}\n', frame(evidence='x' * (MAX_FRAME + 1)),
                'x' * (MAX_FRAME + 1), frame()]:
        # Last valid-looking frame remains part of an overlong unterminated line.
        for start in range(0, len(bad), 7):
            registry._ingest_output(source, bad[start:start + 7])
    await runner._drain_watch_notifications(registry.completion_queue)
    assert records(authority.db)[0]['disposition'] == 'enrolled'
    assert len(source._work_buffer) <= MAX_FRAME
    good = frame(evidence='opaque /do/not/read $(do-not-execute) 漢字')
    for char in good[:-1]:
        registry._ingest_output(source, char)
    await runner._drain_watch_notifications(registry.completion_queue)
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 1
    registry._ingest_output(source, '\n')
    await runner._drain_watch_notifications(registry.completion_queue)
    assert records(authority.db)[0]['disposition'] == 'admitted'


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['before', 'after', 'unknown'])
async def test_reopen_preserves_obligation_and_never_replays_admitted_or_unknown(monkeypatch, fault):
    from gateway.session_authority import initialize_session_authority
    from hermes_state_runtime import list_session_admissions, claim_session_input, settle_session_input
    from tools.process_work_results import records
    from gateway import process_work_results as delivery
    runner, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    registry._ingest_output(source, frame())
    if fault != 'before':
        real = delivery.admit_automation

        async def lost_ack(*args, **kwargs):
            await real(*args, **kwargs)
            raise RuntimeError('simulated lost ACK after SQLite commit')
        with monkeypatch.context() as patcher:
            patcher.setattr(delivery, 'admit_automation', lost_ack)
            await runner._drain_watch_notifications(registry.completion_queue)
        assert records(authority.db)[0]['disposition'] == 'admitted'
        if fault == 'unknown':
            first = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=first['admission_id'],
                                 generation=first['generation'], outcome='completed')
            assert claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)['status'] == 'started'
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='reopened')
    for _ in range(3):
        await runner._drain_watch_notifications(registry.completion_queue)
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(rows) == 2
    assert records(authority.db)[0]['disposition'] == 'admitted'
    if fault == 'unknown':
        # Existing native recovery settles unattended unknown automation interrupted,
        # never replays it. Do not change that policy for this producer.
        assert (rows[-1]['status'], rows[-1]['outcome']) == ('terminal', 'interrupted')


@pytest.mark.asyncio
async def test_spawn_log_survives_crash_before_ingestion_and_checkpoint_reopens(monkeypatch):
    from gateway.session_context import scoped_current_session_id
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import records
    runner, authority, key, sid = await owner(monkeypatch)
    original = ProcessRegistry()

    def crash_before_reader(session, *args, **kwargs):
        original._running[session.id] = session
        original._write_checkpoint()
    monkeypatch.setattr(original, '_track_started', crash_before_reader)
    code = 'print(' + repr(frame()) + ', end="", flush=True)'
    with scoped_current_session_id(sid):
        source = original.spawn_local(shlex.quote(sys.executable) + ' -c ' + shlex.quote(code),
            session_key=key, completion_work=[{'work_id': 'build', 'incarnation': '1'}])
    try:
        assert source.output_log, 'enrolled producers need existing durable log transport, not a lost pipe'
        source.process.wait(timeout=5)
        assert records(authority.db)[0]['disposition'] == 'enrolled'
        reopened = ProcessRegistry()
        assert reopened.recover_from_checkpoint() == 1
        recovered = reopened.get(source.id)
        recovered._reader_thread.join(timeout=5)
        await runner._drain_watch_notifications(reopened.completion_queue)
        assert records(authority.db)[0]['disposition'] == 'admitted'
        receipt = reopened.poll(source.id)
        assert receipt.get('work_results', [{}])[0].get('disposition') == 'admitted'
        assert receipt['work_results'][0]['acceptance'] == 'not_verified'
    finally:
        if source.process.poll() is None:
            source.process.kill()
        source.process.wait(timeout=5)


@pytest.mark.asyncio
async def test_duplicate_work_incarnation_cannot_be_enrolled_by_another_process(monkeypatch):
    from tools.process_work_results import enroll, records
    from tools.process_registry import ProcessSession
    _, authority, key, sid = await owner(monkeypatch)
    enrolled(key, sid)
    second = ProcessSession(id='proc_second', command='fixture', session_key=key, parent_session_id=sid)
    with pytest.raises(ValueError, match='already enrolled'):
        enroll(second, [{'work_id': 'build', 'incarnation': '1'}])
    assert len(records(authority.db)) == 1


@pytest.mark.asyncio
async def test_idle_drain_and_unavailable_adapter_never_start_a_model(monkeypatch):
    from tools.process_work_results import records
    runner, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    scheduled = []
    monkeypatch.setattr(authority, '_schedule', lambda ref: scheduled.append(ref))
    for _ in range(3):
        await runner._drain_watch_notifications(registry.completion_queue)
    assert scheduled == []
    registry._ingest_output(source, frame())
    adapters, runner.adapters = runner.adapters, {}
    for _ in range(3):
        await runner._drain_watch_notifications(registry.completion_queue)
    assert scheduled == []
    assert records(authority.db)[0]['disposition'] == 'pending'
    runner.adapters = adapters
    await runner._drain_watch_notifications(registry.completion_queue)
    assert len(scheduled) == 1


@pytest.mark.asyncio
async def test_existing_drain_services_each_profile_without_cross_owner_leak(tmp_path, monkeypatch):
    from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner
    from gateway.config import Platform, PlatformConfig, GatewayConfig
    from gateway.run import _async_profile_runtime_scope
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import initialize_session_authority, SessionAuthority
    from gateway.session_ingress_context import native_callback
    from gateway.session_identity import replace_source
    from gateway.platforms.event import MessageEvent
    from plugins.platforms.discord.adapter import DiscordAdapter
    from hermes_state_runtime import list_session_admissions
    from tools.process_work_results import records
    import queue

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    for _, home in homes:
        (home / '.env').write_text('DISCORD_ALLOWED_USERS=42\n')
    runner = _runner(root, homes)
    runner.session_authorities = SessionAuthorities(root, multiplexed=True)
    runner._profile_adapters = {}
    runner._profile_configs = {}
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    owners = []
    for name, home in homes:
        async with _async_profile_runtime_scope(home):
            monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
            authority = await initialize_session_authority(runner, profile_id=str(home), instance_id=name, register=False)
            runner.session_authorities.add(home, authority, name=name)
            adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
            if name == 'default':
                runner.session_authority = authority
                runner.adapters = {Platform.DISCORD: adapter}
            else:
                runner._profile_adapters[name] = {Platform.DISCORD: adapter}
                runner._profile_configs[name] = GatewayConfig()
            runner._wire_adapter_handlers(adapter)
            source = replace_source(adapter.build_source(chat_id='42', chat_type='dm', user_id='42'),
                                    profile=None if name == 'default' else name)
            event = MessageEvent(text='human-' + name, source=source, message_id='human')
            from gateway.session_ingress_context import register_transport_home
            profile = None if name == 'default' else name
            register_transport_home(runner, profile, home)
            with native_callback(runner, event, home, profile=profile):
                await authority.admit_native(event)
            key = runner.session_store._generate_session_key(source)
            sid = runner.session_store.peek_session_id(key)
            registry, process = enrolled(key, sid, process='proc_' + name)
            registry._ingest_output(process, frame(evidence='result-' + name))
            owners.append((authority, sid, name))
    register_transport_home(runner, None, root)
    # One unscoped existing drain must serve both, not just the launch profile.
    await runner._drain_watch_notifications(queue.Queue())
    for authority, sid, name in owners:
        rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        assert len(rows) == 2, name
        assert 'result-' + name in rows[-1]['payload']['text']
        assert records(authority.db)[0]['disposition'] == 'admitted'


@pytest.mark.asyncio
async def test_ambiguous_json_and_deep_nesting_are_rejected_without_killing_reader(monkeypatch):
    from tools.process_work_results import records
    runner, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    ambiguous = frame().replace('"status": "completed"', '"status":"failed","status":"completed"')
    registry._ingest_output(source, ambiguous)
    assert records(authority.db)[0]['disposition'] == 'enrolled'
    registry._ingest_output(source, 'HERMES_WORK_RESULT_V1 ' + '[' * 2000 + '0' + ']' * 2000 + '\n')
    registry._ingest_output(source, frame())
    await runner._drain_watch_notifications(registry.completion_queue)
    assert records(authority.db)[0]['disposition'] == 'admitted'


@pytest.mark.asyncio
async def test_enrollment_and_storage_bounds_fail_closed(monkeypatch):
    from tools import process_work_results as work
    from tools.process_registry import ProcessSession
    _, authority, key, sid = await owner(monkeypatch)
    for bindings in [[], [{}], [{'work_id': 'x', 'incarnation': True}],
                     [{'work_id': str(i), 'incarnation': '1'} for i in range(17)]]:
        with pytest.raises(ValueError):
            work.enroll(ProcessSession(id='proc_invalid', command='fixture', session_key=key, parent_session_id=sid), bindings)
    assert work.records(authority.db) == []
    monkeypatch.setattr(work, 'MAX_RECORDS', 1)
    enrolled(key, sid)
    with pytest.raises(ValueError, match='ledger full'):
        enrolled(key, sid, process='proc_overflow', work=[{'work_id': 'other', 'incarnation': '1'}])
    assert len(work.records(authority.db)) == 1


@pytest.mark.asyncio
async def test_protocol_frames_cannot_escape_into_readiness_notifications(monkeypatch):
    _runner, _authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    source.watch_patterns = ['fixture-result']
    registry._ingest_output(source, frame(work='unknown'))
    for text in [frame(), frame()]:
        for char in text:
            registry._ingest_output(source, char)
    assert registry.completion_queue.empty(), 'result frames are not readiness text'
    registry._ingest_output(source, 'ordinary fixture-result\n')
    assert registry.completion_queue.get_nowait()['type'] == 'watch_match'


@pytest.mark.asyncio
async def test_deep_json_does_not_terminate_ingestion_at_normal_recursion_limit(monkeypatch):
    from tools.process_work_results import records
    _, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    before = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(1000)
        registry._ingest_output(source, 'HERMES_WORK_RESULT_V1 ' + '[' * 2000 + '0' + ']' * 2000 + '\n')
    finally:
        sys.setrecursionlimit(before)
    registry._ingest_output(source, frame())
    assert records(authority.db)[0]['disposition'] == 'pending'


@pytest.mark.asyncio
async def test_durable_payload_redacts_secrets_without_erasing_conflicts(monkeypatch):
    from tools.process_work_results import records
    _, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    secret = 'sk-proj-' + 'X' * 40
    registry._ingest_output(source, frame(evidence=secret))
    assert secret not in records(authority.db)[0]['payload']
    registry._ingest_output(source, frame(evidence=secret))
    assert records(authority.db)[0]['error'] is None
    registry._ingest_output(source, frame(evidence='sk-proj-' + 'Y' * 40))
    assert records(authority.db)[0]['error'] == 'conflicting_terminal_result'


@pytest.mark.asyncio
async def test_terminal_public_opt_in_reaches_real_spawn_and_native_owner(monkeypatch, tmp_path):
    from gateway.session_context import scoped_current_session_id
    from tools import process_registry as processes
    from tools.terminal_tool import _handle_terminal
    from hermes_state_runtime import list_session_admissions
    runner, authority, key, sid = await owner(monkeypatch)
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, 'process_registry', registry)
    monkeypatch.setenv('TERMINAL_ENV', 'local')
    code = 'print(' + repr(frame()) + ', end="", flush=True)'
    with scoped_current_session_id(sid):
        result = json.loads(await asyncio.to_thread(_handle_terminal, {
            'command': shlex.quote(sys.executable) + ' -c ' + shlex.quote(code),
            'background': True, 'workdir': str(tmp_path),
            'completion_work': [{'work_id': 'build', 'incarnation': '1'}],
        }, task_id=key))
    assert result.get('error') is None, result
    source = registry.get(result['session_id'])
    try:
        source._reader_thread.join(timeout=5)
        await runner._drain_watch_notifications(registry.completion_queue)
        assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 2
        assert result.get('work_results'), 'spawn receipt must expose the finite enrolled obligations'
    finally:
        if source.process.poll() is None:
            registry.kill_process(source.id)


@pytest.mark.parametrize('extra', [{'background': False}, {'background': True, 'pty': True}])
def test_public_work_binding_refuses_unsupported_modes_before_execution(monkeypatch, extra):
    from tools import terminal_tool as terminal
    called = []
    monkeypatch.setattr(terminal, 'terminal_tool', lambda **kwargs: called.append(kwargs))
    result = json.loads(terminal._handle_terminal({'command': 'must not run',
        'completion_work': [{'work_id': 'build', 'incarnation': '1'}], **extra}))
    assert result.get('error') and called == []


def test_remote_work_binding_never_falls_back_to_an_unenrolled_spawn():
    from tools.terminal_tool_background import _spawn
    with pytest.raises(ValueError, match='local non-PTY'):
        _spawn(None, env=None, env_type='docker', command='must not run', cwd='/',
               effective_task_id='owner', task_id='owner', session_key='agent:main:discord:dm:42',
               effective_pty=False, completion_work=[{'work_id': 'build', 'incarnation': '1'}])


@pytest.mark.asyncio
async def test_recursive_json_decoder_failure_cannot_kill_result_reader(monkeypatch):
    # Python versions with a recursive C decoder can raise RecursionError, not ValueError.
    from tools import process_work_results as work
    _, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    valid = frame()
    def recursive_decoder(*args, **kwargs):
        raise RecursionError('JSON nesting')
    with monkeypatch.context() as patch:
        patch.setattr(work.json, 'loads', recursive_decoder)
        registry._ingest_output(source, valid)
    registry._ingest_output(source, valid)
    assert work.records(authority.db)[0]['disposition'] == 'pending'
