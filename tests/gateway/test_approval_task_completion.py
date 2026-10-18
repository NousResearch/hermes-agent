"""A real producer completion enters the FIFO with its original, revocable task."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_process_spawn_completion_reviewer_uses_live_origin(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal, Submission
    from gateway.session_local import create_local_session
    from hermes_state import AsyncSessionDB, SessionDB
    from tools import approval_context
    from tools.approval_context import current_approval_task_context
    from tools.approval_smart import _smart_approve
    from tools.process_registry import ProcessSession, process_registry
    from tools.terminal_tool_background import spawn_background_process

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(approval_context, '_get_approval_mode', lambda: 'smart')
    review_prompts = []

    def guardian(**kwargs):
        review_prompts.append(kwargs['messages'][1]['content'])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='APPROVE'))])

    monkeypatch.setattr('agent.auxiliary_client.call_llm', guardian)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = db
    runner._session_db = AsyncSessionDB(db)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner', db=db)
    viewer = Principal('test-owner', 'default', frozenset({'session:create', 'session:submit', 'session:read'}), 'viewer')
    ref = create_local_session(authority, viewer, dict(request_id='cli', source='cli', model='fixture', toolsets=[]))
    await authority.attach(viewer, ref)
    live = authority.sessions[ref.session_id]
    target = runner.session_store.lookup_by_session_key(live.route).session_id
    proc = ProcessSession(id='proc_approval_origin', command='printf done', session_key=live.route,
                          parent_session_id=target, started_at=1700000000.0,
                          watcher_platform='local', watcher_chat_id=live.source.chat_id,
                          watcher_user_id=live.source.user_id)
    monkeypatch.setattr('tools.terminal_tool_background._spawn', lambda *_a, **_k: proc)
    monkeypatch.setattr('tools.terminal_tool_background._register_completion_watcher', lambda *_a: None)
    with process_registry._lock:
        process_registry._running[proc.id] = proc
    observed = []

    async def answer(*, message, **_kwargs):
        if message == 'Build only src/cli.py':
            result = await asyncio.to_thread(spawn_background_process,
                command='printf done', env=SimpleNamespace(env={}), env_type='local',
                effective_task_id='fixture-task', task_id='fixture-task', session_key=live.route,
                workdir=None, cwd=str(tmp_path), effective_pty=False, notify_on_complete=True,
                watch_patterns=None, approval_note=None, pty_disabled_reason=None)
            assert json.loads(result)['session_id'] == proc.id
        elif message != 'Check status':
            def review():
                observed.append((current_approval_task_context(), _smart_approve('python -c "print(1)"', 'script')))
            await asyncio.to_thread(review)
        return {'final_response': 'ok', 'messages': []}

    monkeypatch.setattr(runner, '_run_agent', answer)
    try:
        await authority.submit(viewer, Submission('typed', ref, {'text': 'Build only src/cli.py'}, 'queue'))
        await async_wait_until(lambda: getattr(proc, 'approval_task_reservation', None) is not None,
                               timeout=10, what='process reservation')
        from hermes_state_runtime import get_session_admission
        origin = proc.approval_task_reservation.evidence
        await async_wait_until(lambda: get_session_admission(db, admission_id=origin.task_id)['status'] == 'terminal',
                               timeout=10, what='original turn settlement')
        await authority.submit(viewer, Submission('status', ref, {'text': 'Check status'}, 'queue'))
        from hermes_state_runtime import list_session_admissions
        await async_wait_until(lambda: any(row['request_id'] == 'status' and row['status'] == 'terminal'
                                           for row in list_session_admissions(db, session_id=ref.session_id,
                                                                              pending_only=False)),
                               timeout=10, what='newer ordinary turn settlement')
        assert proc.approval_task_reservation.lease.live()
        proc.exited = True
        proc.exit_code = 0
        watcher = {'session_key': live.route, 'platform': 'local', 'chat_type': 'dm',
                   'chat_id': live.source.chat_id, 'user_id': live.source.user_id,
                   'parent_session_id': target}
        event = runner._build_process_completion_event(watcher, proc, proc.id)
        delivered = await runner._deliver_completion_notification('[finished] done', event)
        assert delivered is True, (await runner._classify_completion_target(target, live.route),
                                   event['session_key'], event['parent_session_id'], event['chat_id'])
        await async_wait_until(lambda: bool(observed), timeout=10, what='completion reviewer')
        evidence, verdict = observed[0]
        assert evidence.text == 'Build only src/cli.py'
        assert evidence.task_id == origin.task_id
        assert len(evidence.ancestry) == 1 and evidence.ancestry[0].startswith('completion:')
        assert verdict == 'approve'
        await async_wait_until(lambda: proc.approval_task_reservation is None,
                               timeout=10, what='completion reservation settlement')
        assert 'Build only src/cli.py' in review_prompts[-1]
    finally:
        with process_registry._lock:
            process_registry._running.pop(proc.id, None)
        db.close()


@pytest.mark.asyncio
async def test_delegation_dispatch_completion_reviewer_uses_origin(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal, Submission
    from gateway.session_local import create_local_session
    from hermes_state import AsyncSessionDB, SessionDB
    from tools import async_delegation as delegation
    from tools.approval_context import current_approval_task_context
    from tools.approval_smart import _smart_approve
    from tools.delegate_tool_dispatch import _dispatch_unit
    from tools.process_registry import process_registry

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    delegation._reset_for_tests()
    monkeypatch.setattr('agent.auxiliary_client.call_llm', lambda **_k: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='APPROVE'))]))
    monkeypatch.setattr('tools.delegate_tool_dispatch._execute_and_aggregate',
                        lambda *_a, **_k: {'results': [{'status': 'completed', 'summary': 'done'}]})
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = db
    runner._session_db = AsyncSessionDB(db)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner', db=db)
    viewer = Principal('test-owner', 'default', frozenset({'session:create', 'session:submit', 'session:read'}), 'viewer')
    ref = create_local_session(authority, viewer, dict(request_id='cli', source='cli', model='fixture', toolsets=[]))
    await authority.attach(viewer, ref)
    live = authority.sessions[ref.session_id]
    target = runner.session_store.lookup_by_session_key(live.route).session_id
    handle, observed = [], []

    async def answer(*, message, **_kwargs):
        if message == 'Implement only src/cli.py':
            unit = SimpleNamespace(children=[(0, {'goal': 'work'}, SimpleNamespace())],
                                   task_list=[{'goal': 'work'}], context=None, top_role='worker',
                                   creds={'model': 'fixture'}, live_writers=[])
            routing = dict(session_key=live.route, origin_ui_session_id='', origin_session_id='',
                           parent_session_id=target, max_async_children=1)
            handle.append(await asyncio.to_thread(_dispatch_unit, unit, 'deleg_approval_origin', None, routing))
        else:
            observed.append(await asyncio.to_thread(lambda: (
                current_approval_task_context(), _smart_approve('python -c "print(1)"', 'script'))))
        return {'final_response': 'ok', 'messages': []}

    monkeypatch.setattr(runner, '_run_agent', answer)
    try:
        await authority.submit(viewer, Submission('typed', ref, {'text': 'Implement only src/cli.py'}, 'queue'))
        await async_wait_until(lambda: bool(handle), timeout=10, what='delegation dispatch')
        assert handle[0]['status'] == 'dispatched', handle[0]
        from hermes_state_runtime import list_session_admissions
        await async_wait_until(lambda: list_session_admissions(db, session_id=ref.session_id,
                              pending_only=False)[0]['status'] == 'terminal', timeout=10,
                              what='origin settlement')
        event = await asyncio.to_thread(process_registry.completion_queue.get, True, 10)
        assert event['delegation_id'] == 'deleg_approval_origin'
        assert await runner._deliver_completion_notification('delegation done', event) is True
        await async_wait_until(lambda: bool(observed), timeout=10, what='delegation continuation reviewer')
        evidence, verdict = observed[0]
        assert evidence.text == 'Implement only src/cli.py'
        assert evidence.ancestry[0].startswith('completion:')
        assert verdict == 'approve'
        await async_wait_until(lambda: delegation.approval_reservation_for_completion(
            'deleg_approval_origin')[1] is None, timeout=10, what='delegation reservation settlement')
    finally:
        delegation._reset_for_tests()
        db.close()
