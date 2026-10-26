"""Saved request readback cannot retire newer work or revive a partial stale claim."""
import asyncio
from contextlib import closing
from dataclasses import replace
import shlex
import sys

import pytest

from tests.gateway.test_kanban_finite_admission import _CONFIG

pytest_plugins = ['tests.gateway.test_kanban_finite_admission']


@pytest.mark.parametrize('partial', [False, True], ids=['committed-readback', 'partial-before-input'])
def test_saved_old_request_never_controls_successor(kanban_owner, monkeypatch, tmp_path, partial):
    from gateway.session_kanban import run_task
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_state_runtime import RuntimeStoreError
    from tools import process_registry as module
    from gateway.session_context import clear_session_vars, set_session_vars

    owner = kanban_owner(_CONFIG)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    monkeypatch.setattr('hermes_cli.kanban_process_stop.request_native_stop', lambda *a: None)

    async def scenario():
        process = None
        try:
            original_submit = owner.authority.submit
            if partial:
                async def unavailable(*a, **kw):
                    raise RuntimeError('fixture failure before native input commit')
                monkeypatch.setattr(owner.authority, 'submit', unavailable)
                with pytest.raises(RuntimeError, match='before native input commit'):
                    await run_task(owner.connection, owner.params)
                monkeypatch.setattr(owner.authority, 'submit', original_submit)
                old = None
            else:
                old = await run_task(owner.connection, owner.params)
            with closing(connect(board='owned')) as conn:
                assert kb.block_task(conn, owner.params['task_id'], expected_run_id=owner.params['run_id'])
                assert kb.unblock_task(conn, owner.params['task_id'])
                task = kb.claim_task(conn, owner.params['task_id'])
            newer = await run_task(owner.connection, {
                **owner.params, 'run_id': task.current_run_id, 'claim_lock': task.claim_lock})
            sid = newer['session_id']
            route = owner.authority.sessions[sid].route
            tokens = set_session_vars(session_id=sid, session_key=route)
            try:
                process = registry.spawn_local(
                    shlex.join([sys.executable, '-c', 'import threading; threading.Event().wait(60)']),
                    cwd=str(tmp_path), owner_task_id=sid, task_id=sid, session_key=route, persist_on_release=True)
            finally:
                clear_session_vars(tokens)
            assert not process.exited
            calls = []
            original_resume = owner.connection.resume
            async def resumed(*args, **kwargs):
                calls.append('resume')
                return await original_resume(*args, **kwargs)
            async def submitted(*args, **kwargs):
                calls.append('submit')
                return await original_submit(*args, **kwargs)
            monkeypatch.setattr(owner.connection, 'resume', resumed)
            monkeypatch.setattr(owner.authority, 'submit', submitted)
            if partial:
                with pytest.raises(RuntimeStoreError, match='stale_kanban_claim'):
                    await run_task(owner.connection, owner.params)
            else:
                replay = await run_task(owner.connection, owner.params)
                assert replay == old
                with monkeypatch.context() as permission:
                    permission.setattr(owner.connection, 'actor', replace(owner.connection.actor,
                        capabilities=owner.connection.actor.capabilities - {'session:read'}))
                    with pytest.raises(RuntimeStoreError, match='permission_denied'):
                        await run_task(owner.connection, owner.params)
            assert calls == [], 'readback/stale request resumed or resubmitted an old execution'
            assert not process.exited, 'old saved replay killed a process owned by a newer native run'
        finally:
            if process is not None:
                registry.kill_process(process.id)
            await owner.close()
    asyncio.run(scenario())
