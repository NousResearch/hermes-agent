"""Native Stop reports uncertainty and wakes eligible work only after reconciliation."""
import asyncio
import shlex
import sys

import pytest

from gateway.session_ancillary import process_stop_pending, record_process_stop
from gateway.session_controls import AuthorityConnection
from hermes_state_runtime import begin_runtime_epoch, list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from tests.waiting import async_wait_until


def caller(authority, actor=ACTOR):
    connection = AuthorityConnection.__new__(AuthorityConnection)
    connection.authority, connection.actor = authority, actor
    connection._operator, connection._identity = False, {}
    return connection


async def stop(connection, ref=REF):
    return await connection.dispatch({'id': 1, 'method': 'session.interrupt', 'params': {
        'session_id': ref.session_id,
        'execution_generation': connection.authority._handle(ref).execution_generation}})


@pytest.mark.asyncio
async def test_native_refusal_is_visible_and_successful_retry_runs_saved_follower(tmp_path, monkeypatch):
    from gateway import session_automation
    from tools import process_registry as module
    db, authority = _authority(tmp_path, monkeypatch)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    command = shlex.join([sys.executable, '-c', 'import threading; threading.Event().wait(60)'])
    own = registry.spawn_local(command, cwd=str(tmp_path), task_id='s', owner_task_id='s', session_key='route')
    neighbor = registry.spawn_local(command, cwd=str(tmp_path), task_id='neighbor', owner_task_id='neighbor', session_key='route')
    kill, executed = registry.kill_process, []

    async def execute(owner, ref, row):
        assert own.exited and not neighbor.exited
        executed.append(row['request_id'])
        return 'done'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(registry, 'kill_process', lambda *a, **kw: {'status': 'error', 'error': 'fixture refusal'})
    try:
        response = await stop(caller(authority))
        assert response['error']['data']['reason'] == 'process_stop_unknown'
        assert process_stop_pending(db, 's') is not None and not own.exited
        await _submit(authority, 'saved-follower')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert executed == []
        monkeypatch.setattr(registry, 'kill_process', kill)
        withdraw = session_automation.withdraw_consumed_process_notices

        async def refused_withdraw(owner, ref):
            return False

        monkeypatch.setattr(session_automation, 'withdraw_consumed_process_notices', refused_withdraw)
        response = await stop(caller(authority))
        assert response['error']['data']['reason'] == 'process_stop_unknown'
        assert own.exited and not neighbor.exited and executed == []
        assert process_stop_pending(db, 's') is not None
        monkeypatch.setattr(session_automation, 'withdraw_consumed_process_notices', withdraw)
        response = await stop(caller(authority))
        if 'error' in response:
            # The actual completion may already have resumed the saved human
            # between this control's generation snapshot and its native CAS.
            assert response['error']['data']['reason'] == 'stale_generation'
            assert authority._handle(REF).execution_generation == 1
        await async_wait_until(lambda: executed == ['saved-follower'] and authority.sessions['s'].task.done(), timeout=10)
        assert process_stop_pending(db, 's') is None
        assert executed == ['saved-follower']
    finally:
        monkeypatch.setattr(registry, 'kill_process', kill)
        kill(own.id)
        kill(neighbor.id)
        db.close()


@pytest.mark.asyncio
async def test_fresh_native_managed_session_without_checkpoint_accepts_first_input(tmp_path, monkeypatch):
    from gateway.session_contract import Submission
    from gateway.session_local import create_local_session
    from gateway.session_policy import build_policy
    from gateway.session_managed_worker import managed_policy
    from tests.gateway.test_session_local_reopen_on_turn import _ended_local_session
    from tools import process_registry as module
    authority, connection, _earlier = await _ended_local_session(tmp_path, monkeypatch)
    params = {'request_id': 'fresh-managed', 'source': 'tui', 'cwd': str(tmp_path),
              'model': 'm', 'toolsets': [], 'ignore_user_config': True}
    policy = build_policy(params, {}, profile_terminal=False)
    ref = create_local_session(authority, connection.actor, params, trusted_policy=policy)
    authority.runner._adapter_for_source = lambda source: authority.runner.adapters.get(source.platform)
    monkeypatch.setattr(module, 'process_registry', module.ProcessRegistry())
    assert managed_policy(authority, ref) is not None and not module._checkpoint_path().exists()
    executed = []

    async def execute(owner, target, row):
        executed.append(row['request_id'])
        return 'first answer'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        assert 'result' in await stop(connection, ref)
        assert process_stop_pending(authority.db, ref.session_id) is None
        await authority.submit(connection.actor, Submission('first-input', ref, {'text': 'hello'}, 'queue'))
        await asyncio.wait_for(authority.sessions[ref.session_id].task, 10)
        assert executed == ['first-input']
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('evidence', ['old-missing', 'malformed', 'unreadable'])
async def test_repeated_native_stop_preserves_unresolved_evidence(tmp_path, monkeypatch, evidence):
    from tools import process_registry as module
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(module, 'process_registry', module.ProcessRegistry())
    token = record_process_stop(authority, REF, 0)
    checkpoint = module._checkpoint_path()
    if evidence == 'old-missing':
        authority.epoch = begin_runtime_epoch(db, instance_id='reloaded')
        assert not checkpoint.exists()
    else:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text('not valid checkpoint JSON' if evidence == 'malformed' else '[]')
        if evidence == 'unreadable':
            from pathlib import Path
            read = Path.read_text

            def unreadable(path, *args, **kwargs):
                if path == checkpoint:
                    raise PermissionError('fixture checkpoint read refusal')
                return read(path, *args, **kwargs)

            monkeypatch.setattr(Path, 'read_text', unreadable)
    try:
        for _attempt in range(2):
            response = await stop(caller(authority))
            assert response['error']['data']['reason'] == 'process_stop_unknown'
            assert process_stop_pending(db, 's') == token
        assert list_session_admissions(db, session_id='s') == []
    finally:
        db.close()
