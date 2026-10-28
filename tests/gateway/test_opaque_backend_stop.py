"""Opaque backend wrapper results never acknowledge native job-scope cleanup."""
import asyncio
import json
import shlex
import sys
import threading
import uuid

import psutil
import pytest

from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit, _wire_turn_agent
from tests.gateway.test_session_authority_preadopt_stop import Agent
from tests.gateway.test_unknown_backend_stop_checkpoint import ShellBackend
from tests.waiting import wait_until


class Backend(ShellBackend):
    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=True)
        super().__init__(root)
        self.reader_entered, self.reader_release = threading.Event(), threading.Event()
        self.commands, self.fail_reads = [], False

    def execute(self, command, **kwargs):
        self.commands.append(command)
        if command.startswith('O='):
            self.reader_entered.set()
            if not self.reader_release.wait(90):
                raise TimeoutError('finite fixture reader gate')
            if self.fail_reads:
                raise OSError('fixture backend unavailable')
        return super().execute(command, **kwargs)


async def spawn(registry, backend, *, ignore_term=False, route='route', owner='s'):
    ready, release = backend.root / 'ready', backend.root / 'release'
    code = ('import os,signal,time; from pathlib import Path; '
            + ('signal.signal(signal.SIGTERM,signal.SIG_IGN); ' if ignore_term else '')
            + f'print("partial output",flush=True); Path({str(ready)!r}).write_text(str(os.getpid())); '
            + f'p=Path({str(release)!r}); end=time.monotonic()+120\n'
            + 'while not p.exists() and time.monotonic()<end: time.sleep(.01)\n'
            + 'print("final original output",flush=True)')
    session = await asyncio.to_thread(registry.spawn_via_env, backend,
        'exec ' + shlex.join([sys.executable, '-c', code]), cwd=str(backend.root),
        task_id=owner, owner_task_id=owner, session_key=route, persist_on_release=True)
    await asyncio.to_thread(wait_until, ready.exists, timeout=20)
    root = psutil.Process(session.pid)
    tree = [root, *root.children(recursive=True)]
    return session, release, tree


def alive(tree):
    for child in tree:
        try:
            if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            pass
    return False


async def cleanup(backend, release, tree, session):
    release.touch()
    backend.reader_release.set()
    for child in reversed(tree):
        try:
            if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                child.kill()
        except psutil.NoSuchProcess:
            pass
    await asyncio.to_thread(session._completion_event.wait, 15)


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass  # Exact native backend fixture PIDs, never host identity inference in production.
@pytest.mark.parametrize('control', ['direct', 'bulk', 'native'])
@pytest.mark.parametrize('ignore_term', [False, True], ids=['ordinary-child', 'ignores-term'])
async def test_opaque_stop_remains_unknown_through_reader_exit_and_reload(tmp_path, monkeypatch, control, ignore_term):
    from gateway.session_ancillary import process_stop_pending, retry_process_stop
    from tools import process_registry as module
    from tools.process_registry_results import load_completed_results, save_completed_result
    db, authority = _authority(tmp_path, monkeypatch)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    backend, foreign = Backend(tmp_path / 'own'), Backend(tmp_path / 'foreign')
    session, release, tree = await spawn(registry, backend, ignore_term=ignore_term)
    neighbor, neighbor_release, neighbor_tree = await spawn(registry, foreign, route='foreign-route', owner='foreign')
    agent = Agent()
    agent.session_id = 's'
    authority.runner._cached_agent_for = lambda route: agent
    started, finish = asyncio.Event(), asyncio.Event()
    executed = []

    async def execute(owner, ref, row):
        _wire_turn_agent(owner, row['generation'], agent)
        executed.append(row['request_id'])
        if row['request_id'] == 'head':
            started.set()
            await asyncio.wait_for(finish.wait(), 60)
        return 'finite fixture answer'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(registry, '_terminate_host_pid', lambda *a, **kw: pytest.fail('opaque PID signaled as host'))
    try:
        await _submit(authority, 'head')
        await asyncio.wait_for(started.wait(), 15)
        await _submit(authority, 'human follower')
        if control == 'direct':
            result = await asyncio.to_thread(registry.kill_process, session.id)
            assert (result['status'], result['error']) == ('error', 'process_stop_unknown')
        elif control == 'bulk':
            assert await asyncio.to_thread(registry.kill_all, 's', source='session.interrupt') == 0
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        finish.set()
        await asyncio.wait_for(authority.sessions['s'].task, 20)
        token = process_stop_pending(db, 's')
        assert token and executed == ['head'] and alive(tree)
        assert not session.exited and session._scope_stop_pending
        assert not any(command.startswith('kill ') for command in backend.commands)
        assert alive(neighbor_tree) and not neighbor._scope_stop_pending
        assert not registry.is_completion_consumed(neighbor.id)
        # The backend really completes, but its delayed reader has not run.
        release.touch()
        await asyncio.to_thread(wait_until, lambda: not alive(tree), timeout=20)
        assert not session.exited
        backend.reader_release.set()
        assert await asyncio.to_thread(session._completion_event.wait, 20)
        assert session.exit_code == 0 and session.completion_reason == 'exited'
        assert 'final original output' in session.output_buffer
        assert session._scope_stop_pending
        cold = module.ProcessRegistry()
        cold._finished.update(load_completed_results(owns_entry=lambda row: row['id'] == session.id, strict=True))
        assert cold._finished[session.id].pid_scope == 'sandbox'
        if control == 'direct' and not ignore_term:
            # A pre-upgrade cached receipt omitted backend scope. Neither its
            # checkpoint rewrite nor its late result can erase native evidence.
            stale = cold._finished[session.id]
            stale.pid_scope = 'host'
            assert cold._write_checkpoint() and save_completed_result(stale)
            cold._finished.update(load_completed_results(owns_entry=lambda row: row['id'] == session.id, strict=True))
            assert cold._finished[session.id].pid_scope == 'sandbox'
        assert cold._write_checkpoint()
        monkeypatch.setattr(module, 'process_registry', cold)
        assert cold.recover_from_checkpoint() == 0
        assert not await retry_process_stop(authority, REF)
        assert cold.kill_process(session.id)['error'] == 'process_stop_unknown'
        await _submit(authority, 'later human')
        await asyncio.wait_for(authority.sessions['s'].task, 20)
        assert executed == ['head'] and process_stop_pending(db, 's') == token
        rows = list_session_admissions(db, session_id='s')
        assert {row['status'] for row in rows if row['request_id'] != 'head'} == {'queued'}
        saved = json.loads(module._checkpoint_path().read_text())
        assert {session.id, neighbor.id} <= {row['session_id'] for row in saved}
        assert alive(neighbor_tree)
    finally:
        finish.set()
        await cleanup(backend, release, tree, session)
        await cleanup(foreign, neighbor_release, neighbor_tree, neighbor)
        db.close()


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize('backend_lost', [False, True], ids=['ordinary-completion', 'transport-lost'])
async def test_ordinary_backend_history_and_lost_resource_evidence(tmp_path, monkeypatch, backend_lost):
    from gateway.session_ancillary import process_stop_pending
    from tools import process_registry as module
    db, authority = _authority(tmp_path, monkeypatch)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    backend = Backend(tmp_path / uuid.uuid4().hex)
    session, release, tree = await spawn(registry, backend)
    executed = []

    async def execute(owner, ref, row):
        executed.append(row['request_id'])
        return 'finite fixture answer'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        if backend_lost:
            backend.fail_reads = True
        else:
            release.touch()
            await asyncio.to_thread(wait_until, lambda: not alive(tree), timeout=20)
        backend.reader_release.set()
        assert await asyncio.to_thread(session._completion_event.wait, 20)
        assert session.completion_reason == ('lost' if backend_lost else 'exited')
        assert session.exit_code == (-1 if backend_lost else 0)
        if not backend_lost:
            assert 'final original output' in session.output_buffer
        assert session.id in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        # Uncertainty is resource evidence, not an execution lock absent Stop.
        await _submit(authority, 'ordinary human continuation')
        await asyncio.wait_for(authority.sessions['s'].task, 15)
        assert executed == ['ordinary human continuation'] and process_stop_pending(db, 's') is None
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        await _submit(authority, 'held after explicit Stop')
        await asyncio.wait_for(authority.sessions['s'].task, 15)
        assert executed == ['ordinary human continuation'] and process_stop_pending(db, 's')
        assert alive(tree) == backend_lost
    finally:
        await cleanup(backend, release, tree, session)
        db.close()
