"""Validated child handoff transfers native Stop and retained-result authority durably."""
import asyncio
from dataclasses import replace
import json
import shlex
import sys
import threading
from types import SimpleNamespace
import weakref

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from hermes_state_runtime import begin_runtime_epoch
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority
from tests.waiting import wait_until


def _spawn(registry, root, owner, parent, route, *, logged=False, release=None):
    from gateway.session_context import clear_session_vars, set_session_vars
    from unittest.mock import patch
    from tools.process_registry_output_log import output_dir
    ready = root / ('ready-' + owner + '-' + parent + '-' + route)
    body = f'import pathlib,threading,time; pathlib.Path({str(ready)!r}).touch(); end=time.monotonic()+120\n'
    if release is None:
        body += 'threading.Event().wait(120)\n'
    else:
        body += f'while not pathlib.Path({str(release)!r}).exists() and time.monotonic()<end: threading.Event().wait(.01)\n'
    command = ('' if logged else 'exec ') + shlex.join([sys.executable, '-c', body + 'print("retained handoff output")'])
    tokens = set_session_vars(session_id=parent, ui_session_id='s', session_key=route)
    try:
        if logged:
            output_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
            with patch('tools.process_registry.worker_output_log',
                       side_effect=lambda sid: output_dir() / (sid + '.log')):
                process = registry.spawn_local(command, cwd=str(root), task_id=owner,
                    owner_task_id=owner, session_key=route, persist_on_release=True)
        else:
            process = registry.spawn_local(command, cwd=str(root), task_id=owner,
                owner_task_id=owner, session_key=route, persist_on_release=True)
    finally:
        clear_session_vars(tokens)
    wait_until(ready.exists, timeout=20, what='native handoff fixture ready')
    return process


def _child(parent):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.delegate_tool_child_run import _register_child
    child = SimpleNamespace(_delegate_parent_ref=weakref.ref(parent), _subagent_id='sa-0-handoff-stop',
        _delegate_depth=1, _parent_session_id=parent.session_id, _hard_interrupt_requested=threading.Event())
    tokens = set_session_vars(session_id='s', ui_session_id='s', session_key='route')
    try:
        sid = _register_child(child, parent, 'handoff fixture', owner_session_id=None,
                              owner_transport=None, owner_session_record=None)
    finally:
        clear_session_vars(tokens)
    return sid, child


class Parent:
    session_id = 'parent-worker-uuid'
    _current_task_id = 'parent-turn'
    _process_owner_task_ids = {'parent-turn'}
    _gateway_session_key = 'route'

    def hard_interrupt(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['native', 'managed', 'cold', 'late-finalize'])
async def test_handoff_after_child_retirement_survives_stale_writers_and_stop(tmp_path, monkeypatch, mode):
    from agent.managed_worker import WorkerControls
    from gateway.session_ancillary import process_stop_pending
    from tools import process_registry as module
    from tools.delegate_tool import _unregister_subagent
    from tools.process_registry_handoff import handoff_process
    from tools.process_registry_results import load_completed_results, save_completed_result
    db, authority = _authority(tmp_path, monkeypatch)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    parent = Parent()
    sid, _registered_child = _child(parent)
    gate = tmp_path / 'finish-handoff'
    process = _spawn(registry, tmp_path, sid, 'native-child-session', 'route', logged=True, release=gate)
    process.notify_on_complete = True
    registry._write_checkpoint()
    stale = module.ProcessRegistry()
    stale.recover_from_checkpoint()
    stale_process = stale._running[process.id]
    assert stale_process.parent_session_id == 'native-child-session'
    neighbors = [_spawn(registry, tmp_path, 'parent-turn', 'foreign-parent', 'route'),
                 _spawn(registry, tmp_path, 'parent-turn', 's', 'foreign-route')]
    try:
        snapshot, release_writer = threading.Event(), threading.Event()
        writes = []
        writer = None
        if mode == 'cold':
            from tools import process_registry_checkpoint as checkpoint
            redact = checkpoint.redact_sensitive_text
            def held_snapshot(*args, **kwargs):
                if threading.current_thread().name == 'stale-native-writer':
                    snapshot.set()
                    assert release_writer.wait(20)
                return redact(*args, **kwargs)
            monkeypatch.setattr(checkpoint, 'redact_sensitive_text', held_snapshot)
            writer = threading.Thread(target=lambda: writes.append(stale._write_checkpoint()),
                                      name='stale-native-writer', daemon=True)
            writer.start()
            assert await asyncio.to_thread(snapshot.wait, 20)
        try:
            result = handoff_process(process.id, {'data': 'parent durable watcher'}, sid)
        finally:
            release_writer.set()
            if writer is not None:
                await asyncio.to_thread(writer.join, 20)
                assert not writer.is_alive() and writes == [True]
        assert result['status'] == 'handed_off'
        assert process.parent_session_id == 's' and process.owner_task_id == 'parent-turn'
        assert process.ownership_generation == 1
        # A second interpreter adopted the child before handoff. Its subsequent
        # checkpoint may not overwrite the approved parent, including its cache.
        if mode != 'late-finalize':
            assert stale._write_checkpoint()
            assert stale_process.parent_session_id == 's' and stale_process.ownership_generation == 1
        row = next(row for row in json.loads(module._checkpoint_path().read_text()) if row['session_id'] == process.id)
        assert row['parent_session_id'] == 's' and row['ownership_generation'] == 1
        _unregister_subagent(sid)
        if mode == 'late-finalize':
            # Exercise the actual cached log reader's completion publication,
            # not an invented successful completion while a kernel job lives.
            gate.touch()
            assert await asyncio.to_thread(stale_process._completion_event.wait, 20)
            assert await asyncio.to_thread(process._completion_event.wait, 20)
            event, = [event for event in list(stale.completion_queue.queue) if event['session_id'] == process.id]
            assert event['parent_session_id'] == 's' and event['owner_task_id'] == 'parent-turn'
            from gateway.run import GatewayRunner
            watched = GatewayRunner._build_process_completion_event(
                {'parent_session_id': 'native-child-session', 'session_key': 'old-child-route'}, stale_process, process.id)
            assert watched['parent_session_id'] == 's' and watched['session_key'] == 'route'
            assert save_completed_result(replace(stale_process, parent_session_id='native-child-session',
                owner_task_id=sid, task_id=sid, ownership_generation=0))
            retained = load_completed_results(owns_entry=lambda row: row['id'] == process.id, strict=True)[process.id]
            assert retained.parent_session_id == 's' and retained.ownership_generation == 1
            assert retained.exit_code == 0 and 'retained handoff output' in retained.output_buffer
            assert not save_completed_result(replace(stale_process, owner_task_id=sid,
                parent_session_id='native-child-session', ownership_generation=0,
                pid=neighbors[0].pid, host_start_time=neighbors[0].host_start_time))
        if mode == 'managed':
            controls = WorkerControls.__new__(WorkerControls)
            controls.route, controls.session_id, controls.agent = 'route', 's', parent
            controls.stopped, controls.stop_processes_requested = threading.Event(), False
            controls.lock, controls.task_checks, controls.clarifications = threading.Lock(), {}, {}
            controls.stop(processes=True)
        else:
            if mode == 'cold':
                monkeypatch.setattr(module, 'process_registry', module.ProcessRegistry())
                source = authority.sessions['s'].source
                epoch = begin_runtime_epoch(db, instance_id='cold-handoff')
                authority = SessionAuthority(authority.runner, profile_id='owned', instance_id='cold-handoff', db=db, epoch=epoch)
                authority.sessions['s'] = LiveSession(source, 'route')
            authority.runner._cached_agent_for = lambda route: None
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
            assert process_stop_pending(db, 's') is None
        assert await asyncio.to_thread(process._completion_event.wait, 20)
        assert all(not neighbor.exited for neighbor in neighbors)
    finally:
        _unregister_subagent(sid)
        registry.kill_process(process.id)
        for neighbor in neighbors:
            registry.kill_process(neighbor.id)
        db.close()


def test_failed_handoff_write_retains_child_identity(tmp_path, monkeypatch):
    from tools import process_registry as module
    from tools.delegate_tool import _unregister_subagent
    from tools.process_registry_handoff import handoff_process
    import utils
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    parent = Parent()
    sid, _registered_child = _child(parent)
    process = _spawn(registry, tmp_path, sid, 'native-child-session', 'route')
    old = process.owner_task_id, process.parent_session_id, process.ownership_generation
    write = utils.atomic_json_write
    try:
        def refused(path, *args, **kwargs):
            if path == module._checkpoint_path():
                raise PermissionError('fixture native checkpoint write refusal')
            return write(path, *args, **kwargs)
        with monkeypatch.context() as failure:
            failure.setattr(utils, 'atomic_json_write', refused)
            assert 'error' in handoff_process(process.id, {'data': 'parent watcher'}, sid)
        assert old == (process.owner_task_id, process.parent_session_id, process.ownership_generation)
        row = next(row for row in json.loads(module._checkpoint_path().read_text()) if row['session_id'] == process.id)
        assert row['parent_session_id'] == old[1] and row['ownership_generation'] == old[2]
    finally:
        _unregister_subagent(sid)
        registry.kill_process(process.id)
