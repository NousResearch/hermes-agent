"""Managed Stop uses the worker's registry, retaining route and raw-owner isolation."""
import shlex
import sys
import threading
from types import SimpleNamespace
from io import BytesIO

import pytest

from tests.waiting import wait_until


@pytest.fixture
def registry(monkeypatch):
    from tools import process_registry as module
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    yield registry
    for process in list(registry._running.values()):
        registry.kill_process(process.id)


def _spawn(registry, root, owner, route, persist=False):
    ready = root / f'managed-ready-{owner}-{route}-{len(registry._running)}'
    body = f'import pathlib,threading; pathlib.Path({str(ready)!r}).touch(); threading.Event().wait(120)'
    command = 'exec ' + shlex.join([sys.executable, '-c', body])
    process = registry.spawn_local(command, cwd=str(root), task_id=owner, owner_task_id=owner,
                                  session_key=route, persist_on_release=persist)
    wait_until(ready.exists, timeout=20, what='managed Stop Python process ready')
    return process


def test_worker_stop_ends_own_persisted_processes_and_preserves_neighbors(tmp_path, registry):
    from agent.managed_worker import WorkerControls
    processes = [_spawn(registry, tmp_path, owner, route, persist) for owner, route, persist in [
            ('worker-session', 'worker-route', False), ('worker-session', 'worker-route', True),
            ('other-owner', 'worker-route', True), ('worker-session', 'other-route', True)]]
    controls = WorkerControls.__new__(WorkerControls)
    controls.route, controls.stopped = 'worker-route', threading.Event()
    controls.stop_processes_requested = False
    controls.lock, controls.task_checks, controls.clarifications = threading.Lock(), {}, {}
    interrupted = threading.Event()
    controls.agent = SimpleNamespace(session_id='worker-session', hard_interrupt=lambda *a, **k: interrupted.set())
    controls.stop()
    assert all(not p.exited for p in processes), 'lifecycle Stop preserves the persisted-process policy'
    controls.stop(processes=True)
    assert interrupted.is_set()
    assert [p.exited for p in processes] == [True, True, False, False]
    # Work that crossed the first Stop snapshot is also reaped at the turn boundary.
    late = _spawn(registry, tmp_path, 'worker-session', 'worker-route')
    controls.stop_processes()
    assert late.exited


def test_native_managed_stop_frame_sets_deadline_and_decodes_process_cleanup(tmp_path, monkeypatch, registry):
    from agent.managed_worker import WorkerControls, encode_frame
    from gateway.session_managed_worker import ManagedWorker, interrupt_managed
    process = _spawn(registry, tmp_path, 's', 'route', True)
    worker = ManagedWorker(None)  # production command queue and deadline latch, without a pipe writer
    from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority
    from hermes_state_runtime import admit_session_input, claim_session_input
    db, authority = _authority(tmp_path, monkeypatch)
    authority._managed_workers = {'s': worker}
    admit_session_input(db, epoch=authority.epoch, principal_id=ACTOR.subject, session_id='s',
                        request_id='stopped', payload={'text': 'work'})
    row = claim_session_input(db, epoch=authority.epoch, session_id='s')
    authority._stamp_claim(REF, authority.sessions['s'], row)
    assert interrupt_managed(authority, ACTOR, REF, row['generation'])
    assert worker.stop.is_set() and worker.stopped_at is not None
    frames = worker.commands.get_nowait()
    controls = WorkerControls.__new__(WorkerControls)
    controls.route, controls.stopped, controls.finish = 'route', threading.Event(), threading.Event()
    controls.stop_processes_requested = False
    controls.lock, controls.task_checks, controls.clarifications = threading.Lock(), {}, {}
    controls.agent = SimpleNamespace(session_id='s', hard_interrupt=lambda *a, **k: None)
    monkeypatch.setattr(sys, 'stdin', SimpleNamespace(buffer=BytesIO(encode_frame(frames) + encode_frame({'type': 'finish'}))))
    controls.read()
    assert controls.finish.is_set() and controls.stopped.is_set()
    assert process.exited and process.completion_reason == 'killed'
    db.close()
