"""Managed Stop uses the worker's registry, retaining route and raw-owner isolation."""
import shlex
import sys
import threading
from types import SimpleNamespace
from io import BytesIO

import pytest


@pytest.fixture
def registry(monkeypatch):
    from tools import process_registry as module
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    yield registry
    for process in list(registry._running.values()):
        registry.kill_process(process.id)


def test_worker_stop_ends_own_persisted_processes_and_preserves_neighbors(tmp_path, registry):
    from agent.managed_worker import WorkerControls
    command = shlex.join([sys.executable, '-c', 'import threading; threading.Event().wait(40)'])
    processes = [registry.spawn_local(command, cwd=str(tmp_path), task_id=owner, owner_task_id=owner,
        session_key=route, persist_on_release=persist) for owner, route, persist in [
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
    late = registry.spawn_local(command, cwd=str(tmp_path), task_id='worker-session',
        owner_task_id='worker-session', session_key='worker-route')
    controls.stop_processes()
    assert late.exited


def test_native_managed_stop_frame_sets_deadline_and_decodes_process_cleanup(tmp_path, monkeypatch, registry):
    from agent.managed_worker import WorkerControls, encode_frame
    from gateway.session_managed_worker import ManagedWorker, interrupt_managed
    from gateway.session_contract import SessionRef
    command = shlex.join([sys.executable, '-c', 'import threading; threading.Event().wait(40)'])
    process = registry.spawn_local(command, cwd=str(tmp_path), task_id='s', owner_task_id='s', session_key='route',
                                   persist_on_release=True)
    worker = ManagedWorker(None)  # production command queue and deadline latch, without a pipe writer
    authority = SimpleNamespace(_managed_workers={'s': worker}, authorize=lambda *a: None,
                                check_approval_generation=lambda *a: None)
    assert interrupt_managed(authority, object(), SessionRef('profile', 's'), 1)
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
