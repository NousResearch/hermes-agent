"""Native parent ownership survives a worker's ephemeral per-turn process owner UUID."""
import asyncio
import shlex
import sys
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority


@pytest.mark.asyncio
@pytest.mark.platforms('linux', 'macos')
@pytest.mark.parametrize('managed', [False, True], ids=['owner-recovery', 'managed-worker'])
async def test_stop_uses_runtime_parent_and_route_without_crossing_foreign_parent(tmp_path, monkeypatch, managed):
    from tools import process_registry as module
    from gateway.session_context import clear_session_vars, set_session_vars
    from agent.managed_worker import WorkerControls
    db, authority = _authority(tmp_path, monkeypatch)
    original = module.ProcessRegistry()
    command = shlex.join([sys.executable, '-c', 'import threading; threading.Event().wait(60)'])
    objects = []
    # parent_session_id is captured by the native registry from the runtime binding,
    # not supplied by a terminal tool argument. A raw-owner collision cannot override it.
    for raw, parent, route in [('turn-uuid', 's', 'route'), ('s', '', 'route'),
                               ('s', 'foreign-parent', 'route'), ('turn-uuid', 's', 'foreign-route'),
                               ('other-turn', 'foreign-parent', 'route')]:
        tokens = set_session_vars(session_id=parent, session_key=route)
        try:
            objects.append(original.spawn_local(command, cwd=str(tmp_path), task_id='display-name',
                owner_task_id=raw, session_key=route, persist_on_release=True))
        finally:
            clear_session_vars(tokens)
    try:
        if managed:
            monkeypatch.setattr(module, 'process_registry', original)
            controls = WorkerControls.__new__(WorkerControls)
            controls.route, controls.session_id = 'route', 's'
            controls.stopped, controls.lock = threading.Event(), threading.Lock()
            controls.stop_processes_requested = False
            controls.task_checks, controls.clarifications = {}, {}
            controls.agent = SimpleNamespace(session_id='s', hard_interrupt=lambda *a, **kw: None)
            controls.stop(processes=True)
        else:
            current = module.ProcessRegistry()
            monkeypatch.setattr(module, 'process_registry', current)
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        for process in objects[:2]:
            assert await asyncio.to_thread(process._completion_event.wait, 10)
        assert all(not process.exited for process in objects[2:])
        from tools.process_registry import _checkpoint_path
        import json
        retained = json.loads(_checkpoint_path().read_text())
        assert {p.id for p in objects[2:]} <= {entry['session_id'] for entry in retained}
    finally:
        for process in objects:
            original.kill_process(process.id)
        db.close()
