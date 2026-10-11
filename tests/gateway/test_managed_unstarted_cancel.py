"""Cancellation before reservation closes any acquired child and settles the unexecuted claim."""
import asyncio
import os
import subprocess
import sys
import threading

import pytest

from gateway.session_contract import Submission
from hermes_state_runtime import get_session_admission
from tests.gateway.test_session_authority_cancel_settlement import _authority, ACTOR, REF


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['environment', 'spawn', 'hello'])
async def test_cancel_before_worker_reservation_settles_interrupted(tmp_path, monkeypatch, phase):
    from gateway import session_managed_worker as managed
    db, authority = _authority(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    children = []
    original_spawn = subprocess.Popen
    def environment(owner):
        if phase == 'environment':
            entered.set()
            assert release.wait(10)
        return dict(os.environ)
    def spawn(args, **kwargs):
        if phase == 'spawn':
            entered.set()
            assert release.wait(10)
        child = original_spawn([sys.executable, '-c', 'import sys; sys.stdin.buffer.read()'], **kwargs)
        children.append(child)
        return child
    original_next = managed.ManagedWorker.next_frame
    async def next_frame(worker, *args, **kwargs):
        if phase == 'hello':
            entered.set()
        return await original_next(worker, *args, **kwargs)
    monkeypatch.setattr(managed, '_worker_env', environment)
    monkeypatch.setattr(managed.subprocess, 'Popen', spawn)
    monkeypatch.setattr(managed.ManagedWorker, 'next_frame', next_frame)
    async def execute(owner, ref, row):
        return await managed.execute_managed(owner, ref, row, policy=None)
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    with db:
        receipt = await authority.submit(ACTOR, Submission('cancelled-startup', REF, {'text': 'work'}, 'queue'))
        task = authority.sessions['s'].task
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            release.set()
            await asyncio.wait_for(task, 10)
            row = get_session_admission(db, admission_id=receipt.admission_id)
            assert row['status'] == 'terminal' and row['outcome'] == 'interrupted'
            assert authority.sessions['s'].event_stream.execution == {}
            if phase != 'environment':
                async with asyncio.timeout(10):
                    while not children or any(child.poll() is None for child in children):
                        await asyncio.sleep(.01)
            assert db._read_one('SELECT COUNT(*) FROM worker_executions')[0] == 0
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
