"""Cancelling threaded Popen acquisition still reaps the child it eventually creates."""
import asyncio
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_cancelled_spawn_reaps_late_child(monkeypatch):
    from gateway import session_managed_worker as managed
    entered, release, created = (threading.Event() for _ in range(3))
    original = subprocess.Popen
    children = []
    def spawn(args, **kwargs):
        entered.set()
        assert release.wait(10)
        child = original([sys.executable, '-c', 'import sys; sys.stdin.read()'], **kwargs)
        children.append(child)
        created.set()
        return child
    monkeypatch.setattr(managed.subprocess, 'Popen', spawn)
    monkeypatch.setattr(managed, '_worker_env', lambda authority: None)
    authority = SimpleNamespace(profile_id='profile', pending_results={})
    task = asyncio.create_task(managed.execute_managed(
        authority, SimpleNamespace(session_id='s'), {'admission_id': 'unstarted'}, None))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        assert await task == ''
        assert authority.pending_results['unstarted']['result']['interrupted'] is True
        release.set()
        assert await asyncio.to_thread(created.wait, 5)
        await asyncio.to_thread(children[0].wait, 5)
    finally:
        release.set()
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdin.close()
            child.stdout.close()
