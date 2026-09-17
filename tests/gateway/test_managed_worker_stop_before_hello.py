"""Stop is never acknowledged into a queue nobody drains: a managed child that stays alive
without ever sending hello is terminated on Stop and the admission settles as interrupted."""
import asyncio
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from gateway.session_contract import Principal, SessionRef


@pytest.mark.linux_only
def test_stop_before_hello_terminates_child_and_settles_interrupted(monkeypatch):
    from gateway import session_managed_worker as managed
    spawned = []
    original = subprocess.Popen

    def wedged_child(args, **kwargs):
        # An interpreter that starts but never introduces itself (loader/stdio stall).
        child = original([sys.executable, '-c', 'import time\nwhile True: time.sleep(1)'], **kwargs)
        spawned.append(child)
        return child
    monkeypatch.setattr(managed.subprocess, 'Popen', wedged_child)
    ref = SessionRef('profile', 'session')
    checks = []
    authority = SimpleNamespace(profile_id='profile', pending_results={}, waiters={},
        authorize=lambda actor, ref, cap: checks.append(cap),
        check_approval_generation=lambda session_id, generation: checks.append(generation))
    row = {'admission_id': 'adm', 'principal_id': 'owner', 'generation': 3, 'payload': {'text': 'go'}}
    actor = Principal('owner', 'profile', frozenset({'session:control'}), 'transport')

    async def scenario():
        turn = asyncio.create_task(managed.execute_managed(authority, ref, row, policy=None))
        async with asyncio.timeout(10):
            while ref.session_id not in getattr(authority, '_managed_workers', {}):
                await asyncio.sleep(.01)
        assert managed.interrupt_managed(authority, actor, ref, 3) is True
        async with asyncio.timeout(10):
            return await turn
    assert asyncio.run(scenario()) == ''
    assert checks == ['session:control', 3]
    assert authority.pending_results['adm']['result']['interrupted'] is True
    assert ref.session_id not in authority._managed_workers
    child = spawned[0]
    assert child.poll() is not None and not psutil.pid_exists(child.pid) or psutil.Process(child.pid).status() == psutil.STATUS_ZOMBIE
