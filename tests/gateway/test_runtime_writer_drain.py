"""Real startup threads retain runtime exclusion after a drain deadline."""
import asyncio
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest


@pytest.mark.linux_only
@pytest.mark.parametrize('partial_start', [False, True])
def test_live_runtime_writers_prevent_final_lock_release(tmp_path, monkeypatch, partial_start):
    import gateway.run as run
    from gateway.run_bootstrap import _start_gateway_start_cron_and_housekeeping
    from gateway.status import acquire_gateway_runtime_lock, release_gateway_runtime_lock
    from cron import scheduler_provider

    home = tmp_path / 'runtime'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    gate = threading.Event()
    started = threading.Event()
    threads = []

    class Provider:
        name = 'fixture'
        def start(self, stop, **kwargs):
            started.set()
            gate.wait(20)

    provider = Provider()
    monkeypatch.setattr(scheduler_provider, 'resolve_cron_scheduler', lambda: provider)
    monkeypatch.setattr(scheduler_provider, 'scheduler_for_profile_mode', lambda p, **kw: p)
    monkeypatch.setattr(run, '_start_gateway_housekeeping', lambda *a, **kw: gate.wait(20))
    original_start = threading.Thread.start

    def start_thread(thread):
        if thread.name == 'gateway-housekeeping' and partial_start:
            raise RuntimeError('owned thread-start failure')
        threads.append(thread)
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, 'start', start_thread)
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False), adapters={})

    async def start():
        return _start_gateway_start_cron_and_housekeeping(runner)

    probe = '''
from gateway.status import acquire_gateway_runtime_lock, release_gateway_runtime_lock
claimed = acquire_gateway_runtime_lock()
print('CLAIMED', claimed)
if claimed:
    release_gateway_runtime_lock()
'''

    def contender():
        result = subprocess.run([sys.executable, '-c', probe], env=os.environ.copy(),
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    assert acquire_gateway_runtime_lock()
    try:
        if partial_start:
            with pytest.raises(RuntimeError, match='owned thread-start failure'):
                asyncio.run(start())
        else:
            asyncio.run(start())
        assert started.wait(5)
        assert contender() == 'CLAIMED False'
        release_gateway_runtime_lock()
        assert contender() == 'CLAIMED False', 'final cleanup unlocked while a runtime writer survived'
    finally:
        gate.set()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
        release_gateway_runtime_lock()
    assert contender() == 'CLAIMED True'
