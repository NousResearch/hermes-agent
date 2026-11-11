"""Single restart deadline, including a frozen pre-stop event loop."""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway import restart_handoff


def test_duplicate_restart_cannot_extend_deadline(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(restart_handoff.time, 'monotonic', lambda: now[0])
    runner = SimpleNamespace(_restart_after_turn_timeout=20, _restart_drain_timeout=0)
    first = restart_handoff.begin(runner, stop_budget=60, arm=False)
    assert first.deadline == 180
    now[0] = 110
    assert restart_handoff.begin(runner, stop_budget=600, arm=False) is first
    assert first.remaining() == 70
    assert first.phase_budget(100, reserve=10) == 60
    now[0] = 181
    assert first.remaining() == first.phase_budget(30) == 0


def test_thread_watchdog_covers_frozen_pre_stop_loop(tmp_path):
    code = '''
import asyncio
import time
from types import SimpleNamespace
from gateway.restart_handoff import begin
async def main():
    runner = SimpleNamespace(_restart_after_turn_timeout=0, _restart_drain_timeout=0)
    begin(runner, stop_budget=0.2)
    print('ARMED', flush=True)
    time.sleep(30)
asyncio.run(main())
'''
    env = dict(os.environ)
    env.pop('PYTEST_CURRENT_TEST', None)
    env['HERMES_HOME'] = str(tmp_path / 'home')
    env['HERMES_RUNTIME_DIR'] = str(tmp_path / 'runtime')
    env['HERMES_DISABLE_LAZY_INSTALLS'] = '1'
    with subprocess.Popen([sys.executable, '-c', code], env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as child:
        try:
            stdout, stderr = child.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            pytest.fail('pre-stop watchdog did not terminate frozen fixture')
    assert 'ARMED' in stdout
    assert child.returncode == 75, stderr


def test_maintenance_drain_does_not_create_deadline():
    runner = SimpleNamespace(_external_drain_active=True)
    assert restart_handoff.snapshot(runner) == {'admission_ready': False, 'drain_reason': 'external_drain'}
    assert not hasattr(runner, '_restart_handoff')
