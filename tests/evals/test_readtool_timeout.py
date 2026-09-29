"""Read-tool benchmark deadlines must produce failed records (#82978)."""

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import psutil
import pytest

from evals.readtool import runner


# Executes the real worker/metrics code, with no provider or model calls.
BOOTSTRAP = '''
import os, sys, time, subprocess, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from evals.readtool import runner
mode, receipt = sys.argv[2:4]
class Agent:
    session_total_tokens = 17
    def __init__(self, **kw):
        assert kw['model'] == 'fixture'
        assert kw['provider'] == 'fixture'
        assert kw['enabled_toolsets'] == ['file']
    def run_conversation(self, prompt):
        if mode == 'timeout':
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
            Path(receipt).write_text(str(child.pid))
            time.sleep(60)
        if mode == 'auth':
            raise RuntimeError('authentication failed')
        if mode == 'error':
            raise RuntimeError('fixture failure')
        if mode == 'empty':
            return {}
        return {'final_response': 'empty', 'messages': [{'role': 'assistant'}]}
fake = types.ModuleType('run_agent')
fake.AIAgent = Agent
sys.modules['run_agent'] = fake
runner.build_workspace = lambda path: path.mkdir()
raise SystemExit(runner._worker_main(sys.argv[4:]))
'''


@pytest.mark.parametrize('mode', ['timeout', 'success', 'empty', 'error', 'auth'])
def test_task_worker_obeys_deadline_and_preserves_result_contract(tmp_path, monkeypatch, mode):
    real_popen = subprocess.Popen
    processes = []
    homes = []
    receipt = tmp_path / 'descendant.pid'
    before = dict(os.environ)

    def launch(command, **kwargs):
        if '--worker' not in command:
            return real_popen(command, **kwargs)
        homes.append(Path(kwargs['env']['HERMES_HOME']).parent)
        proc = real_popen([sys.executable, '-c', BOOTSTRAP, str(runner.REPO_ROOT),
                           mode, str(receipt), *command[3:]], **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    task = replace(runner.TASKS_BY_ID['empty_config'], timeout_s=6)
    try:
        if mode == 'auth':
            with pytest.raises(SystemExit, match='harness config error'):
                runner.run_task(task, 'fixture', 'fixture', 0.5, ['file'])
        else:
            result = runner.run_task(task, 'fixture', 'fixture', 0.5, ['file'])
            if mode == 'timeout':
                assert result['score'] == 0
                assert result['error'] == 'TimeoutError: task exceeded 3s'
                assert receipt.exists(), 'worker must reach its blocking call'
                descendant = int(receipt.read_text())
                deadline = time.monotonic() + 5
                while psutil.pid_exists(descendant) and time.monotonic() < deadline:
                    try:
                        if psutil.Process(descendant).status() == psutil.STATUS_ZOMBIE:
                            break
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(0.05)
                assert not psutil.pid_exists(descendant) or psutil.Process(descendant).status() == psutil.STATUS_ZOMBIE
                # A timed-out task must not poison the next repetition.
                mode = 'success'
                following = runner.run_task(task, 'fixture', 'fixture', 0.5, ['file'])
                assert following['error'] is None and following['score'] == 1
            elif mode in ('success', 'empty'):
                assert result['error'] is None
                assert result['score'] == (1 if mode == 'success' else 0)
                assert result['total_tokens'] == 17
                assert result['api_turns'] == (1 if mode == 'success' else 0)
            else:
                assert result['score'] == 0
                assert result['error'] == 'RuntimeError: fixture failure'
        assert processes and all(p.poll() is not None for p in processes)
        assert all(not home.exists() for home in homes)
        assert dict(os.environ) == before
    finally:
        # Restore Popen before invoking the shared Windows taskkill primitive.
        monkeypatch.setattr(runner.subprocess, 'Popen', real_popen)
        from agent.deadline import kill_process_tree
        for proc in processes:
            if proc.poll() is None:
                kill_process_tree(proc.pid)
                proc.wait(timeout=20)
        if receipt.exists() and psutil.pid_exists(int(receipt.read_text())):
            try:
                psutil.Process(int(receipt.read_text())).kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.parametrize('multiplier', [0, -1, float('nan'), float('inf')])
def test_invalid_deadline_does_not_launch_worker(monkeypatch, multiplier):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid deadline launched a worker')
    monkeypatch.setattr(runner.subprocess, 'Popen', unexpected)
    with pytest.raises(ValueError, match='finite and positive'):
        runner.run_task(runner.TASKS[0], 'fixture', 'fixture', multiplier, ['file'])
