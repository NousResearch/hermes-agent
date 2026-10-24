"""Exceptional execute teardown must join its real controls reader before CPython finalizes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]

CHILD = textwrap.dedent('''
    import json, os, sys, threading
    from pathlib import Path
    from types import SimpleNamespace
    import pytest
    from agent import managed_worker as worker
    from tests.gateway.test_managed_turn_contract import assignment
    from tests.waiting import wait_until
    mode = sys.argv[1]
    patch = pytest.MonkeyPatch()
    frame, _ = assignment(Path(os.environ['HERMES_HOME']), patch)
    events, controls = [], []
    partial_read = threading.Event()
    original_read = os.read
    def read(fd, size):
        data = original_read(fd, size)
        if fd == sys.stdin.fileno() and data == b'{"type":':
            partial_read.set()
        return data
    patch.setattr(os, 'read', read)
    original_controls = worker.WorkerControls
    def create_controls(*a, **k):
        obj = original_controls(*a, **k)
        controls.append(obj)
        return obj
    class RPC:
        def __init__(self, home): pass
        def __call__(self, *a, **k): return {'owner_epoch': 1}
        def close(self): events.append('rpc.close')
    class Store:
        def __init__(self, *a, **k): pass
        def get_messages_as_conversation(self, sid): return []
        def flush_token_counts(self):
            # The agent has completed but no result may be sent before this flush.
            def reading():
                if controls[0].finish.is_set():
                    return True
                f = sys._current_frames().get(controls[0].reader.ident)
                while f is not None:
                    if f.f_code.co_name in ('read_frame', 'read'):
                        return True
                    f = f.f_back
                return False
            wait_until(reading, timeout=5)
            if mode == 'partial':
                assert partial_read.wait(5), 'controls reader did not consume the partial frame'
            print('READER_READY', flush=True)
            if mode in ('failure', 'partial', 'main_failure'):
                raise RuntimeError('synthetic_pre_result_persistence_failure')
        def finish(self): events.append('store.finish')
        def close(self): events.append('store.close')
    class Agent:
        def __init__(self, **kwargs): pass
        def run_conversation(self, *a, **k): return {'final_response': 'ok', 'completed': True}
    patch.setattr(worker, 'WorkerControls', create_controls)
    patch.setattr(worker, 'bind_worker_policy', lambda frame: None)
    patch.setattr(worker, 'discover_profile_mcp', lambda policy: None)
    patch.setattr(worker, 'retire_agent', lambda agent: None)
    patch.setattr('agent.runtime_session_store.WorkerRPC', RPC)
    patch.setattr('agent.runtime_session_store.RuntimeSessionStore', Store)
    patch.setattr('tools.process_registry.process_registry.recover_from_checkpoint', lambda: None)
    patch.setattr('run_agent.AIAgent', Agent)
    patch.setattr('agent.title_generator.wait_for_title_upgrades', lambda: None)
    patch.setattr(worker, '_die_with_owner', lambda: events.append('owner_eof'))
    if mode.startswith('main'):
        # Keep main's real pipe/framing/execute/error path; only bootstrap identity is synthetic.
        patch.setattr(worker, 'validate_bootstrap', lambda incoming: frame)
        status = worker.main()
        print('MAIN_SETTLED', json.dumps({'events': events,
              'reader_alive': controls[0].reader.is_alive()}), flush=True)
        raise SystemExit(status)
    try:
        worker.execute(frame, SimpleNamespace(send=lambda kind, **payload: events.append(kind)))
    except RuntimeError as exc:
        assert str(exc) == 'synthetic_pre_result_persistence_failure'
        events.append('expected_failure')
    finally:
        print(json.dumps({'events': events, 'reader_alive': controls[0].reader.is_alive(),
                          'stopped': controls[0].stopped.is_set()}), flush=True)
    # Deliberately no test-only close/join: actual execute owns cleanup.
''')


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('mode', ['failure', 'partial', 'finish', 'eof'])
def test_exceptional_execute_exits_while_owner_holds_stdin_open(tmp_path, mode):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path), 'HERMES_HOME': str(tmp_path)}
    child = subprocess.Popen([sys.executable, '-B', '-c', CHILD, mode], cwd=ROOT, env=env,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if mode == 'partial':
            child.stdin.write(b'{"type":'); child.stdin.flush()
        elif mode == 'finish':
            child.stdin.write(b'{"type":"finish"}\n'); child.stdin.flush()
        elif mode == 'eof':
            child.stdin.close()
        # wait(), not communicate(): the parent keeps stdin open during interpreter finalization.
        rc = child.wait(timeout=10)
        stdout, stderr = child.stdout.read().decode(), child.stderr.read().decode()
        print('CHILD', rc, stdout, stderr)
        assert rc == 0, f'actual interpreter exit was {rc}: {stderr}'
        assert '_enter_buffered_busy' not in stderr
        receipt = json.loads(stdout.strip().splitlines()[-1])
        assert receipt['reader_alive'] is False
        if mode in ('failure', 'partial'):
            assert 'expected_failure' in receipt['events'] and 'result' not in receipt['events']
            assert receipt['events'][-3:] == ['store.close', 'rpc.close', 'expected_failure']
        else:
            assert 'result' in receipt['events'] and 'finished' in receipt['events']
            assert ('owner_eof' in receipt['events']) == (mode == 'eof')
            assert receipt['stopped'] == (mode == 'eof')
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close(); child.stdout.close(); child.stderr.close()


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('mode', ['main_failure', 'main_finish'])
def test_main_pipe_bootstrap_and_controls_share_no_buffered_bytes(tmp_path, mode):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path), 'HERMES_HOME': str(tmp_path)}
    child = subprocess.Popen([sys.executable, '-B', '-c', CHILD, mode], cwd=ROOT, env=env,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wire = b'{}\n' + (b'{"type":"finish"}\n' if mode == 'main_finish' else b'')
        child.stdin.write(wire); child.stdin.flush()
        rc = child.wait(timeout=10)
        stdout, stderr = child.stdout.read().decode(), child.stderr.read().decode()
        print('MAIN_CHILD', rc, stdout, stderr)
        assert rc == (1 if mode == 'main_failure' else 0)
        assert '_enter_buffered_busy' not in stderr
        frames = [json.loads(line) for line in stdout.splitlines()]
        assert frames[-1]['type'] == ('error' if mode == 'main_failure' else 'finished')
        receipt = json.loads(next(line.split('MAIN_SETTLED ', 1)[1] for line in stderr.splitlines()
                                  if line.startswith('MAIN_SETTLED ')))
        assert receipt['reader_alive'] is False
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close(); child.stdout.close(); child.stderr.close()
