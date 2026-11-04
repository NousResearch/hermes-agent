"""The real managed-control pipe reader exits on every bounded shutdown path."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]

CHILD = r'''
import json, resource, sys, threading, time
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import agent.managed_worker as mw
mode = sys.argv[1]
if mode == 'eof':
    mw._die_with_owner = lambda: None
controls = mw.WorkerControls(channel=None, route='test')
if mode == 'error':
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        frame = sys._current_frames().get(controls.reader.ident)
        parked = False
        while frame is not None:
            if frame.f_code.co_name == 'read_frame':
                parked = True
                break
            frame = frame.f_back
        if parked:
            break
        threading.Event().wait(0.01)
    if not parked:
        raise RuntimeError('control_reader_not_parked')
print('ready', flush=True)
if mode == 'error':
    try:
        raise RuntimeError('construction_failed')
    except Exception as original:
        controls.close()
        print(f'original={type(original).__name__}:{original}', flush=True)
else:
    if mode == 'eof':
        controls.finish.wait(5)
    else:
        controls.finish.wait(5)
    controls.reader.join(timeout=1)
    print(json.dumps({'stopped': controls.stopped.is_set(), 'finished': controls.finish.is_set(),
                      'reader_alive': controls.reader.is_alive()}), flush=True)
    controls.close()
'''


EXECUTE_CHILD = r'''
import contextlib, os, resource, sys, types
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import agent.managed_worker as mw
import agent.runtime_session_store as rss

class RPC:
    def __init__(self, home): pass
    def __call__(self, method, **params): return {'owner_epoch': 1}
    def close(self): pass
class Store:
    def __init__(self, *args): pass
    def close(self): pass

class Policy:
    skills_prompt = ''
    model, provider, base_url = 'test', None, None
    toolsets, reasoning_config = (), None
    max_turns, source, platform = 1, 'cli', 'cli'
    ignore_rules, pass_session_id = False, False
    def config(self): return {}

policy = Policy()
mw.bind_worker_policy = lambda frame: None
mw.outbox_dir = lambda *args: 'unused'
mw.discover_profile_mcp = lambda policy: None
mw._construct_agent = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('construction_failed'))
rss.WorkerRPC, rss.RuntimeSessionStore = RPC, Store

import gateway.session_kanban as sk
sk.bind_worker_context = lambda frame: None
import gateway.session_policy as sp
sp.restore_policy = lambda frame: policy
sp.register_worker_hooks = lambda policy: None
sp.policy_scope = lambda policy: contextlib.nullcontext()
import tools.process_registry as pr
pr.process_registry.recover_from_checkpoint = lambda: None
import tools.approval as approval
approval.register_gateway_notify = lambda *args: None
approval.unregister_gateway_notify = lambda *args: None
import tools.approval_context as approval_context
approval_context.set_current_session_key = lambda *args: None
import gateway.session_worker_construct as construct
construct.checkpoint_agent_kwargs = lambda *args: {}
construct.construct_kwargs = lambda *args: {}
construct.routing_agent_kwargs = lambda *args: {}
run_agent = types.ModuleType('run_agent')
run_agent.AIAgent = object
sys.modules['run_agent'] = run_agent

scope = {'profile_id': '/tmp/managed-worker-test', 'session_id': 'session', 'execution_id': 'execution',
         'generation': 0, 'pid': os.getpid(), 'birth': 'birth', 'secret': 'secret', 'epoch': 1}
frame = {'home': '/tmp/managed-worker-test', 'scope': scope, 'policy': {}, 'route': 'route',
         'api_key': None, 'user_id': 'user', 'chat_id': 'chat', 'text': 'text'}
reader = mw._ControlPipeReader(sys.stdin.fileno())
try:
    mw.execute(frame, types.SimpleNamespace(send=lambda *args, **kwargs: None), reader)
except Exception as exc:
    print(f'original={type(exc).__name__}:{exc}', flush=True)
'''


def _spawn(mode):
    return subprocess.Popen([sys.executable, '-c', CHILD, mode], cwd=ROOT,
                            env={**os.environ, 'PYTHONPATH': str(ROOT)}, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _finish(proc, message=None):
    if message is not None:
        proc.stdin.write(json.dumps(message) + '\n')
        proc.stdin.flush()
    proc.wait(timeout=5)  # stdin remains open unless the case explicitly closes it
    output = proc.stdout.read()
    error = proc.stderr.read()
    return output, error


@pytest.mark.platforms('linux')
def test_reader_shutdown_on_exception_with_open_stdin_preserves_original_error():
    proc = _spawn('error')
    try:
        assert proc.stdout.readline().strip() == 'ready'
        proc.wait(timeout=5)  # parent deliberately keeps the control pipe open
        output = proc.stdout.read()
        error = proc.stderr.read()
        assert proc.returncode == 0, error
        assert 'original=RuntimeError:construction_failed' in output
        assert 'Fatal Python error' not in error
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.platforms('linux')
def test_execute_construction_failure_stops_real_reader_with_stdin_open():
    proc = subprocess.Popen([sys.executable, '-c', EXECUTE_CHILD], cwd=ROOT,
                            env={**os.environ, 'PYTHONPATH': str(ROOT)}, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        proc.wait(timeout=8)  # actual execute() cleanup must stop the reader without parent EOF
        assert proc.stdout is not None and proc.stderr is not None
        output = proc.stdout.read()
        error = proc.stderr.read()
        assert proc.returncode == 0, error
        assert 'original=RuntimeError:construction_failed' in output
        assert 'Fatal Python error' not in error
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('mode', ['finish', 'stop', 'eof'])
def test_control_reader_handles_clean_finish_stop_and_owner_eof(mode):
    proc = _spawn(mode)
    try:
        assert proc.stdout.readline().strip() == 'ready'
        if mode == 'eof':
            proc.stdin.close()
            proc.wait(timeout=5)
            output = proc.stdout.read()
            error = proc.stderr.read()
        elif mode == 'stop':
            proc.stdin.write('{"type":"stop"}\n{"type":"finish"}\n')
            proc.stdin.flush()
            proc.wait(timeout=5)
            output = proc.stdout.read()
            error = proc.stderr.read()
        else:
            output, error = _finish(proc, {'type': 'finish'})
        assert proc.returncode == 0, error
        assert 'reader_alive' in output and 'false' in output
        if mode == 'stop':
            assert '"stopped": true' in output
        if mode == 'eof':
            assert '"stopped": true' in output
        assert '"finished": true' in output
        assert 'Fatal Python error' not in error
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
