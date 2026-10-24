"""F2: execute's real finally settles an outstanding unlimited clarification.

Only the unavailable agent/persistence/MCP boundaries are synthetic. Controls,
clarify admission/wait and execute/finally run unchanged in a native interpreter.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]

CHILD = textwrap.dedent(r'''
    import json, os, socket, sys, threading
    from pathlib import Path
    from types import SimpleNamespace, ModuleType
    import pytest

    # An accidental provider/credential transport must fail before dispatch.
    def no_network(*args, **kwargs):
        raise AssertionError('fixture attempted network access')
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    socket.getaddrinfo = no_network

    from agent import managed_worker as worker
    from tests.waiting import wait_until

    mode = sys.argv[1]
    patch = pytest.MonkeyPatch()
    frame = json.loads(Path(sys.argv[2]).read_text())
    frame['scope']['pid'] = os.getpid()
    worker.validate_bootstrap(frame)
    events, answers, errors, controls, waiters, states = [], [], [], [], [], []
    registered = threading.Event()
    failure = RuntimeError('synthetic_' + mode + '_failure')

    def send(kind, **payload):
        events.append(kind)
        if kind == 'clarify':
            control = controls[0]
            with control.lock:
                state = control.clarifications[payload['prompt_id']]
                assert not state['event'].is_set()
                states.append(state)
            registered.set()

    class RPC:
        def __init__(self, home):
            # execute has now bound the real policy before any runtime imports.
            patch.setitem(sys.modules, 'run_agent', agent_module)
            from tools.process_registry import process_registry
            patch.setattr(process_registry, 'recover_from_checkpoint', lambda: None)
            patch.setattr('agent.title_generator.wait_for_title_upgrades', lambda: None)
        def __call__(self, *args, **kwargs): return {'owner_epoch': 1}
        def close(self): events.append('rpc.close')

    class Store:
        def __init__(self, *args, **kwargs): pass
        def get_messages_as_conversation(self, sid): return []
        def flush_token_counts(self):
            events.append('store.flush')
            assert registered.is_set() and waiters[0].is_alive()
            raise failure
        def finish(self): events.append('store.finish')
        def close(self): events.append('store.close')

    class Agent:
        def __init__(self, **kwargs):
            self.callback = kwargs['clarify_callback']
            controls.append(self.callback.__self__)
        def run_conversation(self, *args, **kwargs):
            from tools.clarify_gateway import get_clarify_timeout
            assert get_clarify_timeout() == 0, 'fixture is not unlimited'
            def ask():
                try:
                    answers.append(self.callback([{'qid': 'q1', 'question': 'pending?',
                                                   'choices': [], 'multi_select': False}]))
                except BaseException as exc:
                    errors.append(repr(exc))
            waiter = threading.Thread(target=ask, name='f2-unlimited-clarification', daemon=False)
            waiters.append(waiter)
            waiter.start()
            assert registered.wait(2), 'fixture never registered an actual clarification'
            # Observe the actual unbounded Event.wait, not just a sent prompt.
            def blocked():
                current = sys._current_frames().get(waiter.ident)
                while current is not None:
                    if (current.f_code.co_name == 'wait'
                            and current.f_locals.get('self') is states[0]['event']):
                        return current.f_locals.get('timeout') is None
                    current = current.f_back
                return False
            wait_until(blocked, timeout=2)
            events.append('unlimited_wait_registered')
            if mode == 'agent':
                raise failure
            return {'final_response': 'ok', 'completed': True}
        def shutdown_memory_provider(self, messages): events.append('agent.retired')
        def release_clients(self): pass

    # Keep actual policy binding/turn scope, execute, retire_agent and controls intact.
    agent_module = ModuleType('run_agent')
    agent_module.AIAgent = Agent
    # Installing run_agent or importing config before execute would violate the
    # native one-shot policy binder; runtime_session_store itself is import-safe.
    patch.setattr(worker, 'discover_profile_mcp', lambda policy: None)
    patch.setattr('agent.runtime_session_store.WorkerRPC', RPC)
    patch.setattr('agent.runtime_session_store.RuntimeSessionStore', Store)
    caught = False
    try:
        try:
            worker.execute(frame, SimpleNamespace(send=send))
        except RuntimeError as exc:
            assert exc is failure, 'unexpected execute failure: ' + repr(exc)
            caught = True
        assert caught, 'synthetic failure did not cross actual execute/finally'
        waiters[0].join(2)
        control = controls[0]
        with control.lock:
            pending = len(control.clarifications)
        receipt = {'mode': mode, 'events': events, 'caught_expected_failure': caught,
                   'waiter_alive': waiters[0].is_alive(), 'reader_alive': control.reader.is_alive(),
                   'pending': pending, 'answers': answers, 'errors': errors,
                   'stdin_open': not sys.stdin.closed, 'source': worker.__file__}
        print('F2_RECEIPT ' + json.dumps(receipt), flush=True)
        assert not receipt['waiter_alive'], 'F2: execute/finally left unlimited clarification waiter alive'
        assert not receipt['reader_alive'], 'F2: execute/finally left controls reader alive'
        assert pending == 0 and not errors
        assert answers == [{'answers': {}, 'outcome': 'cancelled'}]
        assert events.count('prompt_settled') == 1
        assert 'result' not in events and 'finished' not in events
        assert events.index('unlimited_wait_registered') < events.index('store.close')
        assert events[-2:] == ['store.close', 'rpc.close'] or events[-1] == 'prompt_settled'
    finally:
        # Recover only after the product-state snapshot/assertion. A broken predecessor
        # must produce a finite assertion RED, never an interpreter/thread hang.
        for state in states:
            state['answer'] = worker._INTERRUPTED
            state['event'].set()
        for waiter in waiters:
            waiter.join(2)
            assert not waiter.is_alive(), 'fixture could not settle its non-daemon thread'
        patch.undo()
''')


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('mode', ['persistence', 'agent'])
def test_execute_finally_settles_unlimited_clarification_with_stdin_open(tmp_path, mode, monkeypatch):
    # The override is for unchanged-test predecessor replay, never production config.
    source = Path(os.environ.get('HERMES_F2_SOURCE_ROOT', str(ROOT))).resolve()
    home, hermes, scratch = (tmp_path / name for name in ('home', 'hermes', 'scratch'))
    for path in (home, hermes, scratch):
        path.mkdir()
    env = {key: value for key, value in os.environ.items()
           if key in ('PATH', 'LANG', 'TZ', 'SYSTEMROOT')}
    env.update(HOME=str(home), HERMES_HOME=str(hermes), TMPDIR=str(scratch),
               TMP=str(scratch), TEMP=str(scratch), PYTHONPATH=str(source),
               HERMES_DISABLE_LAZY_INSTALLS='1', HERMES_TEST_ISOLATION=str(hermes),
               PYTHONDONTWRITEBYTECODE='1')
    # Build the real assignment in the parent, whose runtime imports are already
    # present. The fresh child validates its own pid and binds policy in execute.
    from tests.gateway.test_managed_turn_contract import assignment
    frame, _ = assignment(hermes, monkeypatch, finite=False)
    config = json.loads(frame['policy']['config_json'])
    config['agent']['clarify_timeout'] = 0
    frame['policy']['config_json'] = json.dumps(config)
    # A noncredential sentinel bypasses provider-key resolution at the synthetic agent boundary.
    frame['api_key'] = 'fixture-no-provider'
    frame_path = hermes / ('frame-' + mode + '.json')
    frame_path.write_text(json.dumps(frame))
    child = subprocess.Popen([sys.executable, '-B', '-c', CHILD, mode, str(frame_path)], cwd=source, env=env,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # No communicate()/EOF/finish/stop: ownership stays open through native exit.
        rc = child.wait(timeout=10)
        stdout, stderr = child.stdout.read().decode(), child.stderr.read().decode()
        print('F2_NATIVE_EXIT', rc, '\nSTDOUT\n' + stdout + '\nSTDERR\n' + stderr)
        lines = [line[len('F2_RECEIPT '):] for line in stdout.splitlines()
                 if line.startswith('F2_RECEIPT ')]
        assert len(lines) == 1, f'invalid fixture (no product snapshot): exit={rc}: {stderr}'
        receipt = json.loads(lines[0])
        assert Path(receipt['source']).resolve() == source / 'agent' / 'managed_worker.py'
        assert receipt['caught_expected_failure'] and receipt['stdin_open']
        assert not receipt['waiter_alive'], 'F2: execute/finally left unlimited clarification waiter alive'
        assert rc == 0, f'native interpreter exit={rc}: {stderr}'
        assert '_enter_buffered_busy' not in stderr and 'Fatal Python error' not in stderr
        assert receipt['reader_alive'] is False and receipt['pending'] == 0
        assert child.stdin.closed is False
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()
