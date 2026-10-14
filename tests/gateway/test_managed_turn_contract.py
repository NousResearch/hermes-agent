"""Managed turns carry finite approval policy, accounting, and live tool details."""
import asyncio
from dataclasses import replace
import json
import os
import threading
from types import SimpleNamespace

import pytest


def assignment(tmp_path, monkeypatch, *, finite=True, unattended=False, yolo=False):
    from gateway.session_local import _bypass_policy
    from gateway.session_managed_worker import _bootstrap
    from tools.approval import clear_session
    clear_session('managed-contract')
    policy = _bypass_policy({'cwd': str(tmp_path), 'model': 'fixture', 'provider': 'custom',
                            'base_url': 'http://127.0.0.1:9/v1', 'ignore_user_config': True}, private_secrets={})
    policy = replace(policy, request_json=json.dumps({**json.loads(policy.request_json), 'yolo': yolo}))
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda *args: None)
    live = SimpleNamespace(source=SimpleNamespace(user_id='u', chat_id='c'), route='managed-contract')
    authority = SimpleNamespace(sessions={'sid': live}, runner=None, profile_id=str(tmp_path))
    ref = SimpleNamespace(session_id='sid')
    row = {'payload': {'text': 'hello', 'finite': finite, 'unattended': unattended}}
    scope = {'profile_id': str(tmp_path), 'session_id': 'sid', 'execution_id': 'x', 'generation': 1,
             'pid': os.getpid(), 'birth': 0, 'secret': 's', 'epoch': 1}
    return _bootstrap(authority, ref, row, policy, scope), (authority, ref, row, policy, scope)


@pytest.mark.parametrize('unattended', [False, True])
def test_child_executes_under_finite_policy_and_preserves_receipt(tmp_path, monkeypatch, unattended):
    from agent import managed_worker as worker
    from gateway.session_finite import finite_turn_required, unattended_turn
    from tools.approval import _yolo_active, clear_session
    frame, _ = assignment(tmp_path, monkeypatch, unattended=unattended)
    observed, frames = [], []

    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_prompt_tokens = 100
            self.session_completion_tokens = 20
        def run_conversation(self, text, **kwargs):
            observed.append((finite_turn_required(), unattended_turn(), _yolo_active()))
            self.kwargs['tool_start_callback']('call-1', 'read_file', {'path': 'report.txt'})
            self.kwargs['tool_complete_callback']('call-1', 'read_file', {'path': 'report.txt'},
                                                 '{"error":"file missing"}')
            self.session_prompt_tokens += 7
            self.session_completion_tokens += 3
            return {'final_response': 'budget summary', 'completed': False, 'partial': True,
                    'turn_exit_reason': 'max_iterations_reached(1/1)', 'prompt_tokens': 107,
                    'completion_tokens': 23, 'estimated_cost_usd': .02, 'model': 'fixture', 'provider': 'custom',
                    'messages': [{'role': 'assistant', 'content': 'large history'}]}

    class Store:
        def __init__(self, *args): pass
        def get_messages_as_conversation(self, sid): return []
        def flush_token_counts(self): pass
        def finish(self): pass
        def close(self): pass

    class Controls:
        def __init__(self, *args):
            self.stopped = threading.Event()
            self.finish = threading.Event()
            self.finish.set()
        def approval(self, data): pytest.fail('unexpected approval')
        def clarify(self, questions): pytest.fail('unexpected question')

    monkeypatch.setattr(worker, 'bind_worker_policy', lambda frame: None)
    monkeypatch.setattr(worker, 'discover_profile_mcp', lambda policy: None)
    monkeypatch.setattr(worker, 'retire_agent', lambda agent: None)
    monkeypatch.setattr(worker, 'WorkerControls', Controls)
    monkeypatch.setattr('agent.runtime_session_store.WorkerRPC', lambda home: lambda *a, **k: {'owner_epoch': 1})
    monkeypatch.setattr('agent.runtime_session_store.RuntimeSessionStore', Store)
    monkeypatch.setattr('tools.process_registry.process_registry.recover_from_checkpoint', lambda: None)
    monkeypatch.setattr('run_agent.AIAgent', Agent)
    monkeypatch.setattr('agent.title_generator.wait_for_title_upgrades', lambda: None)
    clear_session('managed-contract')  # the child owns a fresh approval registry
    worker.execute(frame, SimpleNamespace(send=lambda kind, **payload: frames.append({'type': kind, **payload})))
    assert observed == [(True, unattended, unattended)]
    receipt = next(f for f in frames if f['type'] == 'result')
    assert receipt['result']['completed'] is False
    assert receipt['result']['estimated_cost_usd'] == .02
    assert receipt['result']['provider'] == 'custom'
    assert receipt['result']['turn_exit_reason'] == 'max_iterations_reached(1/1)'
    assert 'messages' not in receipt['result']
    _, usage = worker.accept_result(receipt['result'])
    assert usage == {'input_tokens': 7, 'output_tokens': 3, 'total_tokens': 10}
    events = [f for f in frames if f['type'].startswith('tool.')]
    assert events[0]['tool_call_id'] == 'call-1' and events[0]['args']['path'] == 'report.txt'
    assert events[1]['is_error'] is True
    assert json.loads(events[1]['result'])['error'] == 'file missing'


def test_current_yolo_revocation_overrides_frozen_launch(tmp_path, monkeypatch):
    from gateway.session_managed_worker import _bootstrap
    from tools.approval import disable_session_yolo, clear_session, is_session_yolo_enabled
    frame, args = assignment(tmp_path, monkeypatch, yolo=True)
    assert json.loads(frame['policy']['request_json'])['turn_v1']['yolo'] is True
    from gateway.session_managed_worker import worker_turn_scope
    disable_session_yolo('managed-contract')
    assert json.loads(_bootstrap(*args)['policy']['request_json'])['turn_v1']['yolo'] is False
    clear_session('managed-contract')
    with worker_turn_scope(frame):
        assert is_session_yolo_enabled('managed-contract') is True
    clear_session('managed-contract')



def test_committed_surface_crosses_the_bootstrap_validated_and_one_turn_only(tmp_path, monkeypatch):
    from agent.managed_worker import validate_bootstrap
    from gateway.session_kanban import run_worker_turns
    from tools.voice_live import VOICE_LIVE_TURN_NOTE
    frame, (authority, ref, row, policy, scope) = assignment(tmp_path, monkeypatch)
    surface = {'surface': 'voice-live', 'voice_context': 'User: UNIQUE_PRIOR_CONTEXT', 'voice_turn': True}
    from gateway.session_managed_worker import _bootstrap
    voiced = validate_bootstrap(json.loads(json.dumps(_bootstrap(
        authority, ref, {'payload': {**row['payload'], 'surface_v1': surface}}, policy, scope))))
    seen = []
    agent = SimpleNamespace(valid_tool_names=set(), run_conversation=lambda text, **kw: seen.append(
        (agent._voice_turn_pending, agent._gateway_turn_context_notes)) or {'final_response': 'ok'})
    run_worker_turns(agent, voiced, [])
    run_worker_turns(agent, frame, [])  # the same (reused) agent's next, plain turn
    (marker, notes), plain = seen
    assert marker is True and notes.startswith(VOICE_LIVE_TURN_NOTE) and 'UNIQUE_PRIOR_CONTEXT' in notes
    assert plain == (False, '')
    # The child re-checks the object with the admission validator: no smuggled model input.
    request = json.loads(voiced['policy']['request_json'])
    request['turn_v1']['surface_v1'] = {'voice_context': 'User: smuggled'}
    forged = {**voiced, 'policy': {**voiced['policy'], 'request_json': json.dumps(request)}}
    with pytest.raises(ValueError, match='invalid_managed_worker_bootstrap'):
        run_worker_turns(agent, forged, [])
    assert len(seen) == 2

@pytest.mark.asyncio
async def test_stop_during_environment_preparation_prevents_bootstrap(tmp_path, monkeypatch):
    from gateway import session_managed_worker as managed
    started, release = threading.Event(), threading.Event()
    frames = []
    def env(authority):
        started.set()
        assert release.wait(5)
    class Worker:
        def __init__(self, process):
            self.stop = asyncio.Event()
        def interrupt(self):
            self.stop.set()
        async def next_frame(self, *args, **kwargs):
            from hermes_state_runtime import RuntimeStoreError
            assert self.stop.is_set(), 'pending Stop was not adopted'
            raise RuntimeStoreError('managed_worker_stopped')
        def close(self): pass
    monkeypatch.setattr(managed, '_worker_env', env)
    monkeypatch.setattr(managed.subprocess, 'Popen', lambda *a, **k: object())
    monkeypatch.setattr(managed, 'ManagedWorker', Worker)
    authority = SimpleNamespace(profile_id='profile', pending_stops={}, pending_results={})
    def adopt(sid, generation, worker):
        if authority.pending_stops.pop(sid, None) == generation:
            worker.interrupt()
    authority.adopt_agent = adopt
    ref = SimpleNamespace(session_id='sid')
    row = {'admission_id': 'adm', 'generation': 3}
    task = asyncio.create_task(managed.execute_managed(authority, ref, row, None))
    assert await asyncio.to_thread(started.wait, 5)
    authority.pending_stops['sid'] = 3
    release.set()
    await asyncio.wait_for(task, 5)
    assert not frames
    assert authority.pending_results['adm']['result']['interrupted'] is True
    assert not authority.pending_stops


def test_shared_tool_events_bound_large_content_without_losing_error_or_path():
    from gateway.run_turn_progress import _tool_lifecycle_payload
    start = _tool_lifecycle_payload('call', 'write_file', {'path': 'report.txt', 'content': 'x' * 1_000_000})
    assert start['tool_id'] == 'call' and start['args']['path'] == 'report.txt'
    assert len(json.dumps(start)) < 100_000
    from gateway.run_turn_progress import _tool_complete_payload
    complete = _tool_complete_payload('call', 'write_file', {}, {'error': 'bad output ' + 'x' * 1_000_000})
    assert complete['is_error'] is True
    assert 'error' in json.loads(complete['result'])
    assert len(json.dumps(complete)) < 100_000
