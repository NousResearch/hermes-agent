"""Owner-expanded source facts reach the consuming worker over its private bootstrap."""
import json
import threading
from types import SimpleNamespace

import pytest

from agent.source_provenance import SourceProvenanceError


@pytest.fixture(autouse=True)
def inert_runtime(monkeypatch):
    """Fail before any test accidentally enters a host runtime or outbound transport."""
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append('forbidden runtime access')
        pytest.fail('real process, local environment, or network access is forbidden')
    import subprocess
    class NoProcess(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            forbidden(*args, **kwargs)
    monkeypatch.setattr('tools.environments.local.LocalEnvironment.__init__', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', NoProcess)
    for name in ('run', 'check_output', 'check_call', 'call'):
        monkeypatch.setattr(f'subprocess.{name}', forbidden)
    monkeypatch.setattr('os.system', forbidden)
    monkeypatch.setattr('socket.socket.connect', forbidden)
    monkeypatch.setattr('socket.socket.connect_ex', forbidden)
    monkeypatch.setattr('socket.create_connection', forbidden)
    monkeypatch.setattr('socket.getaddrinfo', forbidden)
    monkeypatch.setattr('asyncio.create_subprocess_exec', forbidden)
    monkeypatch.setattr('asyncio.create_subprocess_shell', forbidden)
    yield
    assert not attempts, 'a caught/background guard failure must never count as a passing test'


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', [False, True])
async def test_admitted_safe_anthropic_source_reaches_worker(tmp_path, monkeypatch, changed):
    from agent import managed_worker as child
    from gateway import session_managed_worker as owner
    from gateway.session_local import _bypass_policy

    source = tmp_path / 'source.py'
    source.write_text('value = 1\n')
    policy = _bypass_policy(dict(cwd=str(tmp_path), model='claude-fixture', provider='anthropic',
        base_url='https://api.anthropic.com', safe_mode=True), private_secrets={})
    monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda *a: policy)
    monkeypatch.setattr('gateway.session_policy.route_launch_key', lambda *a: 'FAKE-key')
    monkeypatch.setattr('agent.model_metadata.get_model_context_length_async',
                        lambda *a, **k: _value(128_000))
    live = SimpleNamespace(source=SimpleNamespace(user_id='human', chat_id='c'), route='managed-source',
        event_stream=SimpleNamespace(lock=threading.Lock()), controls=SimpleNamespace(snapshot=lambda *a: None))
    authority = SimpleNamespace(profile_id=str(tmp_path), runner=None, sessions={'sid': live},
        executing={'sid': 1}, pending_results={}, adopt_agent=lambda *a: None)
    ref = SimpleNamespace(session_id='sid')
    row = dict(admission_id='adm', generation=1, principal_id='human',
        payload={'text': 'Inspect @file:source.py:1-1', 'source_slices_v1': [{'forged': True}]})
    scope = dict(profile_id=str(tmp_path), session_id='sid', execution_id='x', generation=1,
                 pid=123, birth=0, secret='fake-scope', epoch=1)
    captured = []

    class Worker:
        def __init__(self, process):
            self.stop, self.secrets = threading.Event(), []
            self.writer = SimpleNamespace(start=lambda: None)
            self.frames = iter([{}, {'type': 'result', 'result': {'final_response': 'ok'}}, {'type': 'finished'}])
        async def next_frame(self, *a, **k): return next(self.frames)
        def send(self, frame):
            if 'version' in frame: captured.append(frame)
        def close(self): pass

    monkeypatch.setattr(owner, 'ManagedWorker', Worker)
    monkeypatch.setattr(owner, '_worker_env', lambda *a: {})
    monkeypatch.setattr(owner, 'reserve_admission_worker', lambda *a, **k: scope)
    monkeypatch.setattr(owner, '_prompt_frame', lambda *a: False)
    monkeypatch.setattr(owner, '_settled_session_usage', lambda *a: {})
    monkeypatch.setattr('gateway.session_worker_spawn.acquire_process',
                        lambda *a, **k: _value(SimpleNamespace(wait=lambda *a: None)))
    monkeypatch.setattr('gateway.session_managed_worker_log.worker_log_path', lambda *a: None)
    monkeypatch.setattr('gateway.session_runtime_workers.tracked_write', lambda a, fn: _value(fn()))
    assert owner.managed_policy(authority, ref) is policy
    assert await owner.execute_managed(authority, ref, row, policy) == 'ok'
    [frame] = captured
    assert 'value = 1' in frame['text']
    assert len(frame.get('source_slices_v1', [])) == 1
    # The admitted payload cannot supply private source facts.
    assert 'forged' not in json.dumps(frame['source_slices_v1'])
    consumer = SimpleNamespace(session_id='sid')
    if changed:
        source.write_text('value = 2\n')
        with pytest.raises(SourceProvenanceError, match='content_mismatch'):
            child._construct_agent(frame, lambda **k: consumer, policy, api_key='FAKE-key')
        assert consumer._source_provenance_pending_turn_id is None
    else:
        assert child._construct_agent(frame, lambda **k: consumer, policy, api_key='FAKE-key') is consumer
        turn = consumer._source_provenance_pending_turn_id
        [grant] = consumer._source_provenance_registry.grants_for_request(f'{turn}:api:1')
        assert grant.session_id == 'sid' and grant.canonical_path == source


async def _value(value):
    return value


@pytest.mark.parametrize('bad', ['extra', 'grant', 'relative', 'bool', 'range', 'bytes', 'count', 'aggregate'])
def test_private_bootstrap_rejects_invalid_source_facts(tmp_path, monkeypatch, bad):
    from agent.managed_worker import validate_bootstrap
    from tests.gateway.test_managed_turn_contract import assignment
    frame, _ = assignment(tmp_path, monkeypatch)
    fact = dict(path=str(tmp_path / 'source.py'), line_start=1, line_end=1, content='value = 1\n')
    facts = [fact]
    if bad == 'extra': fact['unexpected'] = True
    elif bad == 'grant': fact['session_id'] = 'forged-consumer'
    elif bad == 'relative': fact['path'] = 'source.py'
    elif bad == 'bool': fact['line_start'] = True
    elif bad == 'range': fact['line_end'] = 2001
    elif bad == 'bytes': fact['content'] = 'x' * 262_145
    elif bad == 'count': facts *= 65
    elif bad == 'aggregate':
        fact['content'] = 'x' * 140_000
        facts *= 2
    with pytest.raises(ValueError, match='invalid_managed_source_slices'):
        validate_bootstrap({**frame, 'source_slices_v1': facts})


def test_worker_source_facts_cannot_select_another_consumer(tmp_path):
    from gateway.source_context import bind_worker_source_slices
    source = tmp_path / 'source.py'
    source.write_text('value = 1\n')
    frame = {'scope': {'session_id': 'assigned'}, 'source_slices_v1': [dict(
        path=str(source), line_start=1, line_end=1, content='value = 1\n')]}
    consumer = SimpleNamespace(session_id='other')
    with pytest.raises(ValueError, match='invalid_managed_source_identity'):
        bind_worker_source_slices(consumer, frame)
    assert not hasattr(consumer, '_source_provenance_registry')
