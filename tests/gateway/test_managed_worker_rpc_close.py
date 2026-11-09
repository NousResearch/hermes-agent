"""A managed worker closes its owner connection when ``execute`` ends, after the store, on success and on
a start failure; a failing close never replaces the turn's own outcome."""
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_managed_turn_contract import assignment


def _execute(tmp_path, monkeypatch, events, *, start_error=None, close_error=None):
    from agent import managed_worker as worker

    class RPC:
        def __init__(self, home): pass
        def __call__(self, method, **params): return {'owner_epoch': 1}
        def close(self):
            events.append('rpc.close')
            if close_error is not None:
                raise close_error

    class Agent:
        def __init__(self, **kwargs):
            if start_error is not None:
                raise start_error
            self.session_prompt_tokens = self.session_completion_tokens = 0
        def run_conversation(self, text, **kwargs):
            return {'final_response': 'ok', 'completed': True}

    class Store:
        def __init__(self, *args): pass
        def get_messages_as_conversation(self, sid): return []
        def flush_token_counts(self): pass
        def finish(self): events.append('store.finish')
        def close(self): events.append('store.close')

    class Controls:
        def __init__(self, *args):
            self.stopped, self.finish = threading.Event(), threading.Event()
            self.finish.set()
        def approval(self, data): pass
        def clarify(self, questions): pass

    frame, _ = assignment(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, 'bind_worker_policy', lambda frame: None)
    monkeypatch.setattr(worker, 'discover_profile_mcp', lambda policy: None)
    monkeypatch.setattr(worker, 'retire_agent', lambda agent: None)
    monkeypatch.setattr(worker, 'WorkerControls', Controls)
    monkeypatch.setattr('agent.runtime_session_store.WorkerRPC', RPC)
    monkeypatch.setattr('agent.runtime_session_store.RuntimeSessionStore', Store)
    monkeypatch.setattr('tools.process_registry.process_registry.recover_from_checkpoint', lambda: None)
    monkeypatch.setattr('run_agent.AIAgent', Agent)
    monkeypatch.setattr('agent.title_generator.wait_for_title_upgrades', lambda: None)
    worker.execute(frame, SimpleNamespace(send=lambda kind, **payload: events.append(kind)))


def test_execute_closes_the_rpc_after_the_store_on_success(tmp_path, monkeypatch):
    events = []
    _execute(tmp_path, monkeypatch, events)
    assert events[-3:] == ['finished', 'store.close', 'rpc.close']


def test_execute_closes_the_rpc_after_the_store_on_a_start_failure(tmp_path, monkeypatch):
    events = []
    with pytest.raises(RuntimeError, match='construct_failed'):
        _execute(tmp_path, monkeypatch, events, start_error=RuntimeError('construct_failed'))
    assert events == ['store.close', 'rpc.close']


@pytest.mark.parametrize('start_error', [None, RuntimeError('construct_failed')])
def test_a_failing_rpc_close_never_masks_the_turns_outcome(tmp_path, monkeypatch, start_error):
    events = []
    if start_error is None:
        _execute(tmp_path, monkeypatch, events, close_error=OSError('close'))
        assert events[-3:] == ['finished', 'store.close', 'rpc.close']
    else:
        with pytest.raises(RuntimeError, match='construct_failed'):
            _execute(tmp_path, monkeypatch, events, start_error=start_error, close_error=OSError('close'))
        assert events[-1] == 'rpc.close'
