"""Non-sensitive, identity-bound wait observations never stand in for a receipt."""
import json
import threading
from types import SimpleNamespace

from agent.runtime_session_store import WorkerRPC


def test_progress_reports_waiting_on_same_operation_without_extending_deadline(tmp_path, monkeypatch):
    import time
    rpc = WorkerRPC(tmp_path)
    rpc.budget = SimpleNamespace(rpc_timeout_seconds=.4, recovery_timeout_seconds=.1,
                                progress_interval_seconds=.03, close_timeout_seconds=.05)
    entered, release = threading.Event(), threading.Event()
    result, errors = [], []
    class Link:
        def send(self, request_id, message):
            entered.set()
        def receive(self, timeout):
            if not release.wait(timeout):
                raise TimeoutError('worker_receipt_timeout')
            return {'result': 'stored'}
    monkeypatch.setattr(rpc, '_connection', lambda: Link())
    params = dict(session_id='synthetic',execution_id='exec',generation=0,epoch=1,
                  sequence=7,operation='session.activity',payload={'ts':1},secret='must-not-appear')
    def call():
        try:
            result.append(rpc('worker.persist', **params))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=call)
    try:
        thread.start()
        assert entered.wait(2)
        from tests.waiting import wait_until
        wait_until(lambda: getattr(rpc, 'observation', {}).get('phase') == 'awaiting_receipt', timeout=2)
        first = dict(rpc.observation)
        assert first['sequence'] == 7 and first['execution_id'] == 'exec'
        assert 'must-not-appear' not in json.dumps(first) and 'payload' not in first
        thread.join(2)
        assert not thread.is_alive() and isinstance(errors[0], TimeoutError)
        assert rpc.observation['phase'] == 'unknown'
        assert rpc.observation['deadline'] == first['deadline']
    finally:
        release.set()
        thread.join(3)
