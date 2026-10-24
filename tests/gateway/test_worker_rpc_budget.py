"""Budget contracts at the real synchronous RPC boundary; no model/tool lifetime cap."""
import json
import pytest
from types import SimpleNamespace

from agent.runtime_session_store import WorkerRPC
from hermes_state import SessionDB


def test_default_receipt_observation_outlasts_storage_retry(tmp_path, monkeypatch):
    """A healthy permitted DB retry must not outlast the caller observation budget."""
    observed = []
    class Link:
        reader = SimpleNamespace(is_alive=lambda: True)
        def send(self, request_id, message):
            self.request_id = request_id
            assert json.loads(message)['method'] == 'worker.persist'
        def receive(self, timeout):
            observed.append(timeout)
            return {'id': self.request_id, 'result': {'stored': True}}
    rpc = WorkerRPC(tmp_path)
    monkeypatch.setattr(rpc, '_connection', lambda: Link())
    assert rpc('worker.persist', operation='session.activity') == {'stored': True}
    assert observed[0] > SessionDB._TRANSCRIPT_WRITE_PATIENCE_S, (
        'receipt wait expires before allowed storage retry plus queue/maintenance/send margin', observed)


def test_home_scoped_config_changes_receipt_budget(tmp_path, monkeypatch):
    (tmp_path / 'config.yaml').write_text('worker_persistence:\n  rpc_timeout_seconds: 180\n  recovery_timeout_seconds: 25\n')
    observed = []
    class Link:
        def send(self, request_id, message):
            self.request_id = request_id
        def receive(self, timeout):
            observed.append(timeout)
            return {'id': self.request_id, 'result': 'stored'}
    rpc = WorkerRPC(tmp_path)
    monkeypatch.setattr(rpc, '_connection', lambda: Link())
    assert rpc('worker.persist') == 'stored'
    assert 150 < observed[0] <= 155, observed


def test_rpc_lock_queue_is_bounded_without_sending(tmp_path, monkeypatch):
    import threading
    import time
    rpc = WorkerRPC(tmp_path)
    # Explicit test-clock policy, never a serving config mutation.
    rpc.budget = SimpleNamespace(rpc_timeout_seconds=.2, recovery_timeout_seconds=.05,
                                progress_interval_seconds=.025, close_timeout_seconds=.1)
    rpc.lock.acquire()
    calls, errors = [], []
    monkeypatch.setattr(rpc, '_connection', lambda: calls.append('unexpected connection'))
    def call():
        try:
            rpc('worker.persist')
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=call)
    start = time.monotonic()
    thread.start()
    thread.join(.6)
    bounded = not thread.is_alive()
    rpc.lock.release()
    thread.join(2)
    assert bounded, 'RPC lock queue exceeded finite observation deadline'
    assert isinstance(errors[0], TimeoutError) and not calls
    assert time.monotonic() - start < 1


def test_hung_send_returns_unknown_and_fences_unsettled_transport(tmp_path, monkeypatch):
    import threading
    import time
    from agent.runtime_session_store import WorkerPersistenceError
    rpc = WorkerRPC(tmp_path)
    rpc.budget = SimpleNamespace(rpc_timeout_seconds=.3, recovery_timeout_seconds=.05,
                                progress_interval_seconds=.025, close_timeout_seconds=.05)
    entered, release = threading.Event(), threading.Event()
    calls, errors = [], []
    class Link:
        def send(self, request_id, message):
            calls.append(json.loads(message))
            entered.set()
            release.wait(5)
        def receive(self, timeout):
            return {'result': 'late'}
    monkeypatch.setattr(rpc, '_connection', lambda: Link())
    def call():
        try:
            rpc('worker.persist', sequence=1)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=call)
    try:
        thread.start()
        assert entered.wait(2)
        thread.join(1)
        assert not thread.is_alive(), 'send blocked beyond RPC end-to-end deadline'
        assert isinstance(errors[0], TimeoutError)
        with __import__('pytest').raises(WorkerPersistenceError, match='unsettled'):
            rpc('worker.persist', sequence=2)
        assert len(calls) == 1
    finally:
        release.set()
        thread.join(3)


def test_send_error_after_dispatch_never_resubmits_mutation(tmp_path, monkeypatch):
    from websockets.exceptions import ConnectionClosedError
    rpc = WorkerRPC(tmp_path)
    calls = []
    class Link:
        def send(self, request_id, message):
            calls.append(json.loads(message))
            raise ConnectionClosedError(None, None)
    monkeypatch.setattr(rpc, '_connection', lambda: Link())
    with __import__('pytest').raises(ConnectionClosedError):
        rpc('worker.persist', sequence=1)
    assert len(calls) == 1, 'unknown send outcome must not trigger another mutation dispatch'


def test_hung_close_has_finite_grace_and_retains_transport(tmp_path, monkeypatch):
    import threading
    rpc = WorkerRPC(tmp_path)
    rpc.budget = SimpleNamespace(rpc_timeout_seconds=.3, recovery_timeout_seconds=.05,
                                progress_interval_seconds=.025, close_timeout_seconds=.05)
    release = threading.Event()
    class Link:
        ws = SimpleNamespace(close=lambda: release.wait(5))
        def send(self, request_id, message):
            raise OSError('lost send')
    rpc.link = Link()
    monkeypatch.setattr(rpc, '_connection', lambda: rpc.link)
    errors = []
    def call():
        try:
            rpc('worker.persist', sequence=1)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=call)
    try:
        thread.start()
        thread.join(1)
        assert not thread.is_alive(), 'close handshake defeated the RPC bound'
        assert isinstance(errors[0], OSError)
    finally:
        release.set()
        thread.join(3)


@pytest.mark.parametrize('value', [True, 0, -1, 60, '.inf', '.nan', 'text', 4000])
def test_invalid_user_budget_fails_closed(tmp_path, value):
    encoded = str(value).lower() if isinstance(value, bool) else str(value)
    (tmp_path / 'config.yaml').write_text('worker_persistence:\n  rpc_timeout_seconds: ' + encoded + '\n')
    with pytest.raises(ValueError, match='invalid_worker_persistence_budget'):
        WorkerRPC(tmp_path)


def test_frozen_worker_budget_wins_over_later_home_config(tmp_path, monkeypatch):
    from agent import safe_worker_policy
    (tmp_path / 'config.yaml').write_text('worker_persistence:\n  rpc_timeout_seconds: 180\n')
    monkeypatch.setattr(safe_worker_policy, 'worker_config_snapshot',
                        lambda: {'worker_persistence': {'rpc_timeout_seconds': 240}})
    assert WorkerRPC(tmp_path).budget.rpc_timeout_seconds == 240


def test_empty_frozen_budget_uses_defaults_without_profile_read(tmp_path, monkeypatch):
    from agent import safe_worker_policy
    from agent.worker_rpc_budget import WorkerRPCBudget
    (tmp_path / 'config.yaml').write_text('worker_persistence:\n  rpc_timeout_seconds: 180\n')
    monkeypatch.setattr(safe_worker_policy, 'worker_config_snapshot', lambda: {})
    assert WorkerRPC(tmp_path).budget == WorkerRPCBudget()


@pytest.mark.parametrize('value', [True, 0, 60, 4000])
def test_invalid_frozen_budget_is_not_replaced_by_valid_profile(tmp_path, monkeypatch, value):
    from agent import safe_worker_policy
    (tmp_path / 'config.yaml').write_text('worker_persistence:\n  rpc_timeout_seconds: 180\n')
    monkeypatch.setattr(safe_worker_policy, 'worker_config_snapshot',
                        lambda: {'worker_persistence': {'rpc_timeout_seconds': value}})
    with pytest.raises(ValueError, match='invalid_worker_persistence_budget'):
        WorkerRPC(tmp_path)


@pytest.mark.parametrize('rpc_seconds', [None, 240], ids=['defaults', 'explicit'])
def test_native_bound_snapshot_ignores_later_malformed_profile(tmp_path, rpc_seconds):
    import os
    from pathlib import Path
    import subprocess
    import sys
    config = {} if rpc_seconds is None else {'worker_persistence': {'rpc_timeout_seconds': rpc_seconds}}
    expected = 120 if rpc_seconds is None else rpc_seconds
    # A fresh real interpreter binds the actual process-wide policy before runtime imports.
    child = '''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from agent.safe_worker_policy import _bind_worker_config
_bind_worker_config(json.loads(sys.argv[3]))
(Path(sys.argv[2])/'config.yaml').write_text('worker_persistence: [invalid')
from agent.runtime_session_store import WorkerRPC
rpc=WorkerRPC(Path(sys.argv[2]))
assert rpc.budget.rpc_timeout_seconds == int(sys.argv[4])
rpc.close()
'''
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, HERMES_HOME=str(tmp_path), HOME=str(tmp_path))
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
        env.pop(key, None)
    result = subprocess.run([sys.executable, '-I', '-B', '-c', child, str(root), str(tmp_path),
                             json.dumps(config), str(expected)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr

