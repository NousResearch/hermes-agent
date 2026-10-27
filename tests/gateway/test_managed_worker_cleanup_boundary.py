"""Only native physical cleanup uncertainty may leave the worker's result handshake alive."""
from io import BytesIO
import shlex
import sys
import threading
from types import SimpleNamespace

import pytest

from agent.managed_worker import WorkerControls, encode_frame
from hermes_state_runtime import RuntimeStoreError


@pytest.mark.parametrize('error', [None, RuntimeError('process_stop_unknown'),
    RuntimeError('storage_failure'), RuntimeError('process_stop_unknown:detail'),
    OSError('process_stop_unknown'), RuntimeStoreError('process_stop_unknown')],
    ids=['settled', 'native-cleanup-unknown', 'different-runtime-error', 'different-message',
         'io-error', 'genuine-store-error'])
def test_cleanup_retry_preserves_real_errors(error):
    controls = WorkerControls.__new__(WorkerControls)
    calls = []
    def cleanup():
        calls.append('cleanup')
        if error is not None:
            raise error
    controls.stop_processes = cleanup
    if error is None:
        assert controls.retry_process_stop() is True
    elif type(error) is RuntimeError and str(error) == 'process_stop_unknown':
        assert controls.retry_process_stop() is False
    else:
        with pytest.raises(type(error)) as raised:
            controls.retry_process_stop()
        assert raised.value is error
    assert calls == ['cleanup']


@pytest.mark.platforms('linux')
def test_actual_unreadable_job_ownership_keeps_finish_control_and_stop_fanout(tmp_path, monkeypatch):
    from tools import process_registry as module
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    job = registry.spawn_local(shlex.join([sys.executable, '-c', "print('original'); raise SystemExit(7)"]),
        cwd=str(tmp_path), task_id='s', owner_task_id='s', session_key='route')
    assert job._completion_event.wait(10)
    path = tmp_path / 'logs' / 'process-results' / (job.id + '.json')
    original = path.read_bytes()
    controls = WorkerControls.__new__(WorkerControls)
    controls.route, controls.stopped, controls.finish = 'route', threading.Event(), threading.Event()
    controls.stop_processes_requested = False
    clarification, interrupted = threading.Event(), threading.Event()
    controls.lock = threading.Lock()
    controls.clarifications = {'pending': {'event': clarification, 'answer': None}}
    controls.agent = SimpleNamespace(session_id='s', hard_interrupt=lambda *a, **kw: interrupted.set())
    frames = encode_frame({'type': 'stop', 'processes': True}) + encode_frame({'type': 'finish'})
    monkeypatch.setattr(sys, 'stdin', SimpleNamespace(buffer=BytesIO(frames)))
    try:
        path.write_text('{unreadable retained ownership')
        with pytest.raises(RuntimeError, match='^process_stop_unknown$'):
            controls.stop_processes()
        controls.read()
        assert controls.stopped.is_set() and controls.finish.is_set() and interrupted.is_set()
        assert clarification.is_set() and controls.stop_processes_requested
        assert controls.clarifications['pending']['answer'] is not None
        assert path.read_text() == '{unreadable retained ownership'
        assert job.output_buffer == 'original\n' and job.exit_code == 7 and job._result_complete
    finally:
        path.write_bytes(original)
    assert controls.retry_process_stop() is True
