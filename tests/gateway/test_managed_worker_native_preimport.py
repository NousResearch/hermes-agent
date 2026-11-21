"""Windows Kanban managed-worker hang: numpy (scipy-openblas) loaded lazily during agent
construction never returned from its DLL initialisation once the worker's reader threads were
running. ``main()`` now imports it first, while the worker is still single-threaded.
"""
import builtins
import sys

from agent import managed_worker


class _NullStream:
    def close(self):
        pass


def _main_call_order(monkeypatch):
    """Run main() up to the bootstrap read (which raises EOF) and record the call order."""
    order = []
    monkeypatch.setattr(managed_worker.os, 'dup', lambda fd: fd)
    monkeypatch.setattr(managed_worker.os, 'dup2', lambda a, b: None)
    monkeypatch.setattr(managed_worker.os, 'fdopen', lambda *a, **k: _NullStream())

    class _FakeChannel:
        def __init__(self, stream):
            self.stream = _NullStream()
            order.append('channel')

        def send(self, kind, **payload):
            order.append(f'send:{kind}')

    def _no_bootstrap(stream):
        raise EOFError('managed_worker_pipe_closed')

    monkeypatch.setattr(managed_worker, '_preimport_native_extensions', lambda: order.append('preimport'))
    monkeypatch.setattr(managed_worker, 'WorkerChannel', _FakeChannel)
    monkeypatch.setattr(managed_worker, 'hello', lambda: order.append('hello') or {})
    monkeypatch.setattr(managed_worker, 'read_frame', _no_bootstrap)
    assert managed_worker.main() == 1
    return order


def _record_imports(monkeypatch, fail_numpy=False):
    seen = []
    real_import = builtins.__import__

    def _spy(name, *args, **kwargs):
        seen.append(name)
        if fail_numpy and name == 'numpy':
            raise ImportError('numpy not installed')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _spy)
    return seen


def test_preimport_runs_before_the_channel_and_hello(monkeypatch):
    order = _main_call_order(monkeypatch)
    assert order[0] == 'preimport'
    assert order.index('preimport') < order.index('channel') < order.index('hello')


def test_windows_preimport_loads_numpy(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    seen = _record_imports(monkeypatch)
    managed_worker._preimport_native_extensions()
    assert 'numpy' in seen


def test_windows_preimport_tolerates_missing_numpy(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    seen = _record_imports(monkeypatch, fail_numpy=True)
    assert managed_worker._preimport_native_extensions() is None
    assert 'numpy' in seen


def test_posix_preimport_is_a_no_op(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    seen = _record_imports(monkeypatch)
    managed_worker._preimport_native_extensions()
    assert 'numpy' not in seen
