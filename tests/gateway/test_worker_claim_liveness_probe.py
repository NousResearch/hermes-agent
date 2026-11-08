"""``worker.persist`` must not fail an authenticated write when its liveness probe cannot run.

2026-11-06, fleet-wide: ``gateway/session_worker.py::_claim`` probes the claiming process with
psutil on every persist. ``psutil.Process.status()`` on Windows allocates the process table in C and
raised ``MemoryError`` under host memory pressure — not a ``psutil.Error``, so it escaped ``_claim``,
escaped the dispatch boundary (which answers only ``RuntimeStoreError``/``sqlite3.Error``), and the
owner replied JSON-RPC ``-32603`` "internal error". The worker's store made that failure sticky, so
every later write (transcript, turn lease, and the close-time flush) raised, the child exited 1, the
owner settled its admission ``unknown``, and the Kanban run died with
``Kanban execution unknown; retained by owner`` — after the agent had already produced its answer.

The probe is a secondary PID-recycle fence, not the authentication: the claim digest binds the
caller to its stored assignment. A probe that could not be EVALUATED is unprovable, never evidence
against the caller, so the write continues. A probe that evaluated and disagreed still refuses.
"""
import contextlib
import os
import subprocess
import sys

import psutil
import pytest

from gateway.session_worker import _claim
from hermes_state_runtime import RuntimeStoreError

_PROFILE = 'engineering-lead'
_SECRET = 's' * 32


class _Conn:
    def execute(self, *_args, **_kwargs):
        return self

    def fetchone(self):
        return None


class _Db:
    """The one read ``_claim`` makes: no retained worker assignment, so origin = session_id."""

    def _read_ctx(self):
        return contextlib.nullcontext(_Conn())


class _Actor:
    profile_id = _PROFILE
    capabilities = frozenset({'worker:adopt'})
    subject = 'local'


class _Authority:
    profile_id = _PROFILE
    db = _Db()


class _Connection:
    authority = _Authority()
    actor = _Actor()


class _Process:
    def __init__(self, *, status=None, username: object = 'brett', birth=1000.0):
        self._status, self._username, self._birth = status, username, birth

    def is_running(self):
        return True

    def status(self):
        if isinstance(self._status, BaseException):
            raise self._status
        return self._status if self._status is not None else psutil.STATUS_RUNNING

    def username(self):
        if isinstance(self._username, BaseException):
            raise self._username
        return self._username


def _params(**overrides):
    params = {'profile_id': _PROFILE, 'session_id': 'local-owned', 'execution_id': 'w1',
              'generation': 1, 'pid': 4242, 'birth': 1000.0, 'secret': _SECRET}
    params.update(overrides)
    return params


def _install(monkeypatch, process, *, birth=1000.0, factory_error=None):
    def factory(pid):
        if factory_error is not None:
            raise factory_error
        return process
    monkeypatch.setattr(psutil, 'Process', factory)
    monkeypatch.setattr('hermes_cli.process_identity.worker_birth', lambda proc: birth)


def test_an_unevaluable_liveness_probe_does_not_fail_an_authenticated_claim(monkeypatch, caplog):
    """The regression: ``status()`` raising MemoryError must not surface as a failed persist."""
    _install(monkeypatch, _Process(status=MemoryError()))
    with caplog.at_level('WARNING', logger='gateway.session_worker'):
        claim = _claim(_Connection(), None, _params())
    assert isinstance(claim, str) and len(claim) == 64
    assert 'unprovable' in caplog.text


def test_an_unevaluable_username_probe_is_unprovable_too(monkeypatch):
    _install(monkeypatch, _Process(username=MemoryError()))
    assert len(_claim(_Connection(), None, _params())) == 64


def test_a_zombie_worker_is_still_refused(monkeypatch):
    _install(monkeypatch, _Process(status=psutil.STATUS_ZOMBIE))
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        _claim(_Connection(), None, _params())


def test_a_foreign_pid_birth_is_still_refused(monkeypatch):
    _install(monkeypatch, _Process(), birth=2000.0)
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        _claim(_Connection(), None, _params())


def test_a_vanished_worker_is_refused_cleanly(monkeypatch):
    _install(monkeypatch, None, factory_error=psutil.NoSuchProcess(4242))
    with pytest.raises(RuntimeStoreError, match='worker_not_live'):
        _claim(_Connection(), None, _params())


class _Handle:
    def __init__(self, *, ppid=None, status=psutil.STATUS_RUNNING):
        self._ppid, self._status = ppid, status

    def ppid(self):
        if isinstance(self._ppid, BaseException):
            raise self._ppid
        return os.getpid() if self._ppid is None else self._ppid

    def status(self):
        return self._status


def test_an_unevaluable_identity_probe_refuses_the_spawn_cleanly(monkeypatch):
    """The spawn fence must not answer -32603 either: an unverifiable chain is a clean refusal."""
    from gateway.session_worker_reservation import verify_worker_identity
    process = _spawn_short_lived()
    try:
        monkeypatch.setattr(psutil, 'Process', lambda pid: _Handle(ppid=MemoryError()))
        with pytest.raises(RuntimeStoreError, match='worker_not_live'):
            verify_worker_identity(process, _hello(process.pid))
    finally:
        _kill(process)


def test_an_identity_mismatch_stays_permission_denied(monkeypatch):
    """The containment must not rewrite a probe that DID evaluate and disagreed."""
    from gateway.session_worker_reservation import verify_worker_identity
    process = _spawn_short_lived()
    try:
        monkeypatch.setattr(psutil, 'Process',
                            lambda pid: _Handle() if pid == process.pid else _Process())
        monkeypatch.setattr('hermes_cli.process_identity.worker_birth', lambda proc: 2000.0)
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            verify_worker_identity(process, _hello(process.pid))
    finally:
        _kill(process)


def _hello(pid, birth=1000.0):
    return {'type': 'hello', 'pid': pid, 'birth': birth, 'ancestors': []}


def _spawn_short_lived():
    return subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _kill(process):
    process.kill()
    process.wait(timeout=10)
