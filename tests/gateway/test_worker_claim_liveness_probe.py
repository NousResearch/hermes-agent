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
against the caller, so a digest-gated write continues. A probe that evaluated and disagreed still
refuses.

``register`` is the exception, and the reason this file has a second shape: it CREATES the
assignment and never runs ``_verify``, so no digest stands behind it and the probe is its only
evidence the registering pid is live, not a zombie and ours. There an unprovable probe refuses
cleanly (``worker_not_live``, which the dispatcher retries) instead of parking a ``registered`` row
nothing ever verified.
"""
import asyncio
import contextlib
import os
import subprocess
import sys

import psutil
import pytest

from gateway.session_worker import _claim, worker_request
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
    epoch = 1
    db = _Db()

    # The register path's other gates, completed so this file fails on the PROBE it is about and
    # never on a missing stub (a control that dies on AttributeError discriminates nothing).
    def _require_admission_open(self):
        return None

    def authorize(self, _actor, _ref, _capability):
        return None


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
    def factory(pid=None):
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


def _register(monkeypatch, marker='registered'):
    """Stub only what comes AFTER the probe, so a base tree reaches it and book the registration."""
    monkeypatch.setattr('gateway.session_worker.register_worker_execution',
                        lambda *_a, **_k: marker)


def test_a_register_that_could_not_prove_the_worker_live_is_refused(monkeypatch):
    """``register`` CREATES the assignment and never runs ``_verify``: the probe is its only gate.

    Continuing there accepts a registration for a pid/birth nothing verified and parks a
    ``registered`` row that delays the session's next worker until it is reaped.
    """
    for probe in (_Process(status=MemoryError()), _Process(username=MemoryError())):
        _install(monkeypatch, probe)
        _register(monkeypatch)
        with pytest.raises(RuntimeStoreError, match='worker_not_live'):
            asyncio.run(worker_request(_Connection(), None, _params(kind='compute'), operation='register'))


def test_a_register_whose_probe_evaluated_live_still_registers(monkeypatch):
    """The refusal is about an UNPROVABLE probe, not a blanket refusal of the register lane."""
    _install(monkeypatch, _Process())
    _register(monkeypatch, marker='admitted')
    assert asyncio.run(worker_request(_Connection(), None, _params(kind='compute'),
                                      operation='register')) == 'admitted'


def test_the_digest_gated_lanes_continue_on_that_same_probe(monkeypatch):
    """``adopt``/``persist`` authenticate with ``_verify``'s hmac, so unprovable is no refusal there."""
    _install(monkeypatch, _Process(status=MemoryError()))
    assert len(_claim(_Connection(), None, _params())) == 64
    persist = {'epoch': 1, 'sequence': 1, 'operation': 'turn.lease', 'payload': {}}
    for operation, extra in (('adopt', {}), ('persist', persist)):
        # Past the probe: the lane stops at its real gate (no stored row to verify against), never
        # at 'worker_not_live'. A refusal here would break the write the 2026-11-06 incident was.
        with pytest.raises(RuntimeStoreError, match='not_found'):
            asyncio.run(worker_request(_Connection(), None, _params(**extra), operation=operation))


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
