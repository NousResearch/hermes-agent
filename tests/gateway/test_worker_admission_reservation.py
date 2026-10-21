"""Owner-issued admission reservations never weaken idle public registration."""
import os
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from hermes_state import SessionDB
from hermes_state_runtime import (admit_session_input, begin_runtime_epoch, claim_session_input,
                                  register_worker_execution, settle_session_input)


def hello(process):
    """A direct child's self-introduction (no launcher: chain length 0)."""
    return {'type': 'hello', 'pid': process.pid, 'birth': psutil.Process(process.pid).create_time(), 'ancestors': [os.getpid()]}


def test_started_admission_binds_only_exact_owned_process_and_principal(tmp_path):
    from gateway.session_worker_reservation import reserve_admission_worker
    db = SessionDB(tmp_path / 'state.db')
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        db.create_session('owned', 'cli')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        authority = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None)
        admitted = admit_session_input(db, epoch=epoch, principal_id='human', session_id='owned', request_id='input', payload={})
        kwargs = dict(admission_id=admitted['admission_id'], process=process, principal_id='human', hello=hello(process))
        with pytest.raises(Exception, match='producer_not_started'):
            reserve_admission_worker(authority, **kwargs)
        claim = claim_session_input(db, epoch=epoch, session_id='owned')
        with pytest.raises(Exception, match='permission_denied'):
            reserve_admission_worker(authority, **dict(kwargs, principal_id='other'))
        with pytest.raises(Exception, match='stale_generation'):
            register_worker_execution(db, epoch=epoch, execution_id='public', session_id='owned',
                generation=claim['generation'], kind='compute', adoption_secret='public', require_idle=True)
        scope = reserve_admission_worker(authority, **kwargs)
        assert scope['session_id'] == 'owned' and scope['generation'] == claim['generation']
        assert scope['pid'] == process.pid and scope['profile_id'] == str(tmp_path)
        assert db._read_one('SELECT COUNT(*) FROM worker_executions')[0] == 1
        # The reservation is single-use; another bootstrap cannot replace its secret.
        with pytest.raises(Exception, match='admission_conflict'):
            reserve_admission_worker(authority, **kwargs)
        settle_session_input(db, epoch=epoch, admission_id=admitted['admission_id'], generation=claim['generation'], outcome='completed')
        with pytest.raises(Exception, match='producer_not_started'):
            reserve_admission_worker(authority, **kwargs)
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        db.close()


@pytest.mark.asyncio
async def test_reserved_process_uses_existing_authenticated_adoption_and_persistence(tmp_path):
    from gateway.session_worker import worker_request
    from gateway.session_worker_reservation import reserve_admission_worker
    db = SessionDB(tmp_path / 'state.db')
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,  # noqa: ASYNC220 -- Popen returns immediately; the test drives the child through its handle
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        db.create_session('owned', 'cli')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        authority = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None,
                                    wake_after_worker=lambda session_id: None)
        admitted = admit_session_input(db, epoch=epoch, principal_id='human', session_id='owned', request_id='input', payload={})
        claim_session_input(db, epoch=epoch, session_id='owned')
        scope = reserve_admission_worker(authority, admission_id=admitted['admission_id'], process=process, principal_id='human', hello=hello(process))
        actor = SimpleNamespace(subject='human', profile_id=str(tmp_path), capabilities={'worker:adopt'})
        connection = SimpleNamespace(authority=authority, actor=actor)
        identity = {k: v for k, v in scope.items() if k != 'epoch'}
        with pytest.raises(Exception, match='permission_denied'):
            await worker_request(connection, None, identity | {'secret': 'wrong-secret-for-reservation'}, operation='adopt')
        receipt = await worker_request(connection, None, identity, operation='adopt')
        assert receipt['execution_id'] == scope['execution_id']
        params = scope | {'sequence': 1, 'operation': 'execution.finish', 'payload': {}}
        result = await worker_request(connection, None, params, operation='persist')
        assert result['status'] == 'terminal'
        with pytest.raises(Exception, match='stale_generation'):
            await worker_request(connection, None, params | {'sequence': 2}, operation='persist')
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        db.close()


@pytest.mark.asyncio
async def test_input_queued_behind_a_live_worker_is_released_then_runs_when_it_finishes(tmp_path, monkeypatch):
    """A FIFO blocked by a live worker is not an empty FIFO: the messaging waiter is released
    with a pause (never parked on a turn that cannot run yet), and the worker's durable finish
    restarts the drain so the queued input runs without an unrelated submit."""
    import asyncio
    from gateway.config import Platform
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Principal, SessionRef, Submission
    from gateway.session_ingress import admit_message
    from gateway.session_worker import worker_request
    from gateway import session_finite
    from hermes_state_runtime import get_session_admission

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('owned', 'telegram')
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='p', instance_id='owner', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='owner'))
    authority.sessions['owned'] = LiveSession(SimpleNamespace(platform=Platform.TELEGRAM, user_id='human'), 'route')
    ran = []

    async def execute(authority, ref, row):
        ran.append(row['admission_id'])
        return 'answered'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,  # noqa: ASYNC220 -- Popen returns immediately; the test drives the child through its handle
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    actor = Principal('human', 'p', frozenset({'session:submit', 'session:control', 'worker:adopt'}), 't')
    connection = SimpleNamespace(authority=authority, actor=actor)
    ref = SessionRef('p', 'owned')
    scope = {'profile_id': 'p', 'session_id': 'owned', 'execution_id': 'compute', 'generation': 0,
             'pid': process.pid, 'birth': psutil.Process(process.pid).create_time(), 'secret': 'w' * 32}
    try:
        registered = await worker_request(connection, ref, scope | {'kind': 'compute'}, operation='register')
        # Native ingress waits on the admission it committed; a blocked FIFO must answer it.
        event = SimpleNamespace()
        async def admit_native(_event):
            return await authority.submit(actor, Submission('behind', ref, {'text': 'hi'}, 'queue'))
        authority.admit_native = admit_native
        reply = await asyncio.wait_for(admit_message(authority, event), 5)
        assert reply is not None and reply.startswith('⏳'), reply
        assert ran == []
        await worker_request(connection, ref, scope | {'epoch': registered['owner_epoch'], 'sequence': 1,
                                                       'operation': 'execution.finish', 'payload': {}}, operation='persist')
        await asyncio.wait_for(authority.sessions['owned'].task, 5)
        assert len(ran) == 1
        assert get_session_admission(db, admission_id=ran[0])['outcome'] == 'completed'
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        db.close()


@pytest.fixture
def stale_psutil_boot_reference(monkeypatch):
    """Step psutil's macOS boot-time reference the way a clock step leaves it in a long-lived
    process: ``_psosx.INIT_BOOT_TIME`` is frozen at import, and ``create_time()`` adds
    ``abs(INIT_BOOT_TIME - boot_time())``. The real macOS psutil code runs; only the reference,
    which is what a step makes stale, moves (stepping the host clock needs root)."""
    def step(drift):
        monkeypatch.setattr(psutil._psplatform, 'INIT_BOOT_TIME', psutil.boot_time() + drift)
    return step


@pytest.mark.platforms("macos")
@pytest.mark.parametrize('drift', [5.0, -5.0])
def test_reservation_survives_a_clock_step_on_macos(tmp_path, stale_psutil_boot_reference, drift):
    """#140580: after a clock step a long-lived owner's psutil reads a fresh worker's birth
    shifted, while the worker reports its own unshifted one, so the exact match refused every
    managed worker. ``process_birth`` (the kernel start) is what both processes see."""
    from gateway.session_worker_reservation import reserve_admission_worker
    from hermes_cli.process_identity import process_birth
    db = SessionDB(tmp_path / 'state.db')
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        frame = hello(process) | {'birth': process_birth(psutil.Process(process.pid))}  # the worker's own read
        stale_psutil_boot_reference(drift)
        assert psutil.Process(process.pid).create_time() != frame['birth']  # the step is real
        db.create_session('owned', 'cli')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        authority = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None)
        admitted = admit_session_input(db, epoch=epoch, principal_id='human', session_id='owned', request_id='input', payload={})
        claim_session_input(db, epoch=epoch, session_id='owned')
        kwargs = dict(admission_id=admitted['admission_id'], process=process, principal_id='human')
        with pytest.raises(Exception, match='permission_denied'):  # a forged birth still fails
            reserve_admission_worker(authority, hello=frame | {'birth': frame['birth'] + 1}, **kwargs)
        scope = reserve_admission_worker(authority, hello=frame, **kwargs)
        assert scope['pid'] == process.pid and scope['birth'] == frame['birth']
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        db.close()


@pytest.mark.platforms("macos")
@pytest.mark.parametrize('drift', [5.0, -5.0])
def test_process_fingerprints_survive_a_clock_step_on_macos(stale_psutil_boot_reference, drift):
    """KB-6 beyond worker birth: the Kanban submitter fingerprint (``_process_fingerprint`` via
    ``get_process_start_time``; ``_watch_submitter`` exits on a mismatch), the delegation
    ``owner_started_at`` and the spawn ledger's ``_pid_alive_matches`` read the same live process
    the same way before and after a clock step."""
    from gateway.status import get_process_start_time
    from hermes_cli.kanban_db_dispatch import _pid_recycled, _process_fingerprint
    from hermes_cli.process_identity import _pid_alive_matches, _process_create_time
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        start, fingerprint, create = (get_process_start_time(process.pid), _process_fingerprint(process.pid),
                                      _process_create_time(process.pid))
        stale_psutil_boot_reference(drift)
        assert abs(psutil.Process(process.pid).create_time() - create) >= 4  # the step is real
        assert get_process_start_time(process.pid) == start
        assert _process_fingerprint(process.pid) == fingerprint and not _pid_recycled(process.pid, fingerprint)
        assert _pid_alive_matches(process.pid, create, strict=True) is True
        assert _pid_alive_matches(process.pid, create) is True
    finally:
        process.stdin.close()
        process.wait(timeout=5)


def test_process_birth_is_create_time_where_psutil_applies_no_reference():
    """Linux and Windows read ``create_time()`` unchanged, so every stored fingerprint and ledger
    row written before KB-6 still matches there; on macOS with no step psutil returns the kernel
    start unchanged too (``adjust_proc_create_time`` ignores |drift| < 1 s)."""
    from gateway.status import get_process_start_time
    from hermes_cli.process_identity import _pid_alive_matches, process_birth
    proc = psutil.Process()
    assert process_birth(proc) == proc.create_time()
    assert _pid_alive_matches(os.getpid(), proc.create_time(), strict=True) is True
    assert get_process_start_time(os.getpid()) is not None
