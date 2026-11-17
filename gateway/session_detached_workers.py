"""Owner side of ``gateway.detached_workers``: managed-worker turns that survive a gateway restart.

With ``gateway.managed_workers`` and ``gateway.detached_workers`` both true (POSIX only), a turn's
worker is told a private unix socket at bootstrap and the reservation writes a durable record of
it (``RECORD_PREFIX`` + execution id in ``state_meta``, deleted when the execution goes terminal).
The worker no longer dies with its owner's pipe (``agent/managed_worker_detach.py``); a stopping
gateway hands its turns over instead of interrupting them (``stop_authority_turns``).

At boot, before ``recover_session_inputs`` turns every older-epoch ``started`` admission into
``unknown``, ``spare_live_workers`` moves the executions whose recorded worker is still alive (same
pid, kernel birth and user) and their admissions to the new epoch. ``resume_live_workers`` then
runs each one as that session's drain: the claim is stamped as if just claimed, and
``execute_managed`` reattaches over the socket (``reattach``) instead of spawning, so the rest of
the turn's frames reach the same settlement path a never-interrupted turn takes. A record whose
worker is gone, or that fails its handshake, keeps today's verdict: the admission is ``unknown``.
"""
import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile

from agent.managed_worker import encode_frame, outbox_dir, read_frame
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from hermes_state_runtime_workers import DETACHED_RECORD_PREFIX as RECORD_PREFIX, descendant_range

logger = logging.getLogger(__name__)

HANDSHAKE_SECONDS = 15
_MAX_UNIX_PATH = 100  # sun_path is 104 bytes on macOS/BSD, 108 on Linux
_PEERCRED = struct.Struct('3i')


def detach_enabled(policy):
    """The frozen session config opts in (with ``managed_workers``, which routed it here)."""
    gateway = policy.config().get('gateway', {})
    return os.name != 'nt' and isinstance(gateway, dict) and gateway.get('detached_workers') is True


def socket_path(home, execution_id):
    """The worker's private listener: in its 0700 outbox directory when that fits ``sun_path``,
    else in a fresh 0700 ``mkdtemp`` directory (a long profile home)."""
    direct = outbox_dir(home, execution_id) / 'control.sock'
    if len(str(direct).encode()) <= _MAX_UNIX_PATH:
        direct.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return str(direct)
    from gateway.control_socket import _private_runtime_dir
    root = _private_runtime_dir() or Path(tempfile.gettempdir())
    return str(Path(tempfile.mkdtemp(prefix='hmw-', dir=root)) / 'w.sock')


def write_record(conn, scope, admission_id, path):
    """Inside the reservation's transaction: what the next owner needs to find and verify it."""
    record = {'pid': scope['pid'], 'birth': scope['birth'], 'execution_id': scope['execution_id'],
              'generation': scope['generation'], 'session_id': scope['session_id'],
              'admission_id': admission_id, 'socket': path}
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                 (RECORD_PREFIX + scope['execution_id'], json.dumps(record)))


def drop_record(conn, execution_id):
    """A record whose worker is gone (``compact_terminal_receipts`` drops it at every terminal
    transition too)."""
    conn.execute('DELETE FROM state_meta WHERE key=?', (RECORD_PREFIX + execution_id,))


def _worker_alive(record):
    import psutil
    from hermes_cli.process_identity import worker_birth
    try:
        proc = psutil.Process(record['pid'])
        return (worker_birth(proc) == record['birth'] and proc.status() != psutil.STATUS_ZOMBIE
                and proc.username() == psutil.Process().username())
    except psutil.Error:
        return False


def spare_live_workers(db, epoch):
    """Boot fence, before ``recover_session_inputs``: move each live recorded worker's execution,
    its delegated children and its still-started admission to ``epoch``; forget dead records.
    Returns the spared records."""
    with db._read_ctx() as conn:
        records = [json.loads(value) for (value,) in conn.execute(
            'SELECT value FROM state_meta WHERE key>=? AND key<?', (RECORD_PREFIX, RECORD_PREFIX[:-1] + '/'))]
    alive = {record['execution_id']: _worker_alive(record) for record in records}

    def write(conn):
        spared = []
        for record in records:
            execution_id = record['execution_id']
            moved = alive[execution_id] and conn.execute(
                "UPDATE worker_executions SET owner_epoch=? WHERE execution_id=? AND generation=? "
                "AND status IN ('registered','running')", (epoch, execution_id, record['generation'])).rowcount
            admitted = moved and conn.execute(
                "UPDATE session_admissions SET owner_epoch=? WHERE admission_id=? AND generation=? AND status='started'",
                (epoch, record['admission_id'], record['generation'])).rowcount
            if not admitted:
                drop_record(conn, execution_id)
                if moved:  # its admission settled without it: recovery decides as before
                    conn.execute("UPDATE worker_executions SET status='unknown' WHERE execution_id=?", (execution_id,))
                continue
            conn.execute("UPDATE worker_executions SET owner_epoch=? WHERE execution_id>=? AND execution_id<? "
                         "AND status!='terminal'", (epoch, *descendant_range(execution_id)))
            spared.append(record)
        return spared
    return db._execute_write(write)


async def resume_live_workers(authority, records):
    """Run each spared turn as its session's drain (after ``recover_local_sessions``). A turn with
    no restored session to run in is lost as before: ``unknown``, and its worker ended."""
    from gateway.session_authority import _log_drain_failure
    from gateway.session_runtime_workers import persist_work_count
    resumes = authority.detached_resumes = {}
    for record in records:
        row = await asyncio.to_thread(get_session_admission, authority.db, admission_id=record['admission_id'])
        live = authority.sessions.get(row['target_session_id'])
        if live is None or (live.task is not None and not live.task.done()):
            logger.warning('Detached managed worker %s has no session to resume into', record['execution_id'])
            await lose_recorded(authority, row, record)
            continue
        resumes[record['admission_id']] = record
        ref = SessionRef(authority.profile_id, row['target_session_id'])
        live.task = asyncio.create_task(_resume(authority, ref, live, row))
        live.task.add_done_callback(_log_drain_failure)
        live.task.add_done_callback(lambda task: persist_work_count(authority, task))


async def lose_recorded(authority, row, record):
    """Today's verdict for a spared turn this owner cannot run: ``unknown``, worker group ended."""
    from gateway.session_worker_reservation import lose_admission_worker
    with contextlib.suppress(RuntimeStoreError):
        await asyncio.to_thread(lose_admission_worker, authority, row, resume_scope(authority, record))
    await asyncio.to_thread(kill_recorded, record)


def kill_recorded(record):
    """SIGKILL the recorded worker's process group, only while its pid still carries its birth."""
    import signal
    if _worker_alive(record):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(record['pid'], signal.SIGKILL)  # windows-footgun: ok — POSIX only (detach_enabled)


async def _resume(authority, ref, live, row):
    authority._stamp_claim(ref, live, row)
    if await authority._run_claimed(ref, live, row):
        await authority._drain(ref)


def take_resume(authority, row):
    """The spared record ``execute_managed`` reattaches to for this claim, or None (spawn)."""
    return getattr(authority, 'detached_resumes', {}).pop(row['admission_id'], None)


def resume_scope(authority, record):
    return {key: record[key] for key in ('session_id', 'execution_id', 'generation', 'pid', 'birth')} | {
        'epoch': authority.epoch}


class AdoptedProcess:
    """The ``subprocess.Popen`` surface ``ManagedWorker`` uses, over a reattached worker's socket.
    The worker is not this process's child: liveness and signals go through its verified birth."""

    def __init__(self, sock, record):
        self.sock, self.pid, self.birth = sock, record['pid'], record['birth']
        self.stdout, self.stdin, self.stderr = sock.makefile('rb'), sock.makefile('wb'), None
        sock.close()  # the two files keep the descriptor until both close
        self.returncode = None

    def _proc(self):
        import psutil
        from hermes_cli.process_identity import worker_birth
        try:
            proc = psutil.Process(self.pid)
            return proc if worker_birth(proc) == self.birth else None
        except psutil.Error:
            return None

    def poll(self):
        if self.returncode is None and self._proc() is None:
            self.returncode = -1
        return self.returncode

    def wait(self, timeout=None):
        import psutil
        proc = self._proc()
        if proc is not None:
            try:
                proc.wait(timeout)
            except psutil.TimeoutExpired as exc:
                raise subprocess.TimeoutExpired(str(self.pid), timeout) from exc
            except psutil.Error:
                pass
        self.returncode = -1 if self.returncode is None else self.returncode
        return self.returncode

    def _signal(self, method):
        proc = self._proc()
        if proc is not None:
            with contextlib.suppress(Exception):  # health: allow BLE001 -- gone between the check and the signal
                getattr(proc, method)()

    def terminate(self):
        self._signal('terminate')

    def kill(self):
        self._signal('kill')

    def hangup(self):
        """Hand the worker to the next owner: end this side's writes; the worker reads EOF."""
        with contextlib.suppress(OSError):
            self.stdin.flush()
        with contextlib.suppress(OSError):
            socket.socket(fileno=os.dup(self.stdin.fileno())).shutdown(socket.SHUT_WR)


def _handshake(authority, record):
    """Connect, check the listener is the recorded worker (kernel peer pid, then its hello), and
    tell it this owner's epoch. Returns the connected socket."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(HANDSHAKE_SECONDS)
        sock.connect(record['socket'])
        pid, uid, _gid = _PEERCRED.unpack(sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size))
        if pid != record['pid'] or uid != os.getuid():  # windows-footgun: ok — POSIX only (detach_enabled)
            raise RuntimeStoreError('permission_denied')
        reader = sock.makefile('rb', buffering=0)  # unbuffered: nothing past the hello is consumed here
        hello = read_frame(reader)
        reader.close()
        expected = {k: record[k] for k in ('pid', 'birth', 'execution_id', 'generation')}
        if hello != {'type': 'hello', **expected}:
            raise RuntimeStoreError('permission_denied')
        sock.sendall(encode_frame({'type': 'reattached', 'epoch': authority.epoch}))
        sock.settimeout(None)
        return sock
    except BaseException:
        sock.close()
        raise


async def reattach(authority, ref, row, record, worker_class):
    """The ``ManagedWorker`` for a spared turn, registered for Stops like a spawned one."""
    try:
        sock = await asyncio.to_thread(_handshake, authority, record)
    except (OSError, ValueError, EOFError) as exc:
        raise RuntimeStoreError('managed_worker_lost') from exc
    worker = worker_class(AdoptedProcess(sock, record))
    worker.worker, worker.detachable = (record['pid'], record['birth']), True
    workers = getattr(authority, '_managed_workers', None)
    if workers is None:
        workers = authority._managed_workers = {}
    workers[ref.session_id] = worker
    logger.info('Re-adopted managed worker %s pid %s (epoch %s)', record['execution_id'], record['pid'],
                authority.epoch)
    return worker
