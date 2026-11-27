"""Owner side of ``gateway.detached_workers``: managed-worker turns that survive a gateway restart.

With ``gateway.managed_workers`` and ``gateway.detached_workers`` both true on Linux, a turn's
worker is told a private unix socket at bootstrap and the reservation writes a durable record of
it (``RECORD_PREFIX`` + execution id in ``state_meta``, deleted when the execution goes terminal).
The worker no longer dies with its owner's pipe (``agent/managed_worker_detach/channel.py``); a stopping
gateway hands its turns over instead of interrupting them (``stop_authority_turns``).

Native admission recovery transfers each opted-in live worker and its still-started admission to
the new epoch together. After publication, each reservation takes over that session's drain:
the claim is stamped as if just claimed, and resumes over the socket (``reattach``) without
routing through new-turn ingress, so the rest of
the turn's frames reach the same settlement path a never-interrupted turn takes. A record whose
worker is gone, or that fails its handshake, keeps today's verdict: the admission is ``unknown``.
"""
import asyncio
import contextlib
import errno
import io
import logging
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import sys
import tempfile
import time

from agent.managed_worker import encode_frame, outbox_dir, read_frame
from gateway.session_contract import SessionRef
from gateway.session_persistence import authority_persistence
from persistence.core.errors import AuthorityRefusal

logger = logging.getLogger(__name__)

HANDSHAKE_SECONDS = 15
_MAX_UNIX_PATH = 100  # sun_path is 104 bytes on macOS/BSD, 108 on Linux
# Last fallback root for a listener directory, as in gateway.control_socket.
_LAST_RESORT_ROOT = '/tmp'  # no-tmp: ok — AF_UNIX path limit needs the short root; the worker's own dir is 0700
_PEERCRED = struct.Struct('3i')


def detach_enabled(policy):
    """The frozen session config opts in (with ``managed_workers``, which routed it here)."""
    # Both ends authenticate with Linux SO_PEERCRED. Don't create a detached
    # reservation on a platform where the worker could never reattach.
    if sys.platform != 'linux' or not hasattr(socket, 'SO_PEERCRED'):
        return False
    gateway = policy.config().get('gateway', {})
    return isinstance(gateway, dict) and gateway.get('detached_workers') is True


def _fits(path):
    return len(os.fsencode(path)) <= _MAX_UNIX_PATH  # the bytes bind() receives, not characters


def _listener_roots():
    """Short roots for a fallback listener, in order; one that cannot be resolved is skipped."""
    from gateway.control_socket import _private_runtime_dir
    try:
        temp = tempfile.gettempdir()
    except OSError:  # no usable temp directory at all (FileNotFoundError)
        temp = None
    return [str(root) for root in (_private_runtime_dir(), temp, _LAST_RESORT_ROOT) if root is not None]


def socket_path(home, execution_id):
    """The worker's private listener: in its 0700 outbox directory when that fits ``sun_path``
    (a short profile home), else ``hmw-<random>/w.sock`` under the first writable short root it
    fits in, measured in bytes (the random name is not predictable by another user). That directory
    is the worker's (``agent/managed_worker_detach/channel.py``): it creates it 0700 only while it waits for
    a new owner and removes it once reattached or exiting, so a turn that never changes owner leaves
    nothing behind. None when no root fits: the turn then
    runs without a detached reservation."""
    direct = outbox_dir(home, execution_id) / 'control.sock'
    if _fits(direct):
        direct.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return str(direct)
    name = Path('hmw-' + secrets.token_hex(8), 'w.sock')
    for root in _listener_roots():
        path = Path(root) / name
        if _fits(path) and os.access(root, os.W_OK | os.X_OK):
            return str(path)
    logger.warning('No directory is short enough for a detached worker listener; the turn runs undetached.')
    return None


def _worker_alive(record, *, numeric_uid=False):
    import psutil
    from hermes_cli.process_identity import worker_birth
    try:
        proc = psutil.Process(record['pid'])
        if worker_birth(proc) != record['birth'] or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if numeric_uid:
            # Do not invoke NSS (which may block on a remote lookup) in the retry loop.
            return proc.uids().real == os.getuid()  # windows-footgun: ok — POSIX only (detach_enabled)
        return proc.username() == psutil.Process().username()
    except psutil.Error:
        return False


def read_resume_slots(authority):
    """Read startup candidates without publishing routes, starting a drain or recording loss."""
    from gateway.session_authority import admission_row
    persistence = authority_persistence(authority)
    slots = {}
    for record in persistence.worker_executions.detached_reservations(epoch=authority.epoch):
        admission = persistence.admissions.get(admission_id=record['admission_id'])
        if (admission is None or admission.status != 'started'
                or admission.owner_epoch != authority.epoch or admission.generation != record['generation']):
            continue
        row = admission_row(admission)
        slots[row['target_session_id']] = (row, record)
    return slots


def start_reserved_resume_slots(authority):
    """Transfer each reservation to its drain in the current event-loop turn."""
    from gateway.session_authority import _log_drain_failure
    from gateway.session_runtime_workers import persist_work_count
    if not authority.detached_resume_slots:
        return
    for sid, (row, record) in tuple(authority.detached_resume_slots.items()):
        live = authority.sessions.get(sid)
        if live is None or (live.task is not None and not live.task.done()):
            continue
        ref = SessionRef(authority.profile_id, sid)
        live.task = asyncio.create_task(_resume(authority, ref, live, row, record))
        del authority.detached_resume_slots[sid]
        live.task.add_done_callback(_log_drain_failure)
        live.task.add_done_callback(lambda task: persist_work_count(authority, task))


def kill_recorded(record):
    """SIGKILL the recorded worker's process group, only while its pid still carries its birth."""
    import signal
    if _worker_alive(record):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(record['pid'], signal.SIGKILL)  # windows-footgun: ok — POSIX only (detach_enabled)


async def _resume(authority, ref, live, row, record):
    authority._stamp_claim(ref, live, row)
    if await authority._run_claimed(ref, live, row, resume=record):
        await authority._drain(ref)


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


def _wait_for_worker_retry(record, deadline):
    if time.monotonic() >= deadline or not _worker_alive(record, numeric_uid=True):
        return False
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    time.sleep(min(0.05, remaining))
    return time.monotonic() < deadline


def _connect_spared_worker(record, deadline):
    """Wait only for the recorded live worker's listener, before any handshake."""
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('detached_worker_listener_timeout')
            sock.settimeout(remaining)
            sock.connect(record['socket'])
            return sock
        except BaseException as exc:
            sock.close()
            if not isinstance(exc, OSError) or exc.errno not in (errno.ENOENT, errno.ECONNREFUSED):
                raise
            if not _wait_for_worker_retry(record, deadline):
                raise


def _set_handshake_timeout(sock, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('detached_worker_handshake_timeout')
    sock.settimeout(remaining)


class _HandshakeReader(io.RawIOBase):
    """Use the same deadline on every read, without buffering the next frame."""

    def __init__(self, sock, deadline):
        self._socket = sock
        self._deadline = deadline

    def readable(self):
        return True

    def readinto(self, buffer):
        _set_handshake_timeout(self._socket, self._deadline)
        data = self._socket.recv(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def _handshake(record):
    """Connect and authenticate the recorded worker, without confirming ownership."""
    deadline = time.monotonic() + HANDSHAKE_SECONDS
    expected = {k: record[k] for k in ('pid', 'birth', 'execution_id', 'generation')}
    while True:
        sock = _connect_spared_worker(record, deadline)
        try:
            pid, uid, _gid = _PEERCRED.unpack(sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size))
            if pid != record['pid'] or uid != os.getuid():  # windows-footgun: ok — POSIX only (detach_enabled)
                raise AuthorityRefusal('permission_denied')
            try:
                # A retiring listener can drop a queued connection. Only retry if
                # it closed before even the first byte, not after a partial hello.
                _set_handshake_timeout(sock, deadline)
                if not sock.recv(1, socket.MSG_PEEK):
                    raise EOFError('managed_worker_pipe_closed')
            except (EOFError, ConnectionResetError):
                sock.close()
                if not _wait_for_worker_retry(record, deadline):
                    raise
                continue
            with _HandshakeReader(sock, deadline) as reader:
                hello = read_frame(reader)
            if hello != {'type': 'hello', **expected}:
                raise AuthorityRefusal('permission_denied')
            return sock, deadline
        except BaseException:
            sock.close()
            raise


async def _close_unclaimed_handshake(task):
    # Cancellation of the consumer cannot cancel the blocking read. Join it before
    # releasing ownership, so the worker is not left talking to an abandoned socket.
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:  # health: allow BLE001 -- preserve the caller's cancellation
            break
    if not task.cancelled() and task.exception() is None:
        sock, _ = task.result()
        sock.close()


async def reattach(authority, ref, row, record, worker_class):
    """The ``ManagedWorker`` for a spared turn, registered for Stops like a spawned one."""
    connection = asyncio.create_task(asyncio.to_thread(_handshake, record))
    try:
        try:
            sock, deadline = await asyncio.shield(connection)
        except asyncio.CancelledError:
            await _close_unclaimed_handshake(connection)
            raise
        try:
            sid = ref.session_id
            if authority._managed_reattaching[sid] and authority.pending_stops.get(sid) != row['generation']:
                sock.close()
                return None
            if time.monotonic() >= deadline:
                raise TimeoutError('detached_worker_handshake_timeout')
            # Confirm on the owner loop, with no suspension or blocking I/O between the
            # retirement check and publication. This is the first send on the fresh socket.
            sock.setblocking(False)
            sock.sendall(encode_frame({'type': 'reattached', 'epoch': authority.epoch}))
            sock.setblocking(True)
        except BaseException:
            sock.close()
            raise
    except (OSError, ValueError, EOFError) as exc:
        raise AuthorityRefusal('managed_worker_lost') from exc
    worker = worker_class(AdoptedProcess(sock, record))
    worker.worker, worker.detachable = (record['pid'], record['birth']), True
    workers = getattr(authority, '_managed_workers', None)
    if workers is None:
        workers = authority._managed_workers = {}
    workers[ref.session_id] = worker
    logger.info('Re-adopted managed worker %s pid %s (epoch %s)', record['execution_id'], record['pid'],
                authority.epoch)
    return worker
