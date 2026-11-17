"""Worker side of ``gateway.detached_workers``: a managed worker that outlives its owner.

Without ``detach_v1`` in the bootstrap a worker dies with its owner (``managed_worker._die_with_owner``).
With it, the owner's pipe closing (a restart, a crash) leaves the turn running: the worker stops
writing to the dead pipe, buffers its frames, and listens on the private per-execution unix socket
the owner named. The restarted owner, which kept the execution alive at its new epoch at boot
(``gateway/session_detached_workers.py``), connects, is told this worker's identity, answers with
its epoch, and receives the buffered frames; the turn then continues on the socket as it did on the
pipes. Nobody connecting within ``ADOPT_GRACE_SECONDS`` still ends the worker and its group.

Imported only after the bootstrap frame is read; nothing here runs at import.
"""
from collections import deque
import contextlib
import os
from pathlib import Path
import socket
import struct
import threading
import time

from agent.managed_worker import WorkerChannel, encode_frame, read_frame

# How long an orphaned worker waits for a new owner before it ends itself (the old guarantee,
# delayed). A gateway restart takes seconds; a host that never brings one back is not served.
ADOPT_GRACE_SECONDS = 900
# A store call that failed while the owner was still connected: how long it waits to learn the
# owner is gone (its pipe closes moments after its RPC endpoint) before failing as it always did.
OWNER_LOSS_SETTLE_SECONDS = 10
# Frames kept while no owner reads them. Deltas past this are dropped (the transcript holds the
# text); results, tool and prompt frames are always kept.
BUFFER_BYTES = 8 * 1024 * 1024
_DROPPABLE = frozenset({'delta'})
_PEERCRED = struct.Struct('3i')
# Store calls safe to resend: a worker.persist is keyed by (execution, sequence, digest) and an
# exact retry returns its stored receipt.
_RETRY_METHODS = frozenset({'worker.persist'})
# After a reattach, how long a refused store call keeps retrying while the new owner's RPC
# endpoint comes up.
REATTACH_SERVE_SECONDS = 120


def detach_spec(frame):
    """The bootstrap's ``detach_v1`` socket path, or None; ValueError on a malformed one."""
    spec = frame.get('detach_v1')
    if spec is None:
        return None
    if (not isinstance(spec, dict) or set(spec) != {'socket'} or not isinstance(spec['socket'], str)
            or not Path(spec['socket']).is_absolute() or os.name == 'nt'):
        raise ValueError('invalid_managed_worker_bootstrap')
    return spec['socket']


class DetachableChannel(WorkerChannel):
    """The worker's frame channel. Attached: the owner's pipe (later its socket). Orphaned: frames
    are buffered and a listener waits for the next owner."""

    def __init__(self, stream, path, identity):
        super().__init__(stream)
        self.path, self.identity = path, identity
        self.orphaned = False
        self.epoch = None  # the adopting owner's epoch once one reattached
        self.attachments = 0
        self.changed = threading.Condition(self.lock)
        self.buffer, self.buffered = deque(), 0
        self.inbound = self.conn = None
        self.timer = None
        self.result = None  # the sent result frame until the owner's finish acknowledges it
        self.prompts = {}  # open approval/clarify frames by prompt id: a new owner must see them

    def send(self, kind, **payload):
        data = encode_frame({'type': kind, **payload})
        with self.lock:
            if kind == 'result':
                self.result = data
            elif kind in ('approval', 'clarify'):
                self.prompts[payload['data']['request_id'] if kind == 'approval' else payload['prompt_id']] = data
            elif kind == 'prompt_settled':
                self.prompts.pop(payload['prompt_id'], None)
            if not self.orphaned:
                try:
                    self.stream.write(data)
                    self.stream.flush()
                    return
                except OSError:
                    self._orphan_locked()
            if kind == 'activity' or (kind in _DROPPABLE and self.buffered + len(data) > BUFFER_BYTES):
                return
            self.buffer.append(data)
            self.buffered += len(data)

    def orphan(self):
        with self.lock:
            if not self.orphaned:
                self._orphan_locked()

    def _orphan_locked(self):
        self.orphaned = True
        with contextlib.suppress(OSError):
            self.stream.close()
        if self.conn is not None:  # a reattached owner's socket: its reader must see EOF too
            with contextlib.suppress(OSError):
                self.conn.shutdown(socket.SHUT_RDWR)
            self.conn = None
        # The owner drained stdout/stderr; with it gone a print would raise EPIPE inside the turn.
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.dup2(null, 2)
        os.close(null)
        from agent.managed_worker import _die_with_owner
        self.timer = _die_with_owner(ADOPT_GRACE_SECONDS)
        from agent.memory_provider import spawn_context_thread
        spawn_context_thread(self._listen, name='managed-detached-listener').start()
        self.changed.notify_all()

    def _listen(self):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            old = os.umask(0o177)
            try:
                server.bind(self.path)
            finally:
                os.umask(old)
            server.listen(4)
            while True:
                conn, _ = server.accept()
                if not self._reattach(conn):
                    conn.close()
                    continue
                return
        finally:
            server.close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.path)

    def _reattach(self, conn):
        """Hand the turn to a verified same-user owner: hello, its epoch, then the buffered frames."""
        uid = _PEERCRED.unpack(conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size))[1]
        if uid != os.getuid():  # windows-footgun: ok — POSIX only (detach_spec refuses nt)
            return False
        conn.settimeout(30)
        reader, writer = conn.makefile('rb'), conn.makefile('wb')
        try:
            writer.write(encode_frame({'type': 'hello', **self.identity}))
            writer.flush()
            answer = read_frame(reader)
            if set(answer) != {'type', 'epoch'} or answer['type'] != 'reattached' or type(answer['epoch']) is not int:
                return False
            conn.settimeout(None)
            with self.lock:
                # Written to the dead owner's pipe and never answered: the new owner gets them again.
                for data in [*self.prompts.values(), *([self.result] if self.result is not None else [])]:
                    if data not in self.buffer:
                        writer.write(data)
                while self.buffer:
                    writer.write(self.buffer.popleft())
                writer.flush()
                self.buffered = 0
                self.stream, self.inbound, self.conn = writer, reader, conn
                self.epoch, self.orphaned = answer['epoch'], False
                self.attachments += 1
                self.timer.cancel()
                self.changed.notify_all()
            return True
        except (OSError, ValueError, EOFError):
            return False

    def await_owner(self):
        """Controls reader on owner EOF: the next owner's control stream, or None at the grace
        (the grace timer is already ending this process)."""
        with self.lock:
            if not self.orphaned:
                self._orphan_locked()
            self.changed.wait_for(lambda: not self.orphaned, timeout=ADOPT_GRACE_SECONDS)
            return None if self.orphaned else self.inbound

    def drain_before_exit(self):
        """The turn ended while orphaned: its last frames (``finished``) are buffered. Wait for the
        next owner to take them; the grace timer ends this process if none comes."""
        with self.lock:
            self.changed.wait_for(lambda: not self.orphaned or not self.buffer, timeout=ADOPT_GRACE_SECONDS)

    def hold_result(self, finished):
        """Wait for the owner's ``finish``; across an owner loss, for the next owner's (each owner
        that attached gets the usual 30 seconds after it received the result)."""
        seen = self.attachments
        while not finished.wait(30):
            with self.lock:
                if not self.orphaned and self.attachments == seen:
                    return False
                seen = self.attachments
        return True

    def await_reattach(self, seen):
        """A store call failed: True once a new owner attached after attachment ``seen``. The owner's
        loss must show within ``OWNER_LOSS_SETTLE_SECONDS`` or the failure is the call's own."""
        with self.lock:
            if not self.changed.wait_for(lambda: self.orphaned or self.attachments > seen,
                                         timeout=OWNER_LOSS_SETTLE_SECONDS):
                return False
            return self.changed.wait_for(lambda: self.attachments > seen, timeout=ADOPT_GRACE_SECONDS)


class DetachedRPC:
    """The store's owner RPC across an owner restart: a ``worker.persist`` that failed because the
    owner went away is resent, once a new owner reattached, under that owner's epoch (the boot
    fence moved the execution to it). Every other call and failure is the plain ``WorkerRPC``'s."""

    def __init__(self, rpc, channel):
        self.rpc, self.channel = rpc, channel

    def __getattr__(self, name):
        return getattr(self.rpc, name)

    def __call__(self, method, **params):
        deadline = None
        while True:
            seen = self.channel.attachments
            if method == 'worker.persist' and self.channel.epoch is not None:
                params = {**params, 'epoch': self.channel.epoch}
            try:
                return self.rpc(method, **params)
            except Exception:  # health: allow BLE001 -- re-raised unless the owner provably restarted
                if method not in _RETRY_METHODS:
                    raise
                if deadline is not None and time.monotonic() < deadline:
                    time.sleep(1)  # a just-reattached owner may adopt before its RPC endpoint serves
                    continue
                if not self.channel.await_reattach(seen):
                    raise
                deadline = time.monotonic() + REATTACH_SERVE_SECONDS


def identity(scope):
    """What the worker tells a connecting owner, checked against the owner's durable record."""
    return {key: scope[key] for key in ('pid', 'birth', 'execution_id', 'generation')}
