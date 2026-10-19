"""Explicit worker persistence facade; never opens the canonical SQLite store.

This first operation family is deliberately NOT a complete SessionDB substitute.
No existing cron/child/compute consumer is switched until its inventory is covered.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
from hermes_dir_acl import make_user_dir

logger = logging.getLogger(__name__)

_worker_process = False


def is_worker_process():
    """A successful self-registration makes this interpreter compute-only.

    Process-scoped (not ContextVar): import-time recovery and fresh helper threads
    must not regain owner-ledger access. This is cooperative runtime ownership,
    not OS confinement or an authorization credential.
    """
    return _worker_process



def apply_row_annotations(messages, annotations):
    """Mirror the owner's in-place row stamps onto the worker's dicts (None = the owner popped it)."""
    for message, annotation in zip(messages, annotations, strict=True):
        for key, value in annotation.items():
            if value is None:
                message.pop(key, None)
            else:
                message[key] = value

class WorkerPersistenceError(RuntimeError):
    pass


def _caller_summary(limit=8):
    """module:function:line of the frames above this store (its own frames skipped), never
    locals or argument values: best-effort callers swallow the guard's raise (#139029)."""
    frame, frames = sys._getframe(2), []
    while frame is not None and len(frames) < limit:
        module = frame.f_globals.get('__name__', '?')
        if not module.startswith('agent.runtime_session_'):
            frames.append(f'{module}:{frame.f_code.co_name}:{frame.f_lineno}')
        frame = frame.f_back
    return ' <- '.join(frames)


def call_off_event_loop(fn, *args, **kwargs):
    """Call ``fn`` here, or on a worker thread when this thread is running an event loop.

    For synchronous best-effort writers that can run inside a callback Relay drives on its private
    loop (provider/tool callbacks): there ``WorkerRPC`` refuses the call (``synchronous_rpc_on_event_loop``,
    sticky for the rest of the turn; #139029). The callback already blocks that loop, so waiting for the
    thread costs it nothing; the context is copied so session-scoped state survives the hop, and errors
    re-raise here. Async code awaits ``asyncio.to_thread`` instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn(*args, **kwargs)
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='worker-persist-off-loop') as pool:
        return pool.submit(context.run, fn, *args, **kwargs).result()


class _OwnerLink:
    """One owner WebSocket and the only thread that reads it. Events the owner broadcasts to every
    transport are discarded as they arrive, idle or not: left queued, websockets' bounded receive
    buffer pauses socket reads, which also stalls pongs and the close handshake. The reply to the
    one outstanding request id goes to its caller; a closed connection goes to it as the exception."""
    def __init__(self, ws):
        self.ws = ws
        self.awaiting = None
        self.replies = queue.SimpleQueue()
        self.reader = threading.Thread(target=self._read, name='worker-rpc-reader', daemon=True)
        self.reader.start()

    def _read(self):
        try:
            while True:
                try:
                    frame = json.loads(self.ws.recv())
                except ValueError:
                    continue
                if isinstance(frame, dict) and self.awaiting is not None and frame.get('id') == self.awaiting:
                    self.replies.put(frame)
        except Exception as exc:  # health: allow BLE001 -- handed to the caller, which raises it (ConnectionClosed included)
            self.replies.put(exc)

    def send(self, request_id, message):
        self.awaiting = request_id
        self.ws.send(message)

    def receive(self, timeout):
        try:
            reply = self.replies.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError('worker_receipt_timeout') from None
        if isinstance(reply, BaseException):
            raise reply
        return reply


class WorkerRPC:
    """Bounded authenticated calls to an existing owner; never starts a daemon.

    No owner-loop synchronous self-RPC or SQLite fallback is permitted.
    One connection serves every call (serialized by ``lock``, correlated by id); it is
    discovered, identified and ticketed on connect exactly as each call once was.
    """
    def __init__(self, home):
        self.home = Path(home).resolve()
        self.lock = threading.Lock()
        self.endpoint = None
        self.link = None
        self.request_id = 0

    def _connection(self):
        if self.link is not None:
            if self.link.reader.is_alive():
                return self.link
            # The owner closed it while idle, before any send: redial with a fresh ticket.
            self._drop()
        from hermes_cli.gateway_client import _session_ticket, gateway_ws_target
        from hermes_cli.gateway_runtime_discovery import query_identify
        from websockets.sync.client import connect
        # A served secondary has no socket; its multiplexer's descriptor names it.
        from hermes_cli.gateway_runtime import discover_gateway_endpoint
        discovery = discover_gateway_endpoint(self.home, timeout=5)
        if discovery.state == 'ready' and discovery.endpoint is not None:
            self.endpoint = discovery.endpoint
        elif discovery.state != 'draining' or self.endpoint is None:
            raise WorkerPersistenceError('owner_unavailable')
        from hermes_cli.gateway_runtime import control_home_for
        endpoint = self.endpoint
        descriptor = query_identify(control_home_for(self.home, endpoint), timeout=5)
        if descriptor.get('pid') == os.getpid():
            raise WorkerPersistenceError('synchronous_self_rpc')
        ticket = _session_ticket(self.home, endpoint, purpose='worker-adoption')
        url, protocols = gateway_ws_target(endpoint, ticket)
        # The owner is loopback, like connect_gateway's peer: an inherited HTTPS_PROXY must not carry it.
        # No library keepalive: its pong deadline would run independently of (and could fail) a call
        # inside its receipt budget; that budget and the observed close are the liveness checks.
        self.link = _OwnerLink(connect(url, subprotocols=protocols, open_timeout=5, close_timeout=1,
                                       max_size=8 * 1024 * 1024, proxy=None, ping_interval=None))
        return self.link

    def _drop(self):
        link, self.link = self.link, None
        if link is not None:
            with contextlib.suppress(OSError):  # a broken socket is already gone
                link.ws.close()

    def close(self):
        with self.lock:
            self._drop()

    def __call__(self, method, **params):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            logger.warning('synchronous_rpc_on_event_loop: %s on thread %s; caller: %s',
                           method, threading.current_thread().name, _caller_summary())
            raise WorkerPersistenceError('synchronous_rpc_on_event_loop')
        from websockets.exceptions import ConnectionClosed
        with self.lock:
            self.request_id += 1
            request_id = self.request_id
            message = json.dumps({'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params})
            try:
                link = self._connection()
                try:
                    link.send(request_id, message)
                except ConnectionClosed:
                    # websockets raises this from send only before a complete frame is written (closed
                    # on entry, or the socket write failed): the owner closed the link after the idle
                    # check and never saw this request, so redial and send it once more.
                    self._drop()
                    link = self._connection()
                    link.send(request_id, message)
                response = link.receive(20)
            except BaseException:
                # Outcome unknown once sent: surface the error and never resend; the next call redials.
                self._drop()
                raise
            if 'error' in response:
                raise WorkerPersistenceError(response['error'].get('message', 'persistence_failed'))
            if method in ('worker.register', 'worker.adopt') and params.get('pid') == os.getpid():
                global _worker_process
                _worker_process = True
            return response['result']


from agent.runtime_session_compression import RuntimeSessionCompressionMixin


from agent.runtime_session_lifecycle import RuntimeSessionLifecycleMixin


class RuntimeSessionStore(RuntimeSessionCompressionMixin, RuntimeSessionLifecycleMixin):
    """Synchronous durable results with a private byte-bounded retry journal.

    A failed write remains pending, and failure is sticky until explicit retry or
    adoption. Local queue acceptance is never returned as canonical commit.
    Pending journals live outside age-pruned cache trees.
    """
    def __init__(self, rpc, scope, outbox_dir, adopted=None, *, max_bytes=4 * 1024 * 1024):
        self.rpc = rpc
        self.scope = dict(scope)
        self.max_bytes = max_bytes
        self.lock = threading.RLock()
        self.failure = None
        self.path = Path(outbox_dir) / 'pending.json'
        make_user_dir(self.path.parent, mode=0o700, parents=True, exist_ok=True)
        if self.path.parent.is_symlink() or self.path.is_symlink():
            raise WorkerPersistenceError('unsafe_outbox')
        os.chmod(self.path.parent, 0o700)
        from gateway.status import _try_acquire_file_lock
        lock_path = self.path.parent / 'owner.lock'
        if lock_path.is_symlink():
            raise WorkerPersistenceError('unsafe_outbox')
        self._outbox_owner = lock_path.open('a+', encoding='utf-8')
        os.chmod(lock_path, 0o600)
        if not _try_acquire_file_lock(self._outbox_owner):
            self._outbox_owner.close()
            raise WorkerPersistenceError('outbox_in_use')
        try:
            if self.path.exists():
                if self.path.stat().st_size > max_bytes:
                    raise WorkerPersistenceError('outbox_full')
                self.journal = json.loads(self.path.read_text(encoding="utf-8-sig"))
                reconciled = self.journal['scope'] != self.scope
                if reconciled:
                    self._reconcile_adopted(adopted)
            else:
                reconciled = False
                self.journal = {'scope': self.scope, 'next_sequence': 1, 'pending': []}
                self._save(self.journal)
            if self.journal['pending']:
                self.retry_pending()
            if reconciled and self.scope != dict(scope):
                # Replay must land the journal exactly on the adopted assignment (a pending
                # publication moves it to the child); anything else is not this successor's outbox.
                raise WorkerPersistenceError('outbox_scope_mismatch')
            if is_worker_process():
                from tools.async_delegation_worker import bind_worker_delegation_store
                bind_worker_delegation_store(self)
        except Exception:
            self._outbox_owner.close()
            raise

    def _reconcile_adopted(self, adopted):
        """Carry a retained outbox across verified owner replacement.

        ``adopted`` is the separately authenticated ``worker.adopt`` result for the requested
        scope; never self-adopt. Only the owner epoch (forward) and the physical target may
        differ from the journal (a same-epoch restart can still find the target rotated).
        Pending entries replay under the journal's own target with the adopted epoch, so an
        exact lost-ACK retry resolves its original receipt instead of creating a second one;
        the constructor then requires the replayed scope to equal the adopted one.
        """
        old, new = self.journal['scope'], self.scope
        fixed = set(new) - {'epoch', 'session_id'}
        if (not isinstance(adopted, dict) or set(old) != set(new)
                or any(old[key] != new[key] for key in fixed)
                or adopted.get('execution_id') != new.get('execution_id')
                or adopted.get('generation') != new.get('generation')
                or adopted.get('session_id') != new.get('session_id')
                or adopted.get('owner_epoch') != new.get('epoch') or adopted.get('status') != 'running'
                or type(old.get('epoch')) is not int or type(new.get('epoch')) is not int
                or old['epoch'] > new['epoch']
                # Only a pending publication can still move the target; replay must prove it.
                or (old['session_id'] != new['session_id'] and not any(
                    entry['operation'] == 'compression.publish' for entry in self.journal['pending']))):
            raise WorkerPersistenceError('outbox_scope_mismatch')
        candidate = dict(self.journal, scope=dict(old, epoch=new['epoch']))
        self._save(candidate)
        self.scope = candidate['scope']
        self.journal = candidate

    def _save(self, journal):
        if self._outbox_owner.closed:
            raise WorkerPersistenceError('outbox_closed')
        encoded = json.dumps(journal, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode()
        if len(encoded) > self.max_bytes:
            self.failure = 'outbox_full'
            raise WorkerPersistenceError(self.failure)
        from utils import atomic_write_bytes
        # Owner-only journal (the dir is 0700 too); the rename and its directory entry are durable.
        atomic_write_bytes(self.path, encoded, tmp_prefix='.pending-', mode=0o600, fsync_dir=True)

    def _session(self, session_id):
        if session_id != self.scope['session_id']:
            raise WorkerPersistenceError('permission_denied')

    def _apply(self, operation, payload):
        with self.lock:
            if self.failure:
                raise WorkerPersistenceError(self.failure)
            if self.journal['pending']:
                raise WorkerPersistenceError('pending_receipt')
            entry = {'sequence': self.journal['next_sequence'], 'operation': operation, 'payload': payload}
            candidate = json.loads(json.dumps(self.journal))
            candidate['pending'].append(json.loads(json.dumps(entry, allow_nan=False)))
            candidate['next_sequence'] += 1
            try:
                self._save(candidate)
            except Exception as exc:
                self.failure = str(exc)
                raise
            self.journal = candidate
            return self.retry_pending()[0]

    def retry_pending(self):
        with self.lock:
            results = []
            try:
                while self.journal['pending']:
                    entry = self.journal['pending'][0]
                    result = self.rpc('worker.persist', **self.scope, **entry)
                    candidate = dict(self.journal, pending=self.journal['pending'][1:])
                    candidate = self._compression_receipt_journal(candidate, result)
                    self._save(candidate)
                    self.scope = candidate['scope']
                    self.journal = candidate
                    results.append(result)
            except Exception as exc:
                self.failure = str(exc)
                raise
            self.failure = None
            return results

    def adopt(self, epoch):
        """Install only a separately verified worker.adopt result; never self-adopt."""
        with self.lock:
            candidate = dict(self.journal, scope=dict(self.scope, epoch=epoch))
            self._save(candidate)
            self.scope = candidate['scope']
            self.journal = candidate
            self.failure = None

    def append_messages_batch(self, session_id, messages, compression_lock_holder=None,
                              turn_lease_holder=None, chunk_rows=None, turn_lease_ttl_seconds=300.0):
        self._session(session_id)
        if chunk_rows is not None:
            raise WorkerPersistenceError('unsupported_operation')
        if compression_lock_holder is not None or turn_lease_ttl_seconds != 300.0:
            return self._append_compression_messages(session_id, messages, compression_lock_holder,
                turn_lease_holder, turn_lease_ttl_seconds)
        result = self._apply('transcript.append', {'messages': messages, 'turn_lease_holder': turn_lease_holder})
        apply_row_annotations(messages, result['annotations'])
        return result['count']

    def try_acquire_session_turn_lease(self, session_id, holder, *, ttl_seconds=300.0, patience_s=None):
        self._session(session_id)
        return self._apply('turn.acquire', {'holder': holder, 'ttl_seconds': ttl_seconds})['value']

    def acquire_session_turn_lease(self, session_id, holder, *, ttl_seconds=300.0,
            wait_seconds=1800.0, poll_interval_seconds=1.0, on_wait=None,
            wait_notice_interval_seconds=15.0, should_abort=None, acquire_patience_s=0.5,
            on_contended=None):
        # Reuse only the local polling orchestrator, not the SQLite mixin surface. Same signature
        # as the owner's (``on_contended``: a busy-database notice; the RPC try never raises it).
        from hermes_state_compression import SessionCompressionMixin
        return SessionCompressionMixin.acquire_session_turn_lease(self, session_id, holder,
            ttl_seconds=ttl_seconds, wait_seconds=wait_seconds,
            poll_interval_seconds=poll_interval_seconds, on_wait=on_wait,
            wait_notice_interval_seconds=wait_notice_interval_seconds,
            should_abort=should_abort, acquire_patience_s=acquire_patience_s,
            on_contended=on_contended)

    def refresh_session_turn_lease(self, session_id, holder, *, ttl_seconds=300.0, patience_s=None):
        self._session(session_id)
        return self._apply('turn.renew', {'holder': holder, 'ttl_seconds': ttl_seconds})['value']

    def release_session_turn_lease(self, session_id, holder):
        # DurableTurnLease retains its admission target across physical rotation.
        self._apply('turn.cleanup', {'target': session_id, 'holder': holder})

    def queue_token_counts(self, session_id, **usage):
        self._session(session_id)
        self._apply('usage.main', usage)

    def record_auxiliary_usage(self, session_id, task, **usage):
        self._session(session_id)
        self._apply('usage.auxiliary', dict(usage, task=task))

    def flush_token_counts(self, timeout=5.0):
        with self.lock:
            if self.failure:
                raise WorkerPersistenceError(self.failure)
            if self.journal['pending']:
                raise WorkerPersistenceError('pending_receipt')
            return True

    def get_session_title(self, session_id):
        return self.get_session(session_id).get('title')

    # Transcript-tail reads the agent core makes on its session_db (failed-turn boundary, in-place
    # compaction coverage); a missing one raised AttributeError inside a best-effort guard.
    def latest_conversation_role(self, session_id):
        self._session(session_id)
        return self._apply('session.tail_role', {})['value']

    def get_message_role(self, session_id, row_id):
        self._session(session_id)
        return self._apply('session.row_role', {'row_id': int(row_id)})['value']

    def get_compression_failure_cooldown_row(self, session_id):
        row = self.get_session(session_id)
        return {'session_exists': True, 'cooldown_until': row.get('compression_failure_cooldown_until'),
                'error': row.get('compression_failure_error')}

    def get_compression_failure_cooldown(self, session_id):
        import time
        now = time.time()
        row = self.get_compression_failure_cooldown_row(session_id)
        deadline = row['cooldown_until']
        if deadline is None or float(deadline) <= now:
            return None
        return {'cooldown_until': float(deadline), 'remaining_seconds': float(deadline) - now,
                'error': row['error']}

    def _session_number(self, session_id, column, cast, zero):
        row = self.get_session(session_id)
        try:
            return max(zero, cast(row.get(column) or zero))
        except (TypeError, ValueError):
            return zero

    def get_compression_fallback_streak(self, session_id):
        return self._session_number(session_id, 'compression_fallback_streak', int, 0)

    def get_compression_ineffective_count(self, session_id):
        return self._session_number(session_id, 'compression_ineffective_count', int, 0)

    def get_compression_recovery_deadline(self, session_id):
        return self._session_number(session_id, 'compression_recovery_deadline', float, 0.0)

    def get_session_model_config_value(self, session_id, key, default=None):
        from hermes_state_sessions import _parse_model_config
        return _parse_model_config(self.get_session(session_id).get('model_config')).get(key, default)

    def update_conversation_model_config_value(self, session_id, key, update):
        # A worker owns exactly its session row, so the read-modify-write is that row's; the owner applies
        # the patch in its own transaction (``None`` removes the key, as on SessionDB).
        self._session(session_id)
        self.patch_session_model_config(session_id, {key: update(self.get_session_model_config_value(session_id, key))})
        return session_id

    def session_ids_with_model_config_key(self, key):
        # The scan a worker can answer: its own session (it may read no other row).
        sid = self.scope['session_id']
        return [sid] if self.get_session_model_config_value(sid, key) is not None else []

    def update_system_prompt(self, session_id, system_prompt):
        self._session(session_id)
        self._apply('session.prompt', {'system_prompt': system_prompt})

    def patch_session_model_config(self, session_id, patch):
        self._session(session_id)
        self._apply('session.sidecars', {'patch': patch})

    def update_session_tool_names(self, session_id, pin):
        # The tools[] freeze pin ({"version", "tools"}; None clears): without it every fresh worker
        # re-probes check_fns and a config flip between turns silently forks the cached prefix.
        self._session(session_id)
        self._apply('session.tools', {'tool_names': pin})

    def finish(self):
        return self._apply('execution.finish', {})

    def close(self):
        with self.lock:
            try:
                self.flush_token_counts()
            finally:
                self._outbox_owner.close()
