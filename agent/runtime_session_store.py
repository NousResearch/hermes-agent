"""Explicit worker persistence facade; never opens the canonical SQLite store.

This first operation family is deliberately NOT a complete SessionDB substitute.
No existing cron/child/compute consumer is switched until its inventory is covered.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading


_worker_process = False


def is_worker_process():
    """A successful self-registration makes this interpreter compute-only.

    Process-scoped (not ContextVar): import-time recovery and fresh helper threads
    must not regain owner-ledger access. This is cooperative runtime ownership,
    not OS confinement or an authorization credential.
    """
    return _worker_process


class WorkerPersistenceError(RuntimeError):
    pass


class WorkerRPC:
    """Bounded authenticated calls to an existing owner; never starts a daemon.

    websockets.sync owns a dedicated frame receiver, independent of the caller.
    No owner-loop synchronous self-RPC or SQLite fallback is permitted.
    """
    def __init__(self, home):
        self.home = Path(home).resolve()
        self.lock = threading.Lock()

    def __call__(self, method, **params):
        from hermes_cli.gateway_client import _session_ticket
        from hermes_cli.gateway_runtime_discovery import query_identify
        from websockets.sync.client import connect
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise WorkerPersistenceError('synchronous_rpc_on_event_loop')
        with self.lock:
            # A served secondary has no socket; its multiplexer's descriptor names it.
            from hermes_cli.gateway_runtime import discover_gateway_endpoint
            discovery = discover_gateway_endpoint(self.home, timeout=5)
            if discovery.state != 'ready' or discovery.endpoint is None:
                raise WorkerPersistenceError('owner_unavailable')
            from hermes_cli.gateway_runtime import control_home_for
            descriptor = query_identify(control_home_for(self.home, discovery.endpoint), timeout=5)
            if descriptor.get('pid') == os.getpid():
                raise WorkerPersistenceError('synchronous_self_rpc')
            endpoint = discovery.endpoint
            ticket = _session_ticket(self.home, endpoint,
                purpose='interactive' if method == 'worker.register' else 'worker-adoption')
            url = endpoint.api_origin.replace('https:', 'wss:').replace('http:', 'ws:') + '/api/ws'
            with connect(url, subprotocols=['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket],
                         open_timeout=5, close_timeout=1, max_size=8 * 1024 * 1024) as ws:
                ws.send(json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}))
                import time
                deadline = time.monotonic() + 20
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('worker_receipt_timeout')
                    response = json.loads(ws.recv(timeout=remaining))
                    if response.get('id') == 1:
                        break
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
    def __init__(self, rpc, scope, outbox_dir, *, max_bytes=4 * 1024 * 1024):
        self.rpc = rpc
        self.scope = dict(scope)
        self.max_bytes = max_bytes
        self.lock = threading.RLock()
        self.failure = None
        self.path = Path(outbox_dir) / 'pending.json'
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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
                self.journal = json.loads(self.path.read_text(encoding="utf-8"))
                if self.journal['scope'] != self.scope:
                    raise WorkerPersistenceError('outbox_scope_mismatch')
            else:
                self.journal = {'scope': self.scope, 'next_sequence': 1, 'pending': []}
                self._save(self.journal)
            if is_worker_process():
                from tools.async_delegation_worker import bind_worker_delegation_store
                bind_worker_delegation_store(self)
        except Exception:
            self._outbox_owner.close()
            raise

    def _save(self, journal):
        if self._outbox_owner.closed:
            raise WorkerPersistenceError('outbox_closed')
        encoded = json.dumps(journal, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode()
        if len(encoded) > self.max_bytes:
            self.failure = 'outbox_full'
            raise WorkerPersistenceError(self.failure)
        fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            if os.name != 'nt':
                directory = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)

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
        for message, annotation in zip(messages, result['annotations'], strict=True):
            message.update(annotation)
        return result['count']

    def try_acquire_session_turn_lease(self, session_id, holder, *, ttl_seconds=300.0, patience_s=None):
        self._session(session_id)
        return self._apply('turn.acquire', {'holder': holder, 'ttl_seconds': ttl_seconds})['value']

    def acquire_session_turn_lease(self, session_id, holder, *, ttl_seconds=300.0,
            wait_seconds=1800.0, poll_interval_seconds=1.0, on_wait=None,
            wait_notice_interval_seconds=15.0, should_abort=None, acquire_patience_s=0.5):
        # Reuse only the local polling orchestrator, not the SQLite mixin surface.
        from hermes_state_compression import SessionCompressionMixin
        return SessionCompressionMixin.acquire_session_turn_lease(self, session_id, holder,
            ttl_seconds=ttl_seconds, wait_seconds=wait_seconds,
            poll_interval_seconds=poll_interval_seconds, on_wait=on_wait,
            wait_notice_interval_seconds=wait_notice_interval_seconds,
            should_abort=should_abort, acquire_patience_s=acquire_patience_s)

    def refresh_session_turn_lease(self, session_id, holder, *, ttl_seconds=300.0):
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

    def update_system_prompt(self, session_id, system_prompt):
        self._session(session_id)
        self._apply('session.prompt', {'system_prompt': system_prompt})

    def patch_session_model_config(self, session_id, patch):
        self._session(session_id)
        self._apply('session.sidecars', {'patch': patch})

    def update_session_tool_names(self, session_id, tool_names):
        # The tools[] freeze pin: without it every fresh worker re-probes check_fns and a
        # config flip between turns silently forks the cached prefix (in-process stays pinned).
        self._session(session_id)
        self._apply('session.tools', {'tool_names': None if tool_names is None else list(tool_names)})

    def finish(self):
        return self._apply('execution.finish', {})

    def close(self):
        with self.lock:
            try:
                self.flush_token_counts()
            finally:
                self._outbox_owner.close()
