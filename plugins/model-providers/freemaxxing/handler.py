"""Authenticated loopback API; exhausted routes are typed errors, not fake success."""
from __future__ import annotations

import hmac
import json
import socket
import time
from http.server import BaseHTTPRequestHandler

from .policy import (AuthError, Budget, ClientRequestError, ModelNotFoundError,
                     RateLimitError, TransientError, _MAX_REQUEST_BODY_BYTES,
                     )
from .pool import pool
from .upstream import _forward, _open_stream


class ChatCompletionsHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_args):
        pass

    @property
    def owner(self):
        return getattr(self.server, 'backend_pool', pool)

    def _send_json(self, code, body, retry_after=None):
        raw = json.dumps(body, allow_nan=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Connection', 'close')
        if retry_after is not None:
            self.send_header('Retry-After', str(max(1, int(retry_after))))
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(raw)
        except OSError:
            pass

    def _send_error(self, code, message, error_type='freemaxxing_error', retry_after=None):
        self._send_json(code, {'error': {'message': message, 'type': error_type,
                                       'retryable': code in {429, 503},
                                       'router_state_mutated': False}}, retry_after)

    def _require_auth(self):
        expected = str(getattr(self.server, 'auth_token', '') or '')
        supplied = self.headers.get('Authorization', '')
        # Bytes keep malformed non-ASCII header values from raising TypeError.
        valid = expected and hmac.compare_digest(supplied.encode('utf-8'),
                                                  f'Bearer {expected}'.encode('utf-8'))
        if valid:
            return True
        self._send_error(401, 'Unauthorized', 'unauthorized')
        return False

    def _require_runtime(self):
        try:
            guard = getattr(self.server, 'runtime_guard', None)
            if guard is not None:
                guard()
            with self.server.pool_init_lock:
                if not self.server.pool_ready:
                    self.server.pool_initializer()
                    self.server.pool_ready = True
            return True
        except RuntimeError:
            self._send_error(409, 'Freemaxxing requires a single-profile runtime.', 'profile_isolation')
        except Exception:
            self._send_error(503, 'Freemaxxing initialization failed; session is unchanged.',
                             'runtime_unavailable', 5)
        return False

    def do_GET(self):  # noqa: N802
        path = self.path.split('?', 1)[0]
        if path == '/v1/healthz':
            self._send_json(200, {'service': 'freemaxxing', 'status': 'ok'})
        elif self._require_auth():
            if path == '/v1/models':
                self._send_json(200, {'object': 'list', 'data': self.owner.get_aggregated_models()})
            elif path == '/healthz':
                self._send_json(200, {'service': 'freemaxxing', 'health': self.owner.health()})
            else:
                self._send_error(404, 'Unknown endpoint')

    def _read_body(self):
        if self.headers.get('Transfer-Encoding'):
            raise ClientRequestError('Transfer-Encoding is not supported')
        lengths = self.headers.get_all('Content-Length', [])
        if len(lengths) != 1:
            raise ClientRequestError('exactly one Content-Length is required')
        try:
            length = int(lengths[0])
        except ValueError as exc:
            raise ClientRequestError('invalid Content-Length') from exc
        if not 0 < length <= _MAX_REQUEST_BODY_BYTES:
            self._send_error(413 if length > 0 else 400, 'request body outside allowed bounds')
            return None
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ClientRequestError('request body interrupted')
        def invalid(_value):
            raise ValueError('non-finite number')
        try:
            body = json.loads(raw, parse_constant=invalid)
        except (ValueError, UnicodeError) as exc:
            raise ClientRequestError('invalid JSON') from exc
        if (not isinstance(body, dict) or not isinstance(body.get('messages'), list) or
                not body['messages'] or any(not isinstance(m, dict) for m in body['messages'])):
            raise ClientRequestError('messages must be a nonempty array of objects')
        if 'stream' in body and not isinstance(body['stream'], bool):
            raise ClientRequestError('stream must be boolean')
        if body.get('n', 1) != 1 or not isinstance(body.get('model', 'freemaxxing'), str):
            raise ClientRequestError('one completion and a string model are required')
        if any(k in body for k in ('models', 'plugins', 'transforms', 'web_search_options')):
            raise ClientRequestError('upstream fallbacks and paid extensions are not allowed')
        if 'tools' in body and not isinstance(body['tools'], list):
            raise ClientRequestError('tools must be an array')
        return body

    def do_POST(self):  # noqa: N802
        if self.path.split('?', 1)[0] != '/v1/chat/completions':
            self._send_error(404, 'Unknown endpoint')
            return
        if not self._require_auth():
            return
        try:
            body = self._read_body()
        except (ClientRequestError, socket.timeout, OSError) as exc:
            self._send_error(400, str(exc) if isinstance(exc, ClientRequestError) else 'request read timeout',
                             'invalid_request')
            return
        if body is None or not self._require_runtime():
            return
        # Acquire before spawning inference work; no unbounded waiting queue.
        if not self.server.inference_slots.acquire(False):
            self._send_error(503, 'Freemaxxing is at its concurrency limit.', 'capacity', 1)
            return
        try:
            self._complete(body)
        except ClientRequestError as exc:
            self._send_error(400, str(exc), 'invalid_request')
        except Exception:
            self._send_error(503, 'Freemaxxing could not complete this attempt; session is unchanged.',
                             'runtime_unavailable', 5)
        finally:
            self.server.inference_slots.release()

    def _complete(self, body):
        owner, tried = self.owner, set()
        budget = Budget(owner.limits.total)
        owner.refresh_catalogs()
        session = body.pop('freemaxxing_session', None)
        if session is not None and (not isinstance(session, str) or len(session) > 256):
            raise ClientRequestError('invalid session affinity')
        catalog_deadline = time.monotonic() + owner.limits.catalog
        while len(tried) < owner.limits.attempts:
            try:
                budget.remaining()
            except TransientError:
                break
            candidates = [c for c in owner.candidates(body, session) if c.identity not in tried]
            if not candidates:
                remaining = catalog_deadline - time.monotonic()
                if remaining > 0 and owner.catalog_pending():
                    owner.wait_for_catalog(budget, remaining)
                    continue
                break
            candidate = candidates[0]
            backend = candidate.backend
            tried.add(candidate.identity)
            outgoing = dict(body, model=candidate.model)
            start = time.monotonic()
            try:
                if body.get('stream'):
                    result = _open_stream(backend, outgoing, budget, owner=owner)
                else:
                    result = _forward(backend, outgoing, budget, owner=owner)
            except RateLimitError as exc:
                backend.record_failure(exc.retry_after, 'rate_limit')
                continue
            except AuthError:
                backend.record_failure(60, 'auth')
                continue
            except ModelNotFoundError:
                backend.record_failure(300, 'model_unavailable', model=candidate.model)
                continue
            except TransientError:
                backend.record_failure(10, 'transient', model=candidate.model)
                continue
            backend.record_success(model=candidate.model, elapsed=time.monotonic() - start)
            owner.remember(session, candidate.identity)
            if not body.get('stream'):
                self._send_json(200, result)
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Content-Length', str(len(result)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                try:
                    self.wfile.write(result)
                    self.wfile.flush()
                except OSError:
                    pass  # Downstream cancellation is not upstream ill-health.
            return
        self._send_error(503, owner.exhaustion_detail(), 'free_routes_exhausted', owner.retry_after())
