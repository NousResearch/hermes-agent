"""Bounded model-level scheduling and single-flight catalog refresh."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

from .policy import (Budget, Limits, TransientError, _CATALOG_TTL_SECONDS,
                     _MAX_CATALOG_BODY_BYTES,
                     _accept_catalog_id, _is_router_model)
from .transport import Transport, read_bounded


class Backend:
    def __init__(self, name, base_url, api_key='', tier=0, refresh=None,
                 default_model='', *, kind=None, free_tier_only=False):
        self.name, self.base_url = name, base_url.rstrip('/')
        self.api_key, self.tier = api_key, int(tier)
        self.kind = kind or name
        self.refresh, self.default_model = refresh, default_model
        self.free_tier_only = free_tier_only
        self.refresh_lock = threading.RLock()
        self.catalog_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self.cooldown_until = 0.0
        self.last_success = 0.0
        self.last_error_class = None
        self.cached_models = None
        self.cached_models_until = 0.0
        self.rows = {}
        self.model_cooldowns = {}
        self.latencies = {}
        self.closed = False
        self.catalog_retry_until = 0.0
        self.catalog_credentials = (self.base_url, self.api_key)

    def credential_snapshot(self):
        with self.refresh_lock:
            return self.base_url, self.api_key

    def is_available(self):
        with self._state_lock:
            return not self.closed and time.monotonic() >= self.cooldown_until

    def record_failure(self, retry_after=30.0, error_class='transient', *, model=None):
        with self._state_lock:
            until = time.monotonic() + max(0.0, retry_after)
            if model:
                self.model_cooldowns[model] = max(self.model_cooldowns.get(model, 0), until)
            else:
                self.cooldown_until = max(self.cooldown_until, until)
            self.last_error_class = error_class

    def record_success(self, *, model=None, elapsed=None):
        with self._state_lock:
            self.last_success = time.time()
            self.last_error_class = None
            # A concurrent success cannot erase a newer account-wide 429.
            if model and elapsed is not None:
                old = self.latencies.get(model, elapsed)
                self.latencies[model] = 0.8 * old + 0.2 * elapsed

    def set_catalog(self, rows, ttl=_CATALOG_TTL_SECONDS, *, credentials=None):
        accepted, rejected = {}, set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('id'), str):
                continue
            model = row['id']
            if _accept_catalog_id(self, model, row):
                accepted[model] = dict(row)
            else:
                rejected.add(model)
        # An ambiguous duplicate never wins by order or last-writer-wins.
        for model in rejected:
            accepted.pop(model, None)
        with self.refresh_lock, self._state_lock:
            if self.closed or (credentials is not None and credentials != (self.base_url, self.api_key)):
                return
            self.catalog_credentials = credentials or (self.base_url, self.api_key)
            self.rows = accepted
            self.cached_models = list(accepted)
            self.cached_models_until = time.monotonic() + ttl
            self.model_cooldowns = {k: v for k, v in self.model_cooldowns.items()
                                    if k in accepted and v > time.monotonic()}
            self.latencies = {k: v for k, v in self.latencies.items() if k in accepted}

    def set_cached_models(self, models, ttl=_CATALOG_TTL_SECONDS):
        self.set_catalog([{'id': model} for model in models], ttl)

    def get_cached_models(self):
        return list(self.available_rows())

    def available_rows(self):
        with self._state_lock:
            if not self.is_available() or self.catalog_credentials != (self.base_url, self.api_key):
                return {}
            now = time.monotonic()
            # Expired prices cannot establish spend authority. Keyless/local and
            # OpenRouter's server-enforced zero cap can safely use stale IDs.
            if now >= self.cached_models_until and self.kind not in {
                'opencode-free', 'openrouter', 'local',
            }:
                return {}
            return {key: dict(row) for key, row in self.rows.items()
                    if self.model_cooldowns.get(key, 0) <= now}

    def supports_model(self, model):
        return model in self.available_rows()

    def health(self):
        with self._state_lock:
            return {'name': self.name, 'tier': self.tier, 'available': self.is_available(),
                    'retry_after': max(0, self.cooldown_until - time.monotonic()),
                    'last_success': self.last_success, 'last_error_class': self.last_error_class,
                    'models_cached': len(self.rows), 'eligible_models': len(self.available_rows())}


@dataclass(frozen=True)
class Candidate:
    backend: Backend
    model: str
    row: dict

    @property
    def identity(self):
        return self.backend.name, self.model


class BackendPool:
    def __init__(self, limits=None):
        self.limits = limits or Limits()
        self.backends = []
        self._lock = threading.RLock()
        self._index = 0
        self._transport = None
        self._changed = threading.Event()
        self._workers = set()
        self._affinity = {}

    @property
    def transport(self):
        with self._lock:
            if self._transport is None:
                self._transport = Transport(self.limits)
            return self._transport

    def add(self, backend):
        with self._lock:
            if len(self.backends) >= 8 or any(b.name == backend.name for b in self.backends):
                raise ValueError('duplicate backend or backend bound exceeded')
            self.backends.append(backend)

    def snapshot(self):
        with self._lock:
            return list(self.backends)

    def count(self):
        return len(self.snapshot())

    def clear(self):
        with self._lock:
            old, self.backends = self.backends, []
            for backend in old:
                with backend._state_lock:
                    backend.closed = True
            transport, self._transport = self._transport, None
            self._index = 0
            self._affinity.clear()
        if transport is not None:
            transport.close()
        self._changed.set()

    def next(self, requested_model='', exclude=None):
        available = [b for b in self.snapshot() if b.is_available() and
                     b.name not in (exclude or set()) and
                     (_is_router_model(requested_model) or b.supports_model(requested_model))]
        if not available:
            return None
        tier = min(b.tier for b in available)
        group = [b for b in available if b.tier == tier]
        with self._lock:
            selected = group[self._index % len(group)]
            self._index += 1
        return selected

    def _fetch_models(self, backend, budget, transport=None):
        if backend.closed:
            raise TransientError('catalog generation retired')
        observed = backend.credential_snapshot()
        if not observed[1] and backend.refresh is not None:
            _refresh_backend_credentials(backend, require_new=False, observed=observed)
        observed = backend.credential_snapshot()
        response = (transport or self.transport).request(backend, 'GET', '/models', budget,
                                                         catalog=True, credentials=observed)
        raw = read_bounded(response, budget, _MAX_CATALOG_BODY_BYTES)
        try:
            payload = json.loads(raw)
            rows = payload if isinstance(payload, list) else payload['data']
            if not isinstance(rows, list):
                raise ValueError('not a list')
            return rows, observed
        except (ValueError, TypeError, KeyError) as exc:
            raise TransientError('invalid upstream catalog') from exc

    def refresh_catalogs(self):
        for backend in self.snapshot():
            with backend._state_lock:
                due = (backend.cached_models_until <= time.monotonic() + 15 and
                       backend.catalog_retry_until <= time.monotonic())
            if not due or not backend.is_available() or not backend.catalog_lock.acquire(False):
                continue
            transport = self.transport
            def worker(target=backend, transport=transport):
                try:
                    rows, credentials = self._fetch_models(target, Budget(self.limits.catalog), transport)
                    target.set_catalog(rows, credentials=credentials)
                except Exception:
                    # Negative-cache cold failures and back off refresh of stale
                    # lists, without renewing expired price evidence.
                    with target._state_lock:
                        cold = target.cached_models is None
                    # set_catalog acquires credential then state; never call it
                    # while holding state or a concurrent auth refresh can deadlock.
                    if cold:
                        target.set_catalog([], ttl=15)
                    with target._state_lock:
                        target.catalog_retry_until = time.monotonic() + 15
                finally:
                    target.catalog_lock.release()
                    self._changed.set()
                    with self._lock:
                        self._workers.discard(threading.current_thread())
            thread = threading.Thread(target=worker, name='freemaxxing-catalog', daemon=True)
            with self._lock:
                self._workers.add(thread)
            thread.start()

    def candidates(self, body, session=None):
        requested = str(body.get('model') or 'freemaxxing')
        selector, separator, selected = requested.partition('::')
        scoped_auto = bool(separator and _is_router_model(selected))
        groups = []
        for backend in sorted(self.snapshot(), key=lambda b: b.tier):
            if scoped_auto and backend.name != selector:
                continue
            group = []
            for model, row in backend.available_rows().items():
                qualified = f'{backend.name}::{model}'
                if not _is_router_model(requested) and not scoped_auto and requested not in {model, qualified}:
                    continue
                parameters = row.get('supported_parameters')
                if body.get('tools') and isinstance(parameters, list) and 'tools' not in parameters:
                    continue
                if body.get('response_format') and isinstance(parameters, list) and 'response_format' not in parameters:
                    continue
                group.append(Candidate(backend, model, row))
            group.sort(key=lambda c: (c.model != backend.default_model,
                                     backend.latencies.get(c.model, float('inf'))))
            if group:
                groups.append(group)
        # Try another provider before cycling through all models on one provider.
        candidates = [group[i] for i in range(max(map(len, groups), default=0))
                      for group in groups if i < len(group)]
        if session and _is_router_model(requested):
            with self._lock:
                preferred = self._affinity.get(session)
            candidates.sort(key=lambda c: c.identity != preferred)
        return candidates

    def remember(self, session, identity):
        if not session:
            return
        with self._lock:
            self._affinity.pop(session, None)
            if len(self._affinity) >= 1024:
                self._affinity.pop(next(iter(self._affinity)))
            self._affinity[session] = identity

    def catalog_pending(self):
        with self._lock:
            return bool(self._workers)

    def wait_for_catalog(self, budget, timeout=None):
        self._changed.wait(min(timeout or self.limits.catalog, budget.remaining()))
        self._changed.clear()

    def get_aggregated_models(self):
        # Picker/catalog calls are metadata-only and cannot spend quota.
        return [{'id': 'freemaxxing', 'object': 'model', 'owned_by': 'freemaxxing'}]

    def health(self):
        return {'backends': [b.health() for b in self.snapshot()]}

    def exhaustion_detail(self):
        return 'No eligible free route; session state has not been modified.'

    def retry_after(self):
        return max(1, min((b.health()['retry_after'] for b in self.snapshot()
                           if not b.is_available()), default=5))


pool = BackendPool()


def _resolve_auto_model(backend):
    rows = backend.available_rows()
    return backend.default_model if backend.default_model in rows else next(iter(rows), '')


def _refresh_backend_credentials(backend, *, require_new, observed=None):
    if backend.refresh is None:
        return False
    with backend.refresh_lock:
        current = backend.base_url, backend.api_key
        if observed is not None and current != observed:
            return True
        base, key = backend.refresh()
        base, key = str(base or current[0]).rstrip('/'), str(key or '').strip()
        if not key or (require_new and (base, key) == current):
            return False
        with backend._state_lock:
            backend.base_url, backend.api_key = base, key
            if (base, key) != current:
                # Price/capability evidence from another credential cannot authorize
                # a refreshed endpoint, even when its model names happen to match.
                backend.rows = {}
                backend.cached_models = None
                backend.cached_models_until = 0
                backend.catalog_retry_until = 0
        return True
