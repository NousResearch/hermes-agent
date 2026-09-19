"""Private, one-operation selection. Never resolve credentials from a readiness probe.

The 30-second ceiling is snapshot authority, not an account-health promise. No
material is retained by the adapter; the caller owns and closes each binding.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import sqlite3

from hermes_state_runtime import RuntimeStoreError, _epoch

MAX_ROUTE_AGE = 30.0
# These transports consume the existing SDK request-copy Files representation.
_FILES_MODES = frozenset({'chat_completions', 'responses', 'codex_responses', 'anthropic_messages'})


class SelectedRouteUnavailable(RuntimeStoreError):
    pass


def _copy(value):
    """Copy value containers, never clone selected credential-pool identities."""
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy(v) for v in value)
    return value


def publication_lock(owner):
    """Short cache-publication CAS lock, never held across credential work."""
    return owner.__dict__.setdefault('_selected_route_publication_lock', threading.RLock())


class SessionRuntimeSelection:
    """Captured read branches plus deferred cache writes; deliberately not serializable."""
    def __init__(self, runner, key, *, local_policy=False):
        self.runner, self.key = runner, key
        self.local_policy = local_policy
        state = runner._peek_session_state(key) if key else None
        self.override = _copy(state.conversation.model_override) if state else None
        self.reasoning = _copy(state.conversation.reasoning_override) if state else None
        from gateway.session_state import SERVICE_TIER_UNSET as _SERVICE_TIER_UNSET
        self.tier = state.conversation.service_tier_override if state else _SERVICE_TIER_UNSET
        store = getattr(runner, 'session_store', None)
        self.persisted = None
        if key and self.override is None and store is not None and not local_policy:
            try:
                self.persisted = _copy(store.get_model_override(key))
            except Exception:
                # Match the existing rehydration read-failure branch.
                self.persisted = None
        self.last = {k: (s.conversation.last_resolved_model if (s := runner._peek_session_state(k)) else '')
                     for k in (key, '*') if k}
        self.pending_override = None
        self.pending_model = None
        self.used_global_recovery = False
        self.model, self.runtime = '', {}
        self._published = False

    def memory_current(self):
        if self.local_policy:
            return True
        state = self.runner._peek_session_state(self.key) if self.key else None
        from gateway.session_state import SERVICE_TIER_UNSET as _SERVICE_TIER_UNSET
        return ((state.conversation.model_override if state else None) == self.override
            and (state.conversation.reasoning_override if state else None) == self.reasoning
            and (state.conversation.service_tier_override if state else _SERVICE_TIER_UNSET) == self.tier)

    def current(self):
        if self.local_policy:
            return True
        if not self.memory_current():
            return False
        store = getattr(self.runner, 'session_store', None)
        if self.key and self.override is None and store is not None:
            try:
                if store.get_model_override(self.key) != self.persisted:
                    return False
            except Exception:
                if self.persisted is not None:
                    return False
        return all((s.conversation.last_resolved_model if (s := self.runner._peek_session_state(k)) else '') == v
                   for k, v in self.last.items() if k != '*' or self.used_global_recovery)

    def publish(self):
        with publication_lock(self.runner):
            if self._published or not self.current():
                raise SelectedRouteUnavailable('selection_changed')
            if self.pending_override is not None:
                self.runner._session_state(self.key).conversation.model_override = self.pending_override
            if self.pending_model:
                if self.key:
                    self.runner._session_state(self.key).conversation.last_resolved_model = self.pending_model
                star = self.runner._peek_session_state('*')
                if (star.conversation.last_resolved_model if star else '') == self.last.get('*', ''):
                    self.runner._session_state('*').conversation.last_resolved_model = self.pending_model
            self._published = True

    def __reduce__(self):
        raise TypeError('private selection cannot be serialized')


@dataclass(frozen=True)
class RouteReadiness:
    state: str
    supports_prepared_files: bool = False
    reason: str = 'unprepared'
    generation: int | None = None
    deadline: float | None = None


class SelectionScope:
    def __init__(self, adapter, authority, source, session_key, session_id, request_identity, purpose):
        self.adapter, self.authority = adapter, authority
        self.runner, self.db = authority.runner, authority.db
        self.registry = self.runner.session_authorities
        self.db_path = str(Path(self.db.db_path).resolve())
        self.run_store = adapter._run_idempotency_store
        self.run_store_path = self.run_store._db_path
        self.home, self.instance, self.epoch = authority.profile_id, authority.instance_id, authority.epoch
        from agent.secret_scope import current_secret_scope
        from tools.terminal_scope import get_terminal_scope
        self.secret_scope = _copy(current_secret_scope())
        self.terminal_scope = _copy(get_terminal_scope())
        self.source = source
        self.source_value = source.to_dict()
        self.session_key, self.session_id = session_key, session_id
        self.request_identity, self.purpose = request_identity, purpose
        self.revision = self._revision()

    def _revision(self, connection=None):
        def read(conn):
            actual = next((row[2] for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), None)
            if not actual or str(Path(actual).resolve()) != self.db_path:
                raise SelectedRouteUnavailable('owner_unavailable')
            _epoch(conn, self.epoch)
            row = conn.execute('SELECT runtime_revision,model,model_config,source,session_key FROM sessions WHERE id=?',
                               (self.session_id,)).fetchone()
            return tuple(row) if row else None
        if connection is not None:
            return read(connection)
        with self.db._read_ctx() as conn:
            return read(conn)

    def current(self, connection=None):
        from hermes_constants import hermes_home_key, get_hermes_home
        from gateway.session_authorities import authority_for_home
        from agent.secret_scope import current_secret_scope
        from tools.terminal_scope import get_terminal_scope
        r, a = self.runner, self.authority
        try:
            return (hermes_home_key(get_hermes_home()) == hermes_home_key(self.home)
                and r.session_authorities is self.registry
                and authority_for_home(r, self.home) is a and a.runner is r and a.db is self.db
                and a.instance_id == self.instance and a.epoch == self.epoch
                and not r._draining and r._adapter_for_source(self.source) is self.adapter
                and self.adapter.gateway_runner is r and not self.adapter._session_db_cache_closed
                and self.adapter._run_idempotency_store is self.run_store
                and self.run_store._db_path == self.run_store_path
                and current_secret_scope() == self.secret_scope
                and get_terminal_scope() == self.terminal_scope
                and self.source.to_dict() == self.source_value
                and self._revision(connection) == self.revision)
        except (RuntimeStoreError, sqlite3.Error):
            return False

    def matches(self, other):
        return (other is self or isinstance(other, SelectionScope)
            and all(getattr(self, k) is getattr(other, k) for k in
                    ('adapter', 'authority', 'runner', 'db', 'registry', 'request_identity'))
            and all(getattr(self, k) == getattr(other, k) for k in
                    ('home', 'instance', 'epoch', 'session_key', 'session_id', 'purpose', 'revision', 'source_value')))

    def __reduce__(self):
        raise TypeError('private selection scope cannot be serialized')


def selection_scope(adapter, *, source, session_key, session_id, request_identity, purpose):
    from gateway.session_authorities import authority_for_home
    from gateway.config import Platform
    from hermes_constants import get_hermes_home
    runner = adapter.gateway_runner
    if (source.platform != Platform.API_SERVER or source.chat_id != session_id
            or runner.session_store._generate_session_key(source) != session_key):
        raise SelectedRouteUnavailable('owner_unavailable')
    authority = authority_for_home(runner, get_hermes_home())
    if authority is None:
        raise SelectedRouteUnavailable('owner_unavailable')
    scope = SelectionScope(adapter, authority, source, session_key, session_id, request_identity, purpose)
    if not scope.current():
        raise SelectedRouteUnavailable('owner_unavailable')
    return scope


def _inputs(scope, user_config, settings):
    # Same effective loaders used by gateway and runtime-provider, never raw YAML.
    from gateway.run import _load_gateway_config
    from hermes_cli.runtime_provider import load_config
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(scope.runner, scope.source)
    return _copy((_load_gateway_config(), load_config(), user_config, settings,
                  getattr(scope.runner, 'config', None).to_dict(),
                  scope.adapter._model_routes, policy))


class PreparedSelectedRoute:
    def __init__(self, scope, generation, deadline):
        self._scope, self.generation, self.deadline = scope, generation, deadline
        self._material = None
        self._lock = threading.RLock()
        self._state = 'PREPARING'
        self._held_thread = None
        self._inputs = None
        self._user_config, self._settings = None, None
        self._adapter_models = self._adapter_before = None
        self._adapter_recovery = False

    def close(self):
        with self._lock:
            self._material = None
            self._inputs = None
            self._user_config = self._settings = self._adapter_models = self._adapter_before = None
            if self._state != 'CONSUMED':
                self._state = 'RETIRED'

    def __reduce__(self):
        raise TypeError('private selected route cannot be serialized')


def prepare_selected_route(scope, *, user_config=None, api_settings=None, cancelled=None):
    """Explicit off-loop operation. Caller must not hold SQL/spool/cache locks."""
    if not scope.current():
        raise SelectedRouteUnavailable('owner_unavailable')
    # Use the runner's actual multiplex/standalone scope semantics, including
    # home, secret hydration and terminal policy. A stale caller scope refuses.
    with scope.runner._profile_scope_for_source(scope.source):
        return _prepare_scoped(scope, user_config=user_config, api_settings=api_settings, cancelled=cancelled)


def _prepare_scoped(scope, *, user_config, api_settings, cancelled):
    if not scope.current():
        raise SelectedRouteUnavailable('owner_unavailable')
    # Authority-local issuance counter: only serial issuance, never credential work under this lock.
    lock = scope.authority.__dict__.setdefault('_selected_route_lock', threading.RLock())
    with lock:
        generation = scope.authority.__dict__.get('_selected_route_generation', 0) + 1
        scope.authority._selected_route_generation = generation
    b = PreparedSelectedRoute(scope, generation, time.monotonic() + MAX_ROUTE_AGE)
    try:
        b._user_config, b._settings = _copy(user_config), _copy(api_settings or {})
        b._inputs = _inputs(scope, user_config, b._settings)
        material = scope.runner._prepare_session_agent_runtime(source=scope.source,
            session_key=scope.session_key, user_config=user_config)
        from gateway.session_api_turn import prepare_api_runtime
        with publication_lock(scope.adapter):
            b._adapter_before = dict(scope.adapter._last_resolved_model)
        b._adapter_models = dict(b._adapter_before)
        b._adapter_recovery = not material.model
        material.model, material.runtime = prepare_api_runtime(material.model, material.runtime,
            current={'adapter': scope.adapter, 'settings': b._settings},
            pending_models=b._adapter_models)
        finish_selected_route(scope.runner, material, scope.source, scope.session_key, b._settings)
        if (not scope.current() or not material.current()
                or _inputs(scope, user_config, b._settings) != b._inputs
                or b._adapter_recovery and scope.adapter._last_resolved_model != b._adapter_before):
            raise SelectedRouteUnavailable('selection_changed')
        if cancelled is not None and cancelled():
            raise SelectedRouteUnavailable('selection_cancelled')
        b.deadline = _credential_deadline(material.runtime, b.deadline)
        if time.monotonic() >= b.deadline:
            raise SelectedRouteUnavailable('selection_expired')
        b._material = material
        b._state = 'READY'
        return b
    except BaseException:
        b.close()
        raise


def peek_selected_route(scope, binding=None, connection=None):
    """No lazy initialization, config loading, credentials or cache writes."""
    if binding is None:
        return RouteReadiness('UNPREPARED')
    if not binding._scope.matches(scope) or not scope.current(connection):
        return RouteReadiness('UNAVAILABLE', reason='owner_unavailable')
    if time.monotonic() >= binding.deadline:
        return RouteReadiness('STALE', reason='selection_expired')
    if binding._state not in {'READY', 'HELD'} or binding._material is None:
        return RouteReadiness(binding._state, reason='not_ready')
    if not binding._material.memory_current():
        return RouteReadiness('STALE', reason='selection_changed')
    supported = supports_files_runtime(binding._material.runtime)
    return RouteReadiness(binding._state, supported,
        'ready' if supported else 'unsupported_transport', binding.generation, binding.deadline)


@contextmanager
def hold_selected_route(scope, binding):
    """Acquire BEFORE shared-grant and owner SQL writers; never refresh inside."""
    # Effective input read is outside the hold and, crucially, outside SQL writers.
    if not binding._scope.matches(scope) or binding._state != 'READY' or not scope.current():
        raise SelectedRouteUnavailable('not_ready')
    inputs = _inputs(scope, binding._user_config, binding._settings)
    with binding._lock:
        if (not binding._scope.matches(scope) or not scope.current()
                or binding._state != 'READY' or time.monotonic() >= binding.deadline):
            raise SelectedRouteUnavailable('not_ready')
        if (not binding._material.current() or inputs != binding._inputs
                or binding._adapter_recovery and scope.adapter._last_resolved_model != binding._adapter_before):
            binding.close()
            raise SelectedRouteUnavailable('selection_changed')
        binding._state, binding._held_thread = 'HELD', threading.get_ident()
        try:
            yield binding
        finally:
            binding._held_thread = None
            binding.close()


@contextmanager
def consume_selected_route(scope, binding):
    """Execution-only single consume. Yield private material, clear on every exit."""
    if (scope.purpose != 'execute' or not binding._scope.matches(scope)
            or binding._state != 'HELD' or binding._held_thread != threading.get_ident()
            or not scope.current() or time.monotonic() >= binding.deadline):
        raise SelectedRouteUnavailable('not_ready')
    material = binding._material
    try:
        with publication_lock(scope.runner), publication_lock(scope.adapter):
            if binding._adapter_recovery and scope.adapter._last_resolved_model != binding._adapter_before:
                raise SelectedRouteUnavailable('selection_changed')
            material.publish()
            for key, value in binding._adapter_models.items():
                if scope.adapter._last_resolved_model.get(key) == binding._adapter_before.get(key):
                    scope.adapter._last_resolved_model[key] = value
        binding._state = 'CONSUMED'
        yield material
    finally:
        binding.close()


def _credential_deadline(runtime, ceiling):
    """Read only the already-selected pool entry, during explicit preparation."""
    from agent.credential_pool import _parse_absolute_timestamp
    pool = runtime.get('credential_pool')
    if pool is None:
        return ceiling
    entry = pool.current()
    if entry is None or entry.runtime_api_key != runtime.get('api_key'):
        # An explicit API alias key wins after pool resolution. Its validity is
        # unknown; never apply another credential's expiry or change precedence.
        return ceiling
    fields = ('agent_key_expires_at',) if runtime.get('api_key') == getattr(entry, 'agent_key', None) else ('expires_at', 'expires_at_ms')
    times = [_parse_absolute_timestamp(getattr(entry, field, None)) for field in fields]
    known = [v for v in times if v is not None]
    if known:
        return min(ceiling, time.monotonic() + min(known) - time.time())
    return ceiling


def supports_files_runtime(runtime):
    return runtime.get('api_mode') in _FILES_MODES and not runtime.get('command')


def finish_selected_route(runner, material, source, session_key, settings, policy=None):
    """The same final request-option composition for preflight and execution."""
    if policy and policy.model:
        material.model = policy.model
    material.reasoning_config = (policy.reasoning_config if policy else
        runner._resolve_session_reasoning_config(source=source, session_key=session_key, model=material.model))
    material.service_tier = runner._resolve_session_service_tier(source=source, session_key=session_key)
    if settings is not None:
        from gateway.platforms.api_server import _request_reasoning_config, _request_service_tier, _REQUEST_OPTION_MISSING
        requested = _request_reasoning_config(settings.get('model_options'))
        if requested is not None:
            material.reasoning_config = requested
        tier = _request_service_tier(settings.get('model_options'))
        if tier is not _REQUEST_OPTION_MISSING:
            material.service_tier = tier
    material.turn_route = runner._resolve_turn_agent_config('', material.model, material.runtime,
                                                           service_tier=material.service_tier)


def execution_has_files(settings):
    return bool((settings.get('room_dispatch') or {}).get('attachment_manifest_digest'))


def unsupported_files_result():
    detail = 'Prepared Files input is not supported by the selected transport; no model turn was started.'
    return dict(final_response=detail, error=detail, messages=[], api_calls=0, tools=[],
                failed=True, completed=False, failure_reason='prepared_files_unsupported', failure_retryable=False)


def select_execution_route(turn, policy):
    """Real TurnRunner consumer; no second gateway selection after consumption."""
    from gateway.session_api_turn import api_execution
    runner, ctx = turn._runner, turn._ctx
    api = api_execution.get()
    if api is None:
        # Ordinary messaging/LOCAL retains its existing branch and publication boundary.
        material = runner._prepare_session_agent_runtime(source=ctx.source,
            session_key=ctx.session_key, user_config=ctx.user_config)
        material.publish()
        finish_selected_route(runner, material, ctx.source, ctx.session_key, None, policy)
        return material
    scope = selection_scope(api['adapter'], source=ctx.source, session_key=ctx.session_key,
        session_id=ctx.session_id, request_identity=turn, purpose='execute')
    binding = prepare_selected_route(scope, user_config=ctx.user_config, api_settings=api['settings'])
    try:
        if execution_has_files(api['settings']) and not peek_selected_route(scope, binding).supports_prepared_files:
            raise SelectedRouteUnavailable('prepared_files_unsupported')
        with hold_selected_route(scope, binding):
            with consume_selected_route(scope, binding) as material:
                return material
    finally:
        binding.close()
