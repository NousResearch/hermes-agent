"""Private, one-operation selection. Never resolve credentials from a readiness probe.

The 30-second ceiling is snapshot authority, not an account-health promise. No
material is retained by the adapter; the caller owns and closes each binding.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import sqlite3
import weakref
from contextvars import ContextVar

_held_selection = ContextVar("held_selected_route", default=None)

from hermes_state_runtime import RuntimeStoreError, _epoch

MAX_ROUTE_AGE = 30.0
# These transports consume the existing SDK request-copy Files representation.
_FILES_MODES = frozenset({'chat_completions', 'codex_responses', 'anthropic_messages'})


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
    """The session-state owner's writer discipline, also used by API recovery."""
    from gateway.session_state import selection_lock
    return selection_lock(owner)


class SessionRuntimeSelection:
    """Captured read branches plus deferred cache writes; deliberately not serializable."""
    def __init__(self, runner, key, *, local_policy=False, source=None, user_config=None):
        self.runner, self.key = runner, key
        self.local_policy = local_policy
        self.source, self.user_config = source, user_config
        # Effective loader I/O is deliberately outside all owner/cache/SQL locks.
        self._effective_inputs = self.effective_inputs()
        with publication_lock(runner):
            self._capture()

    def effective_inputs(self):
        if self.local_policy:
            return None  # LOCAL resolves its immutable launch policy, not live config.
        from gateway.run import _load_gateway_config
        from hermes_cli.runtime_provider import load_config
        from agent.secret_scope import current_secret_scope
        from tools.terminal_scope import get_terminal_scope
        config = getattr(self.runner, 'config', None)
        import os
        # get_secret() may fall through to process env in standalone mode. Keep
        # this conservative value snapshot operation-local, never on the scope.
        return _copy((_load_gateway_config(), load_config(), self.user_config,
                      config.to_dict() if config is not None else None,
                      self.source.to_dict() if self.source is not None else None,
                      current_secret_scope(), get_terminal_scope(), dict(os.environ)))

    def inputs_current(self):
        return self.effective_inputs() == self._effective_inputs

    def _capture(self):
        runner, key, local_policy = self.runner, self.key, self.local_policy
        state = runner._peek_session_state(key) if key else None
        self.revision = state.conversation.selection_revision if state else 0
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
        return ((state.conversation.selection_revision if state else 0) == self.revision
            and (state.conversation.model_override if state else None) == self.override
            and (state.conversation.reasoning_override if state else None) == self.reasoning
            and (state.conversation.service_tier_override if state else _SERVICE_TIER_UNSET) == self.tier)

    def persistence_guard(self):
        store = getattr(self.runner, 'session_store', None)
        if self.key and self.override is None and store is not None and not self.local_policy:
            return store._lock
        return nullcontext()

    def current(self, *, store_locked=False):
        if self.local_policy:
            return True
        if not self.memory_current():
            return False
        store = getattr(self.runner, 'session_store', None)
        if self.key and self.override is None and store is not None:
            try:
                if store_locked:
                    # Capture already attempted hydration. Preserve the ordinary
                    # unavailable-index fallback while comparing cached entries
                    # under their writer lock; never reconcile/reopen under SQL.
                    entry = store._entries.get(self.key)
                    persisted = (entry.model_override or None) if entry else None
                else:
                    persisted = store.get_model_override(self.key)
                if persisted != self.persisted:
                    return False
            except Exception:
                if self.persisted is not None:
                    return False
        return all((s.conversation.last_resolved_model if (s := self.runner._peek_session_state(k)) else '') == v
                   for k, v in self.last.items() if k != '*' or self.used_global_recovery)

    def publish(self, *, inputs_checked=False, store_locked=False):
        if not inputs_checked and not self.inputs_current():
            raise SelectedRouteUnavailable('selection_changed')
        guard = nullcontext() if store_locked else self.persistence_guard()
        with publication_lock(self.runner), guard:
            if self._published or not self.current(store_locked=True):
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

    def release_snapshots(self):
        # Execution owns runtime/turn_route after consume. Input-validation copies
        # have no execution purpose and must not follow that transfer.
        self._effective_inputs = self.user_config = None
        self.override = self.pending_override = self.reasoning = None

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
        self.source = source
        self.source_value = source.to_dict()
        self.session_key, self.session_id = session_key, session_id
        self.request_identity, self.purpose = request_identity, purpose
        self._generation = 0
        self._replacement_lock = threading.RLock()
        self._binding_ref = None
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
            # Caller already owns the DB SQL lock. Do not acquire it recursively.
            if self.db._read_conns_closed or connection is not self.db._conn:
                raise SelectedRouteUnavailable('owner_unavailable')
            return read(connection)
        with self.db.live_read_connection() as conn:
            if conn is None:
                raise SelectedRouteUnavailable('owner_unavailable')
            return read(conn)

    def current(self, connection=None):
        from hermes_constants import hermes_home_key, get_hermes_home
        from gateway.session_authorities import authority_for_home
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
        self._state_lock = threading.Lock()
        self._state = 'PREPARING'
        self._held_thread = None
        self._inputs = None
        self._user_config, self._settings = None, None
        self._adapter_models = self._adapter_before = None
        self._adapter_recovery = False
        from agent.secret_scope import current_secret_scope
        from tools.terminal_scope import get_terminal_scope
        self._secret_snapshot = _copy(current_secret_scope())
        self._terminal_snapshot = _copy(get_terminal_scope())

    def context_current(self):
        from agent.secret_scope import current_secret_scope
        from tools.terminal_scope import get_terminal_scope
        return (current_secret_scope() == self._secret_snapshot
                and get_terminal_scope() == self._terminal_snapshot)

    def close(self):
        # Close waits for an active operation, then linearizes under the tiny
        # projection lock. Peek takes only the latter, even from SQL callbacks;
        # it never waits for the operation hold or nests into owner/cache locks.
        with self._lock, self._state_lock:
            if self._material is not None:
                self._material.release_snapshots()
            self._material = None
            self._secret_snapshot = self._terminal_snapshot = None
            self._inputs = None
            self._user_config = self._settings = self._adapter_models = self._adapter_before = None
            if self._state != 'CONSUMED':
                self._state = 'RETIRED'

    def __reduce__(self):
        raise TypeError('private selected route cannot be serialized')


def prepare_selected_route(scope, *, user_config=None, api_settings=None, cancelled=None):
    """Explicit off-loop operation. Caller must not hold SQL/spool/cache locks."""
    previous = scope._binding_ref() if scope._binding_ref is not None else None
    if previous is not None and previous._held_thread == threading.get_ident():
        # Reject before scope/profile reads: this thread already owns the hold's
        # non-reentrant store/SQL locks. Never replace an operation from within it.
        raise SelectedRouteUnavailable('selection_in_use')
    if not scope.current():
        raise SelectedRouteUnavailable('owner_unavailable')
    # Use the runner's actual multiplex/standalone scope semantics, including
    # home, secret hydration and terminal policy. A stale caller scope refuses.
    with scope.runner._profile_scope_for_source(scope.source):
        return _prepare_scoped(scope, user_config=user_config, api_settings=api_settings, cancelled=cancelled)


def _prepare_scoped(scope, *, user_config, api_settings, cancelled):
    if not scope.current():
        raise SelectedRouteUnavailable('owner_unavailable')
    # Retire the displaced operation before issuing its replacement. Wait for
    # its active hold without holding runner/cache/SQL locks (hold takes those
    # after the binding lock). The weak link cannot extend credential lifetime.
    with scope._replacement_lock:
        previous = scope._binding_ref() if scope._binding_ref is not None else None
        if previous is not None:
            previous.close()
        with publication_lock(scope.runner):
            generation = scope.authority.__dict__.get('_selected_route_generation', 0) + 1
            scope.authority._selected_route_generation = generation
            scope._generation = generation
        b = PreparedSelectedRoute(scope, generation, time.monotonic() + MAX_ROUTE_AGE)
        scope._binding_ref = weakref.ref(b)
    material = None
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
        if (scope._generation != b.generation or not scope.current() or not material.current()
                or _inputs(scope, user_config, b._settings) != b._inputs
                or b._adapter_recovery and scope.adapter._last_resolved_model != b._adapter_before):
            raise SelectedRouteUnavailable('selection_changed')
        if cancelled is not None and cancelled():
            raise SelectedRouteUnavailable('selection_cancelled')
        b.deadline = _credential_deadline(material.runtime, b.deadline)
        if time.monotonic() >= b.deadline:
            raise SelectedRouteUnavailable('selection_expired')
        material.files_supported = supports_files_runtime(material.runtime)
        with b._state_lock:
            if b._state != 'PREPARING' or scope._generation != b.generation:
                raise SelectedRouteUnavailable('selection_changed')
            b._material = material
            b._state = 'READY'
        return b
    except BaseException:
        if material is not None:
            material.release_snapshots()
        b.close()
        raise


def peek_selected_route(scope, binding=None, connection=None):
    """No lazy initialization, config loading, credentials or cache writes."""
    if binding is None:
        return RouteReadiness('UNPREPARED')
    if not binding._scope.matches(scope) or not scope.current(connection):
        return RouteReadiness('UNAVAILABLE', reason='owner_unavailable')
    with binding._state_lock:
        material = binding._material
        state = binding._state
        if state not in {'READY', 'HELD'} or material is None:
            return RouteReadiness(state, reason='not_ready')
        context_current = binding.context_current()
    if not context_current:
        return RouteReadiness('UNAVAILABLE', reason='owner_unavailable')
    current = material.memory_current()
    # The final short snapshot is the observation's linearization point. close()
    # can retire while the DB/memory checks run, but cannot leave a torn result.
    with binding._state_lock:
        if binding._material is not material or binding._state not in {'READY', 'HELD'}:
            return RouteReadiness(binding._state, reason='not_ready')
        if scope._generation != binding.generation:
            return RouteReadiness('RETIRED', reason='selection_superseded')
        if time.monotonic() >= binding.deadline:
            return RouteReadiness('STALE', reason='selection_expired')
        if scope.db._read_conns_closed:
            return RouteReadiness('UNAVAILABLE', reason='owner_unavailable')
        if not current:
            return RouteReadiness('STALE', reason='selection_changed')
        supported = material.files_supported
        return RouteReadiness(binding._state, supported,
            'ready' if supported else 'unsupported_transport', binding.generation, binding.deadline)


@contextmanager
def hold_selected_route(scope, binding):
    """Acquire BEFORE shared-grant and owner SQL writers; never refresh inside."""
    # Effective input read is outside the hold and, crucially, outside SQL writers.
    if not binding._scope.matches(scope):
        raise SelectedRouteUnavailable('not_ready')  # foreign observation is nondestructive
    with binding._state_lock:
        material = binding._material
        if binding._state != 'READY' or material is None or not binding.context_current():
            raise SelectedRouteUnavailable('not_ready')
        user_config, settings = binding._user_config, binding._settings
    # All loaders/credentials run before the owner hold, never inside a SQL writer.
    try:
        inputs = _inputs(scope, user_config, settings)
        effective_current = material.inputs_current()
        with (binding._lock, publication_lock(scope.runner), publication_lock(scope.adapter),
              material.persistence_guard()):
            if (not scope.current() or scope._generation != binding.generation
                    or time.monotonic() >= binding.deadline):
                raise SelectedRouteUnavailable('not_ready')
            if (not material.current(store_locked=True) or not effective_current or inputs != binding._inputs
                    or binding._adapter_recovery and scope.adapter._last_resolved_model != binding._adapter_before):
                raise SelectedRouteUnavailable('selection_changed')
            with binding._state_lock:
                if binding._state != 'READY' or binding._material is not material:
                    raise SelectedRouteUnavailable('not_ready')
                binding._state, binding._held_thread = 'HELD', threading.get_ident()
            token = _held_selection.set(binding)
            try:
                yield binding
            finally:
                _held_selection.reset(token)
                binding._held_thread = None
    finally:
        binding.close()


@contextmanager
def consume_selected_route(scope, binding):
    """Execution-only single consume. Yield private material, clear on every exit."""
    if (scope.purpose != 'execute' or not binding._scope.matches(scope)
            or scope._generation != binding.generation
            or binding._state != 'HELD' or binding._held_thread != threading.get_ident()
            or not scope.current() or time.monotonic() >= binding.deadline):
        raise SelectedRouteUnavailable('not_ready')
    material = binding._material
    try:
        with publication_lock(scope.runner), publication_lock(scope.adapter):
            if binding._adapter_recovery and scope.adapter._last_resolved_model != binding._adapter_before:
                raise SelectedRouteUnavailable('selection_changed')
            material.publish(inputs_checked=True, store_locked=True)
            for key, value in binding._adapter_models.items():
                if scope.adapter._last_resolved_model.get(key) == binding._adapter_before.get(key):
                    scope.adapter._last_resolved_model[key] = value
        with binding._state_lock:
            if binding._state != 'HELD' or binding._material is not material:
                raise SelectedRouteUnavailable('not_ready')
            binding._state = 'CONSUMED'
        yield material
    finally:
        binding.close()


def held_selected_route(runner=None):
    """A copied context cannot inherit the operation's thread-owned hold."""
    b = _held_selection.get()
    if (b is None or b._state != 'HELD' or b._held_thread != threading.get_ident()
            or runner is not None and b._scope.runner is not runner):
        return None
    return b


def session_store_guard(runner):
    b = held_selected_route(runner)
    material = b._material if b is not None else None
    owns_store = (material is not None and material.key and material.override is None
                  and not material.local_policy)
    return nullcontext() if owns_store else runner.session_store._lock


@contextmanager
def own_session_creation(authority, session_id, source, storage_source, connection):
    """Allow only this held operation's own canonical absent→present SQL write.

    The caller enters before inspecting/inserting the row and exits on the same
    owner transaction. Existing rows cannot change selection revisions here.
    """
    b = held_selected_route(authority.runner)
    if b is None:
        yield
        return
    scope = b._scope
    if (scope.purpose != 'admit' or scope.authority is not authority
            or scope.session_id != session_id or scope.source_value != source.to_dict()
            or storage_source != 'bot_room' or not connection.in_transaction
            or not scope.current(connection)
            or not peek_selected_route(scope, b, connection).supports_prepared_files):
        raise SelectedRouteUnavailable('selection_changed')
    before = scope.revision
    yield
    after = scope._revision(connection)
    if before is None:
        if after != (0, None, None, 'bot_room', scope.session_key):
            raise SelectedRouteUnavailable('selection_changed')
        row = connection.execute('SELECT hidden,origin_json FROM sessions WHERE id=?', (session_id,)).fetchone()
        import json
        if row is None or row[0] != 1 or json.loads(row[1]) != scope.source_value:
            raise SelectedRouteUnavailable('selection_changed')
        scope.revision = after
    elif after != before:
        raise SelectedRouteUnavailable('selection_changed')


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
    """Classify the actual lower construction branch, during explicit preparation.

    agent_init._build_client chooses Anthropic or MoA before the OpenAI factory;
    that factory can delegate to a profile or native Gemini. Only the physical
    OpenAI/Anthropic SDK branches establish this request-copy contract. Never
    call this from peek (profile lookup may initialize the provider registry).
    """
    from urllib.parse import urlsplit
    mode, provider = runtime.get('api_mode'), runtime.get('provider')
    if (mode not in _FILES_MODES or not provider or runtime.get('command')
            or runtime.get('acp_command') or provider == 'moa'
            or urlsplit(runtime.get('base_url') or '').scheme not in {'http', 'https'}):
        return False
    if mode == 'anthropic_messages':
        return True  # _build_client's first branch, before provider-supplied clients
    from providers import get_provider_profile
    from providers.base import ProviderProfile
    from agent.agent_runtime_helpers import _profile_for_base_url
    from agent.auxiliary_client import _GEMINI_NATIVE_PROVIDER_NAMES
    from agent.gemini_native_adapter import is_native_gemini_base_url
    profile = get_provider_profile(provider) or _profile_for_base_url(runtime['base_url'])
    if profile is not None and getattr(profile.create_client, '__func__', None) is not ProviderProfile.create_client:
        return False  # no capability contract for delegated/plugin clients
    if provider in _GEMINI_NATIVE_PROVIDER_NAMES and is_native_gemini_base_url(runtime['base_url']):
        return False
    return True


def supports_files_agent(agent):
    """Post-setup closed positive set, including init-time fallback/client drift."""
    from openai import OpenAI
    from anthropic import Anthropic
    mode = getattr(agent, 'api_mode', None)
    if (getattr(agent, 'acp_command', None) or getattr(agent, 'provider', None) == 'moa'):
        return False
    if mode == 'anthropic_messages':
        return type(getattr(agent, '_anthropic_client', None)) is Anthropic
    return mode in {'chat_completions', 'codex_responses'} and type(getattr(agent, 'client', None)) is OpenAI


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
        try:
            # Resolve request options before validation/publication as API does.
            finish_selected_route(runner, material, ctx.source, ctx.session_key, None, policy)
            material.publish()
            return material
        finally:
            material.release_snapshots()
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
