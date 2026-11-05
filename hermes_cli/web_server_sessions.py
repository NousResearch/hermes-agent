"""Read-only session browsing, latest-descendant lookup and owner maintenance.
"""

import logging
import asyncio
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# Desktop 0.22.2's ``readableError`` keeps the HTTP ``detail`` and shows its "app and backend are
# on different versions" toast (with an Update action) for "out of sync (different versions)".
CLIENT_UPGRADE_REQUIRED = ('client_upgrade_required: this app is out of sync (different versions) '
                           'with the Hermes backend; update the app to edit sessions')


def _session_latest_descendant(session_id: str, store):
    """Resolve a session id to the newest child leaf session on the storage handle ``store``.

    /model may create child sessions; a dashboard refresh should continue the
    newest child instead of reopening the old parent. Returns ``(leaf, path)``.
    """
    sid = store.sessions.resolve_id(session_id)
    if not sid or not store.sessions.get(sid):
        return None, []

    # Continuation edges only: a subagent run, a /branch fork, a /new reset child or a tool-owned row is
    # its own conversation, and resuming INTO one parks the user's chat in a row the sidebar never lists (#115092).
    children = {}
    for edge in store.sessions.descendants(sid):
        if edge.session_id and edge.parent_session_id:
            children.setdefault(edge.parent_session_id, []).append(edge)

    def started(edge):
        try:
            return float(edge.started_at or 0)
        except Exception:
            return 0.0

    current = sid
    path = [sid]
    seen = {sid}
    while children.get(current):
        candidates = [e for e in children[current] if e.session_id not in seen]
        if not candidates:
            break
        candidates.sort(key=started, reverse=True)
        current = candidates[0].session_id
        path.append(current)
        seen.add(current)
    return current, path


def _session_mutation_context(request, profile):
    """Use the same principal and profile boundary for prepare and apply."""
    from fastapi import HTTPException
    from gateway.session_contract import SessionActor
    from hermes_cli.web_server import _has_valid_session_token

    native = getattr(request.state, 'native_http_principal', None)
    session = getattr(request.state, 'session', None)
    if native is not None:
        subject = native['subject']
    elif session is not None:
        from gateway.owner_identity import principal_subject
        subject = principal_subject({"user_id": session.user_id,
            "provider": session.provider, "issuer": session.issuer})
    elif not getattr(request.app.state, 'auth_required', False) and _has_valid_session_token(request):
        from gateway.owner_identity import host_owner_subject
        subject = host_owner_subject()
    else:
        raise HTTPException(status_code=401, detail='Unauthorized')
    authority = getattr(request.app.state, 'session_authority', None)
    home = _session_db_path_for_profile(profile).parent
    capabilities = frozenset({'session:read', 'session:control', 'session:create', 'session:operator'})
    if authority is None:
        if getattr(request.app.state, 'gateway_runner', None) is not None:
            raise HTTPException(status_code=503, detail='session_authority_unavailable')
        # Standalone ``serve``/``dashboard`` (no composed gateway; its owner cutover is a
        # follow-up): the store is the owner. Same principal id form as a gateway authority's
        # (``str(home)``), so import bindings written offline stay valid once one starts.
        return None, SessionActor(subject, str(home.resolve()), capabilities, 'http')
    # ``?profile=`` selects a served home; its own authority (not the launch one) owns the mutation.
    from gateway.session_authorities import authority_for_home
    runner = getattr(request.app.state, 'gateway_runner', None)
    owner = authority_for_home(runner, home) if runner is not None else None
    if owner is None and home.resolve() == authority.persistence.scope.profile_dir:
        owner = authority
    if owner is None:
        raise HTTPException(status_code=403, detail='profile_mismatch')
    authority = owner
    if native is not None and native['profile_id'] != authority.profile_id:
        raise HTTPException(status_code=403, detail='profile_mismatch')
    if native is not None:
        # One transport per verified native request, so the grant's own profile can claim its old
        # local history exactly as over the native socket (``require_history_claim``); a shared
        # 'http' id never can.
        import uuid
        return authority, SessionActor(subject, authority.profile_id, capabilities, 'native-http:' + uuid.uuid4().hex)
    return authority, SessionActor(subject, authority.profile_id, capabilities, 'http')


@contextmanager
def _native_history_transport(request, authority, actor):
    """Bind a verified native HTTP actor as a native legacy transport for one mutation."""
    native = None
    if authority is not None and str(getattr(actor, 'transport_id', '')).startswith('native-http:'):
        native = getattr(request.state, 'native_http_principal', None)
    if native is None:
        yield
        return
    from gateway.session_local_migration import bind_native_transport, unbind_native_transport
    bind_native_transport(authority, actor, {'provider': 'local', 'profile_id': authority.profile_id,
                                             'instance_id': native['instance_id']})
    try:
        yield
    finally:
        unbind_native_transport(authority, actor)


def _mutate_offline_store(profile, actor, params):
    """Standalone edit: the receipt transaction on this process's single shared writer.

    Runs under the exact owner lock (409 while any gateway owns the store, never a
    second writer beside it); revision/generation/idle guards are the gateway's own.
    """
    from fastapi import HTTPException
    from gateway.session_ingress_media import native_inputs_root
    from persistence.authority.runtime import HistoryActor, ImportControl, parse_mutation
    from session_retention import emit_session_deleted
    from session_rules.imports import validate_import_payload
    from gateway.session_persistence import owner_rule
    from session_rules.legacy_local import login_label

    def run():
        store = _open_session_db_for_profile(profile, read_only=False)
        try:
            history = None
            if params['operation'] == 'import':
                normalized, errors = validate_import_payload(params['payload']['sessions'])
                if errors:
                    raise HTTPException(status_code=400, detail={'errors': errors})
                history = ImportControl(
                    HistoryActor(actor.profile_id, actor.subject, login_label(actor.subject),
                                 'session:create' in actor.capabilities),
                    tuple(item['session']['id'] for item in normalized))
            epoch = store.runtime.current_epoch()
            if epoch is None:
                # The reservation held here is the ownership lock begin_epoch requires.
                epoch = store.runtime.begin_epoch(instance_id='dashboard-offline')
            outcome = store.runtime.mutate(
                epoch, principal_id=actor.subject, session_id=params['session_id'],
                request_id=params['request_id'], expected_revision=params['expected_revision'],
                command=parse_mutation(params['operation'], params['payload']),
                expected_generation=params.get('expected_generation'), history=history,
                media_root=str(native_inputs_root()) if params['operation'] == 'delete' else None, owner=owner_rule())
            if outcome.gone is not None:
                emit_session_deleted(store.scope.profile_dir, outcome.gone.session_ids,
                                     reason=outcome.gone.reason, primary=params['session_id'])
            return outcome.result
        finally:
            store.close()
    return _with_session_maintenance(profile, run)


async def _mutate_via_gateway_owner(profile, actor, params):
    """Forward a standalone dashboard mutation to the live gateway owner.

    While a gateway serves the same home, a separately launched ``hermes
    dashboard`` is not the store owner, so the offline path's
    exclusive-maintenance reservation can never win and every write action
    dies on a bare 409 (#138915). The owner exposes the same revision-fenced
    verb over the control socket. With no endpoint the offline store's reservation
    decides: no endpoint does not prove a live owner (a stale bad-mode socket, a parked
    secondary read ``inaccessible`` with the lock free). When the reservation loses and
    discovery read anything but ``absent`` (a control timeout, a profile mismatch, one
    starting or draining), the edit answers 503 with discovery's reason rather than the
    bare 409 (#144141).

    A dial, mint or RPC that dies on an endpoint discovery just identified is
    not owner-less (a slow first dial, or an owner restart between discovery
    and mint): the offline reservation would only lose to that same owner and
    answer with the bare 409 this forward exists to remove, so every transport
    failure surfaces a retryable 503. Only the owner's refusal frame
    (``GatewayRPCError``) is passed on as a verdict.

    The ticket is minted for the dashboard's authenticated principal, not the
    socket UID: ``authorize_history`` compares the actor subject against the
    row's stored owner, so a cold row owned by a dashboard account only takes
    the mutation when the forward acts as that account.
    """
    from fastapi import HTTPException
    from gateway.runtime_ownership import OwnershipConflict
    from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError, connect_gateway
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_cli.gateway_runtime_discovery import DiscoveryError

    home = _session_db_path_for_profile(profile).parent
    # Discovery is read-only and never starts a gateway.
    discovery = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1.5)
    endpoint = discovery.endpoint
    if endpoint is None:
        try:
            return await asyncio.to_thread(_mutate_offline_store, profile, actor, params)
        except HTTPException as exc:
            if discovery.state == 'absent' or not isinstance(exc.__cause__, OwnershipConflict):
                raise
            known = ''.join(f', {value}' for value in (discovery.reason_code, discovery.detail) if value)
            _log.warning('session write for %s: gateway owner not reachable (%s%s)', home, discovery.state, known)
            reason = discovery.reason_code or discovery.state
            raise HTTPException(status_code=503, detail=f'owner_unavailable: {reason}: {exc.detail}') from exc
    if not getattr(endpoint, 'control_home', True):
        # The selected home answered on its own socket (a multiplexer-served profile already
        # names the multiplexer root). Pin it, so the ticket is minted on the socket that just
        # identified as this profile's owner, never on the dashboard launch home's.
        from dataclasses import replace
        endpoint = replace(endpoint, control_home=str(home.expanduser().resolve()))
    call = {key: params[key] for key in
            ('session_id', 'request_id', 'expected_revision', 'operation', 'payload') if key in params}
    if params.get('expected_generation') is not None:
        call['expected_generation'] = params['expected_generation']
    # A sign-in is the machine owner (owner_identity), which is this same-user socket's own peer
    # subject: let the socket mint it natively. The verifier lets a peer NAME only a structured
    # login subject, so forwarding the owner's OS subject explicitly would be refused.
    from gateway.owner_identity import host_owner_subject
    subject = None if actor.subject == host_owner_subject() else actor.subject
    try:
        async with connect_gateway(endpoint=endpoint, subject=subject) as client:
            return await client.rpc('session.mutate', **call)
    except GatewayRPCError:
        # The owner's own refusal frame: a verdict the caller maps to the direct path's status.
        raise
    except (GatewayClientError, DiscoveryError, OSError, TimeoutError) as exc:
        # Transport, not a verdict: a refused WebSocket dial (wrapped as GatewayClientError), a
        # control socket that refuses the mint after a restart (DiscoveryError), a raw socket error
        # or a timeout. Retrying is safe because the request reuses its request_id.
        raise HTTPException(status_code=503, detail='owner_unreachable') from exc


# One reason -> HTTP status map for a session mutation refusal, shared by the direct path
# (AuthorityRefusal) and the owner forward (GatewayClientError): a storage failure is a
# server fault (503), never a session conflict.
_MUTATION_REFUSAL_STATUS = {'permission_denied': 403, 'profile_mismatch': 403, 'not_found': 404,
                            'invalid_params': 400, 'runtime_draining': 503, 'storage_unavailable': 503}


async def _mutate_session_request(request, profile, session_id, *, request_id,
                                  expected_revision, operation, payload, expected_generation=None):
    """Resolve the authenticated principal and owner; never open a second writer."""
    import sqlite3
    from fastapi import HTTPException
    from gateway.session_contract import SessionRef
    from gateway.session_mutations import mutate_session
    from hermes_cli.gateway_client import GatewayClientError
    from persistence.core.errors import (
        MUTATION_GUARD_REFUSALS, AuthorityRefusal, PermanentStoreError, StoreUnavailable, TransientStoreError,
    )
    from session_rules.imports import validate_import_payload

    authority, actor = _session_mutation_context(request, profile)
    if operation == 'import' and authority is not None:
        _, errors = validate_import_payload(payload['sessions'])
        if errors:
            raise HTTPException(status_code=400, detail={'errors': errors})
    if request_id is None and expected_revision is None and expected_generation is None:
        # A pre-fence client (Desktop <= 0.22.2) sends none of the identity. The owner never
        # commits an unfenced write, so name the upgrade; the wording is what those builds'
        # toast summariser maps to "different versions, update" (#139742).
        raise HTTPException(status_code=409, detail=CLIENT_UPGRADE_REQUIRED)
    if request_id is None or expected_revision is None:
        raise HTTPException(status_code=409, detail='mutation_identity_required')
    params = dict(session_id=session_id, request_id=request_id, expected_revision=expected_revision,
        operation=operation, payload=payload)
    if expected_generation is not None:
        params['expected_generation'] = expected_generation
    try:
        if authority is None:
            # A live gateway owner answers first; a home without an endpoint takes
            # the offline store (whose reservation loses to any owner holding the lock).
            result = await _mutate_via_gateway_owner(profile, actor, params)
        else:
            with _native_history_transport(request, authority, actor):
                result = await mutate_session(authority, actor, SessionRef(authority.profile_id, session_id), params)
        return {'ok': True, **result}
    except AuthorityRefusal as exc:
        # The offline store and a composed authority's session DB refuse with the same class.
        raise HTTPException(status_code=_MUTATION_REFUSAL_STATUS.get(exc.reason, 409), detail=exc.reason) from exc
    except GatewayClientError as exc:
        # The owner's own refusal verdicts (transport failures already became 503 above),
        # wire-shaped like AuthorityRefusal's reasons, so a forwarded write answers as the direct path.
        raise HTTPException(status_code=_MUTATION_REFUSAL_STATUS.get(str(exc), 409), detail=str(exc)) from exc
    except MUTATION_GUARD_REFUSALS as exc:
        raise HTTPException(status_code=409, detail='session_busy') from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (sqlite3.Error, TransientStoreError, PermanentStoreError, StoreUnavailable) as exc:
        raise HTTPException(status_code=503, detail='storage_unavailable') from exc


def _with_session_maintenance(profile, operation, *args):
    """Offline bulk actions reserve the exact owner lock through final DB close."""
    from fastapi import HTTPException
    from gateway.runtime_ownership import OwnershipConflict, exclusive_maintenance

    home = _session_db_path_for_profile(profile).parent
    try:
        with exclusive_maintenance([home]):
            return operation(*args)
    except OwnershipConflict as exc:
        # uvicorn runs at log_level=warning, so a handled 4xx leaves no server-side trace and the
        # refusal reads as a UI bug (#138915). Bulk store maintenance has no owner verb: name it.
        _log.warning('session write refused for %s: %s', home, exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc


_SCHEMA_UNAVAILABLE = (
    "Session store schema is unavailable or corrupt. Start or restart this profile's "
    "gateway to reconcile it; if it persists, run `hermes doctor` for diagnosis. "
    "Browsing does not repair the store.")


def _open_session_db_at_path(db_path: Path, *, read_only: bool):
    """A storage interface handle on ``db_path``, which the caller closes. Browsing never initializes or repairs
    the owner's session store.

    Owner startup reconciles schema. The one write a browse may do: a store that lags
    SCHEMA_SQL (missing table/column, not corruption) with NO gateway owning it gets the
    missing declared tables/columns once, in one short locked transaction (#140315); an
    owned store is refused by name instead. Explicit mutation APIs get the profile's
    shared writer.
    """
    import sqlite3
    from contextlib import ExitStack

    from fastapi import HTTPException
    from hermes_cli.web_routers._common import is_transient_store_error
    from persistence.core.errors import CorruptStoreError, DeclarativeDrift, PersistenceError, StoreReplaced
    from persistence_assembly import open_persistence, open_persistence_reader, open_store_admin, profile_scope
    from storage_errors import is_malformed_db_error

    if not read_only:
        return open_persistence(profile_scope(home=db_path.parent))

    try:
        initialized = db_path.stat().st_size > 0
    except FileNotFoundError:
        initialized = False
    if not initialized:
        raise HTTPException(
            status_code=503,
            detail="Session store is not initialized. Start this profile's gateway to initialize it.")

    try:
        admin = open_store_admin(db_path)
        if admin.schema_drift_reason() is not None:
            # A missing declared table/column is upgrade drift, not corruption (#140315): catch
            # an unowned store up once under its runtime lock, refuse by name when a gateway
            # owns it. Never destructive: additive DDL only, no repair from browse.
            try:
                open_store_admin(db_path, read_only=False).reconcile_drift_for_browse()
            except DeclarativeDrift as drift:
                _log.warning("session store at %s is behind SCHEMA_SQL: %s", db_path, drift)
                raise HTTPException(status_code=503, detail=str(drift)) from drift
            if admin.schema_drift_reason() is not None:
                raise HTTPException(status_code=503, detail=_SCHEMA_UNAVAILABLE)
        context = ExitStack()
        store = context.enter_context(open_persistence_reader(db_path))
        # The reader stays open while its handle lives; the handle's ``close`` releases it.
        store._context = context
        return store
    except StoreReplaced:
        raise
    except (PersistenceError, UnicodeDecodeError) as exc:
        cause = exc.__cause__
        if isinstance(exc, CorruptStoreError) and (
                not isinstance(cause, sqlite3.DatabaseError) or is_malformed_db_error(cause)):
            # Same structured 503 the analytics routers publish (``corrupt_store_as_status``): a
            # corrupt image names the repair (`hermes doctor`), never a generic "schema unavailable".
            from hermes_cli.web_routers._common import corrupt_store_status
            raise corrupt_store_status(db_path, exc) from exc
        if is_transient_store_error(exc):
            detail = "Session store is busy (disk I/O or lock). Retry; the list was not cleared."
        else:
            detail = _SCHEMA_UNAVAILABLE
        raise HTTPException(status_code=503, detail=detail) from exc


def _session_db_path_for_profile(profile: Optional[str]) -> Path:
    """state.db path for ``profile`` (None/empty = this process's own)."""
    from hermes_cli.web_server_cron import _cron_profile_home
    from hermes_constants import get_hermes_home

    if profile:
        _name, home = _cron_profile_home(profile)
        return Path(home) / "state.db"
    return get_hermes_home() / "state.db"


def _open_session_db_for_profile(profile: Optional[str], *, read_only: bool):
    """A storage interface handle on ``profile``'s store (None/empty = this process's own state.db).

    Access-mode semantics: see :func:`_open_session_db_at_path`.
    """
    return _open_session_db_at_path(_session_db_path_for_profile(profile), read_only=read_only)


def _listing_filters(*, source=None, sources=None, exclude_sources=None, cwd_prefix=None, min_message_count: int = 0,
                     include_archived: bool = False, archived_only: bool = False,
                     include_subagents: bool = False) -> dict:
    """The filter fields ``SessionQuery`` and ``SessionCountQuery`` share for the dashboard's listing filters;
    ``source`` wins over ``sources``."""
    return dict(
        sources=(source,) if source else (tuple(sources) if sources else None),
        exclude_sources=tuple(exclude_sources or ()), cwd_prefix=cwd_prefix, min_message_count=min_message_count,
        include_archived=include_archived, archived_only=archived_only, include_subagents=include_subagents)


def _session_rows(store, *, limit: int, offset: int = 0, order_by_last_active: bool = False,
                  compact_rows: bool = False, include_pinned: bool = False, **filters) -> list:
    """The listing's row dicts on the storage handle ``store`` for :func:`_listing_filters`' ``filters``."""
    from persistence.conversation.sessions import SessionOrder, SessionQuery, SessionWindow

    query = SessionQuery(**_listing_filters(**filters), include_pinned=include_pinned,
                         order=SessionOrder.LAST_ACTIVE if order_by_last_active else SessionOrder.STARTED_AT)
    entries = store.sessions.list_sessions(query, SessionWindow(limit=limit, offset=offset), compact_rows=compact_rows)
    return [entry.as_list_row() for entry in entries]


# In-process throttle for the opportunistic auto-archive trigger, keyed by
# profile: bounds the config.yaml read to once per window; the sweep itself is
# throttled far more coarsely by state_meta (sessions.min_interval_hours).
_AUTO_ARCHIVE_CHECK_INTERVAL_S = 300.0
_last_auto_archive_check: dict[str, float] = {}


def _maybe_auto_archive_for_profile(profile: Optional[str]) -> None:
    """Config-gated owner-lifetime maintenance, never invoked by browsing.

    The standalone serve ticker maintains its own profile; a composed HTTP
    listener relies on its gateway owner's startup and housekeeping hooks.
    """
    try:
        key = profile or ""
        now = time.monotonic()
        last = _last_auto_archive_check.get(key)
        if last is not None and now - last < _AUTO_ARCHIVE_CHECK_INTERVAL_S:
            return
        _last_auto_archive_check[key] = now

        from hermes_cli.config import load_config as _load_full_config
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        # The config that governs a store is the one in that store's OWN home. A zero-arg
        # load_config() resolves through the PROCESS HERMES_HOME, so the dashboard swept every
        # profile's sessions with the launch profile's sessions.auto_archive/auto_archive_days —
        # one profile's retention silently decided another's.
        profile_home = _session_db_path_for_profile(profile).parent
        _home_token = set_hermes_home_override(str(profile_home))
        try:
            cfg = (_load_full_config().get("sessions") or {})
        finally:
            reset_hermes_home_override(_home_token)
        if not cfg.get("auto_archive", False):
            return
        from hermes_cli.profiles import _check_gateway_running

        # A live gateway owns this profile's store and runs the same sweep on its own
        # housekeeping tick ("state.db maintenance tick" in gateway/run.py, profile-scoped so a
        # multiplexed secondary's store is swept too). Opening it WRITABLE from `hermes
        # serve` adds a second writer to a database another process is already archiving,
        # for zero extra coverage (#110405). `_check_gateway_running` is the canonical
        # per-profile predicate (`_maybe_run_skill_maintenance` below uses it): its
        # multiplexer rung catches a served secondary, which owns no gateway.pid or lock
        # of its own and a bare lock-file probe would report stopped.
        if _check_gateway_running(profile_home):
            return
        from persistence_assembly import open_persistence, profile_scope
        from session_retention import run_auto_archive

        store = open_persistence(profile_scope(home=profile_home))
        try:
            run_auto_archive(
                store, idle_days=float(cfg.get("auto_archive_days", 3)),
                min_interval_hours=int(cfg.get("min_interval_hours", 24)))
        finally:
            store.close()
    except Exception as exc:
        _log.debug("opportunistic auto-archive skipped: %s", exc)


def _skill_maintenance_idle_for(started_at: float) -> Optional[float]:
    """Measure chat inactivity, not socket inactivity (Desktop stays connected)."""
    import tui_gateway.server as gateway

    from hermes_constants import get_hermes_home

    home = get_hermes_home().resolve()
    with gateway._sessions_lock:
        sessions = [session for session in gateway._sessions.values()
                    if Path(session.get("profile_home") or home).resolve() == home]
        if any(session.get("running") for session in sessions):
            return None
        last_active = max(
            [started_at, gateway._closed_session_activity.get(str(home), 0)]
            + [float(session.get("last_active") or started_at) for session in sessions])
    return max(0.0, time.time() - last_active)


def _maybe_run_skill_maintenance(started_at: float) -> None:
    from hermes_constants import get_hermes_home
    from hermes_cli.profiles import _check_gateway_running

    # A live messaging gateway already owns these chores for this profile.
    if _check_gateway_running(get_hermes_home()):
        return

    from agent.curator import maybe_run_curator

    try:
        idle_for = _skill_maintenance_idle_for(started_at)
        if idle_for is not None:
            maybe_run_curator(idle_for_seconds=idle_for)
    except Exception as exc:
        _log.debug("serve curator tick skipped: %s", exc)


async def _auto_archive_ticker_loop(
    interval_s: float = 3600.0, initial_delay_s: float = 90.0) -> None:
    """Poll maintenance for this serve profile, including Desktop-only installs.

    Individual chores own their config/interval gates. Curator additionally uses
    real chat inactivity; merely keeping a Desktop WebSocket open is not activity.
    """
    started_at = time.time()

    def _sweep() -> None:
        _maybe_auto_archive_for_profile(None)
        _maybe_run_skill_maintenance(started_at)

    await asyncio.sleep(initial_delay_s)
    while True:
        try:
            await asyncio.to_thread(_sweep)
        except Exception as exc:
            _log.debug("auto-archive tick skipped: %s", exc)
        await asyncio.sleep(interval_s)
