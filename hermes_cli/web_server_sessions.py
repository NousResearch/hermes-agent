"""Read-only session browsing, latest-descendant lookup and owner maintenance.
"""

import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, Optional

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")

_DESCENDANTS_SQL = """
            WITH RECURSIVE descendants(id, parent_session_id, started_at) AS (
                SELECT id, parent_session_id, started_at FROM sessions WHERE id = ?
                UNION
                SELECT s.id, s.parent_session_id, s.started_at
                FROM sessions s
                JOIN descendants d ON s.parent_session_id = d.id
            )
            SELECT id, parent_session_id, started_at FROM descendants
            """


def _session_latest_descendant(session_id: str, db):
    """Resolve a session id to the newest child leaf session.

    /model may create child sessions; a dashboard refresh should continue the
    newest child instead of reopening the old parent. Returns ``(leaf, path)``.
    """
    sid = db.resolve_session_id(session_id)
    if not sid or not db.get_session(sid):
        return None, []

    conn = getattr(db, "_conn", None)
    if conn is not None:
        keys = ("id", "parent_session_id", "started_at")
        rows = [dict(zip(keys, row)) for row in conn.execute(_DESCENDANTS_SQL, (sid,)).fetchall()]
    else:
        rows = db.list_sessions_rich(limit=10000, offset=0, compact_rows=True)

    children = {}
    for row in rows:
        rid = row.get("id")
        parent = row.get("parent_session_id")
        if rid and parent:
            children.setdefault(parent, []).append(row)

    def started(row):
        try:
            return float(row.get("started_at") or 0)
        except Exception:
            return 0.0

    current = sid
    path = [sid]
    seen = {sid}
    while children.get(current):
        candidates = [r for r in children[current] if r.get("id") not in seen]
        if not candidates:
            break
        candidates.sort(key=started, reverse=True)
        current = candidates[0]["id"]
        path.append(current)
        seen.add(current)
    return current, path


def _session_mutation_context(request, profile):
    """Use the same principal and profile boundary for prepare and apply."""
    from fastapi import HTTPException
    from gateway.session_contract import Principal
    from hermes_cli.web_server import _has_valid_session_token
    from hermes_cli.web_server_cron import _cron_profile_home
    from hermes_state import _default_db_path

    native = getattr(request.state, 'native_http_principal', None)
    session = getattr(request.state, 'session', None)
    if native is not None:
        subject = native['subject']
    elif session is not None:
        from gateway.session_identity import authenticated_subject
        subject = authenticated_subject({"user_id": session.user_id,
            "provider": session.provider, "issuer": session.issuer})
    elif not getattr(request.app.state, 'auth_required', False) and _has_valid_session_token(request):
        from gateway.session_identity import authenticated_subject
        subject = authenticated_subject({'user_id': 'legacy-token-owner', 'provider': 'session-token'})
    else:
        raise HTTPException(status_code=401, detail='Unauthorized')
    authority = getattr(request.app.state, 'session_authority', None)
    if authority is None:
        raise HTTPException(status_code=503, detail='session_authority_unavailable')
    home = Path(_cron_profile_home(profile)[1]) if profile else Path(_default_db_path()).parent
    # ``?profile=`` selects a served home; its own authority (not the launch one) owns the mutation.
    from gateway.session_authorities import authority_for_home
    runner = getattr(request.app.state, 'gateway_runner', None)
    owner = authority_for_home(runner, home) if runner is not None else None
    if owner is None and home.resolve() == Path(authority.db.db_path).parent.resolve():
        owner = authority
    if owner is None:
        raise HTTPException(status_code=403, detail='profile_mismatch')
    authority = owner
    if native is not None and native['profile_id'] != authority.profile_id:
        raise HTTPException(status_code=403, detail='profile_mismatch')
    return authority, Principal(subject, authority.profile_id,
        frozenset({'session:read', 'session:control', 'session:create', 'session:operator'}), 'http')


async def _mutate_session_request(request, profile, session_id, *, request_id,
                                  expected_revision, operation, payload, expected_generation=None):
    """Resolve the authenticated principal and owner; never open a second writer."""
    import sqlite3
    from fastapi import HTTPException
    from gateway.session_contract import SessionRef
    from gateway.session_mutations import mutate_session
    from hermes_state_runtime import RuntimeStoreError

    authority, actor = _session_mutation_context(request, profile)
    if operation == 'import':
        _, errors = authority.db._validate_import_payload(payload['sessions'])
        if errors:
            raise HTTPException(status_code=400, detail={'errors': errors})
    if request_id is None or expected_revision is None:
        raise HTTPException(status_code=409, detail='mutation_identity_required')
    params = dict(session_id=session_id, request_id=request_id, expected_revision=expected_revision,
        operation=operation, payload=payload)
    if expected_generation is not None:
        params['expected_generation'] = expected_generation
    try:
        result = await mutate_session(authority, actor, SessionRef(authority.profile_id, session_id), params)
        return {'ok': True, **result}
    except RuntimeStoreError as exc:
        status = {'permission_denied': 403, 'profile_mismatch': 403, 'not_found': 404,
                  'invalid_params': 400, 'runtime_draining': 503}.get(exc.reason, 409)
        raise HTTPException(status_code=status, detail=exc.reason) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail='storage_unavailable') from exc


def _with_session_maintenance(profile, operation, *args):
    """Offline bulk actions reserve the exact owner lock through final DB close."""
    from fastapi import HTTPException
    from gateway.runtime_ownership import OwnershipConflict, exclusive_maintenance
    from hermes_cli.web_server_cron import _cron_profile_home
    from hermes_state import _default_db_path

    home = Path(_cron_profile_home(profile)[1]) if profile else Path(_default_db_path()).parent
    try:
        with exclusive_maintenance([home]):
            return operation(*args)
    except OwnershipConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _session_db_read_probe_statements() -> tuple:
    """Probe the declared schema without reconciling it on a browsing request."""
    from hermes_state_schema import schema_read_probe_statements

    return schema_read_probe_statements()


def _open_session_db_at_path(db_path: Path, *, read_only: bool):
    """Browsing never initializes, repairs or upgrades the owner's session store.

    Owner startup reconciles schema; explicit mutation APIs retain their writable
    acquisition until their separate owner-RPC migration.
    """
    import sqlite3

    from fastapi import HTTPException
    from hermes_state import SessionDB
    from hermes_state_errors import is_transient_sqlite_error
    from hermes_state_registry import acquire

    if not read_only:
        return acquire(db_path)

    try:
        initialized = db_path.stat().st_size > 0
    except FileNotFoundError:
        initialized = False
    if not initialized:
        raise HTTPException(
            status_code=503,
            detail="Session store is not initialized. Start this profile's gateway to initialize it.")

    try:
        db = SessionDB(db_path=db_path, read_only=True)
        try:
            conn = getattr(db, "_conn", None)
            if conn is not None:
                for statement in _session_db_read_probe_statements():
                    conn.execute(statement).fetchone()
            return db
        except BaseException:
            db.close()
            raise
    except (sqlite3.DatabaseError, UnicodeDecodeError) as exc:
        from hermes_state_errors import is_malformed_db_error
        if isinstance(exc, sqlite3.DatabaseError) and is_malformed_db_error(exc):
            # Same structured 503 the analytics routers publish (``corrupt_store_as_status``): a
            # corrupt image names the repair (`hermes doctor`), never a generic "schema unavailable".
            from hermes_cli.web_routers._common import corrupt_store_status
            raise corrupt_store_status(db_path, exc) from exc
        if isinstance(exc, sqlite3.OperationalError) and is_transient_sqlite_error(exc):
            detail = "Session store is busy (disk I/O or lock). Retry; the list was not cleared."
        else:
            detail = (
                "Session store schema is unavailable or corrupt. Start or restart this profile's "
                "gateway to reconcile it; if it persists, run `hermes doctor` for diagnosis. "
                "Browsing does not repair the store.")
        raise HTTPException(status_code=503, detail=detail) from exc


def _session_db_path_for_profile(profile: Optional[str]) -> Path:
    """state.db path for ``profile`` (None/empty = this process's own)."""
    from hermes_cli.web_server_cron import _cron_profile_home
    from hermes_state import _default_db_path

    if profile:
        _name, home = _cron_profile_home(profile)
        return Path(home) / "state.db"
    return Path(_default_db_path())


def _open_session_db_for_profile(profile: Optional[str], *, read_only: bool):
    """Open a SessionDB for ``profile`` (None/empty = this process's own state.db).

    Access-mode semantics: see :func:`_open_session_db_at_path`.
    """
    return _open_session_db_at_path(_session_db_path_for_profile(profile), read_only=read_only)


# In-process throttle for the opportunistic auto-archive trigger, keyed by
# profile: bounds the config.yaml read to once per window; the sweep itself is
# throttled far more coarsely by state_meta (sessions.min_interval_hours).
_AUTO_ARCHIVE_CHECK_INTERVAL_S = 300.0
_last_auto_archive_check: Dict[str, float] = {}


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
        cfg = (_load_full_config().get("sessions") or {})
        if not cfg.get("auto_archive", False):
            return
        db = _open_session_db_for_profile(profile, read_only=False)
        try:
            db.maybe_auto_archive(
                idle_days=float(cfg.get("auto_archive_days", 3)),
                min_interval_hours=int(cfg.get("min_interval_hours", 24)))
        finally:
            db.close()
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
    from tools.skills_sync_client import maybe_pull_skills
    from tools.skills_sync_client_org import maybe_pull_org_skills

    try:
        idle_for = _skill_maintenance_idle_for(started_at)
        if idle_for is not None:
            maybe_run_curator(idle_for_seconds=idle_for)
    except Exception as exc:
        _log.debug("serve curator tick skipped: %s", exc)
    for pull in (maybe_pull_skills, maybe_pull_org_skills):
        try:
            pull()
        except Exception as exc:
            _log.debug("serve skill sync tick skipped: %s", exc)


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
