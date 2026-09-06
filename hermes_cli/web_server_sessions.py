"""Session-DB access for the dashboard: per-profile SessionDB opening with schema
heal, latest-descendant lookup and the auto-archive ticker.
"""

import logging
import asyncio
import threading
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


# Serialises the one-time writable schema bootstrap for read-only opens, so
# concurrent first-load polls don't open mode=ro against a half-written schema
# ("no such table: sessions").
_session_db_bootstrap_lock = threading.Lock()


def _session_db_read_probe_statements() -> tuple:
    """Stale-schema probes for read-only opens (which skip _reconcile_columns()).
    Derived from SCHEMA_SQL so a new column is probed automatically — a
    hand-written list once went stale and emptied the sidebar after update."""
    from hermes_state_schema import schema_read_probe_statements

    return schema_read_probe_statements()


# Stores where a heal WRITABLE OPEN SUCCEEDED but the read probe still failed:
# one reconciliation cannot fix them (e.g. a NOT-NULL-without-default column),
# so they fall back to the raw read-only open until restart instead of paying
# a writable init per poll. A FAILED writable open (transient lock) is NOT
# recorded — the next poll retries the heal.
_session_db_heal_exhausted: set = set()

# Deduplicates the heal-failure warning per store per process.
_session_db_heal_warned: set = set()


def _is_stale_schema_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "no such table" in message or "no such column" in message


def _open_session_db_at_path(db_path: Path, *, read_only: bool):
    """Open a SessionDB at an explicit path with an explicit access mode.

    Read-only opens bootstrap a missing/zero-byte store once and heal a stale or
    malformed schema through ONE writable open before reopening read-only; the
    healthy read path never takes a write lock.  Tables outside SCHEMA_SQL
    (telemetry ``tel_*``, FTS shadow tables) are outside both probe and heal.
    """
    import sqlite3

    from hermes_state import SessionDB, is_malformed_schema_error

    # Read-only file/sidecar preflight (port of kilocode#12508): repair-or-refuse BEFORE the first
    # connection so users get an actionable message instead of an opaque "attempt to write a readonly
    # database" from deep inside _init_schema.
    if not read_only:
        return SessionDB(db_path=db_path, read_only=False)

    def _needs_bootstrap() -> bool:
        try:
            return db_path.stat().st_size == 0
        except FileNotFoundError:
            return True
        except OSError:
            return False

    if _needs_bootstrap():
        with _session_db_bootstrap_lock:
            if _needs_bootstrap():
                SessionDB(db_path=db_path, read_only=False).close()

    def _open_probed():
        db = SessionDB(db_path=db_path, read_only=True)
        # Unit-test fakes may replace SessionDB without exposing a raw
        # connection. Probe only real connections.
        conn = getattr(db, "_conn", None)
        if conn is not None and str(db_path) not in _session_db_heal_exhausted:
            try:
                for statement in _session_db_read_probe_statements():
                    conn.execute(statement).fetchone()
            except BaseException:
                db.close()
                raise
        return db

    try:
        return _open_probed()
    except (sqlite3.DatabaseError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError = pysqlite could not decode SQLite's own error
        # message because corrupt file bytes were embedded in it; the
        # one-writable-open heal is the only repair path, so treat it as
        # malformed schema.
        if not (
            _is_stale_schema_error(exc)
            or is_malformed_schema_error(exc)
            or isinstance(exc, UnicodeDecodeError)):
            raise
        SessionDB(db_path=db_path, read_only=False).close()
        try:
            return _open_probed()
        except (sqlite3.DatabaseError, UnicodeDecodeError) as still_stale:
            if not _is_stale_schema_error(still_stale):
                raise
            # Writable open succeeded but the store is STILL behind the probe:
            # serve reads without the probe (only queries touching the broken
            # part fail) and stop paying the writable init per poll.
            _session_db_heal_exhausted.add(str(db_path))
            if str(db_path) not in _session_db_heal_warned:
                _session_db_heal_warned.add(str(db_path))
                _log.warning(
                    "state.db at %s is missing schema that a writable "
                    "reconcile could not add (%s); read paths may partially "
                    "fail until the store is repaired",
                    db_path,
                    still_stale)
            return _open_probed()


def _open_session_db_for_profile(profile: Optional[str], *, read_only: bool):
    """Open a SessionDB for ``profile`` (None/empty = this process's own state.db).

    Access-mode semantics: see :func:`_open_session_db_at_path`.
    """
    from hermes_cli.web_server_cron import _cron_profile_home
    from hermes_state import _default_db_path

    if profile:
        _name, home = _cron_profile_home(profile)
        db_path = Path(home) / "state.db"
    else:
        db_path = Path(_default_db_path())
    return _open_session_db_at_path(db_path, read_only=read_only)


# In-process throttle for the opportunistic auto-archive trigger, keyed by
# profile: bounds the config.yaml read to once per window; the sweep itself is
# throttled far more coarsely by state_meta (sessions.min_interval_hours).
_AUTO_ARCHIVE_CHECK_INTERVAL_S = 300.0
_last_auto_archive_check: Dict[str, float] = {}


def _maybe_auto_archive_for_profile(profile: Optional[str]) -> None:
    """Config-gated stale-session auto-archive for ``profile``; never raises.
    ``hermes serve`` runs neither CLI nor gateway startup hooks, so this
    session-list trigger is what makes ``sessions.auto_archive`` work there."""
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


async def _auto_archive_ticker_loop(
    interval_s: float = 3600.0, initial_delay_s: float = 90.0) -> None:
    """Poll-rate timer for the auto-archive sweep (primary profile), so a
    long-idle Desktop keeps sweeping without any ``/api/sessions`` request.
    The real cadence is still owned by state_meta inside ``maybe_auto_archive``."""

    def _sweep() -> None:
        _maybe_auto_archive_for_profile(None)

    await asyncio.sleep(initial_delay_s)
    while True:
        try:
            await asyncio.to_thread(_sweep)
        except Exception as exc:
            _log.debug("auto-archive tick skipped: %s", exc)
        await asyncio.sleep(interval_s)
