"""Session dashboard routes.

Three routers because global route order matters: ``list_router`` (GET
/api/sessions) mounts before the profiles ``sessions_router``, ``search_router``
right after it, ``manage_router`` (mutation/detail) much later.  web_server-owned
helpers are reached via the late-binding seam so monkeypatching keeps working.
"""

import asyncio
import json
import re
import sqlite3
import time
from typing import Callable, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from hermes_cli.web_deps import late
from hermes_cli.web_server_gateway import _strip_session_list_rows
from hermes_cli.web_server_sessions import _maybe_auto_archive_for_profile, _session_latest_descendant
from hermes_cli.web_models import (
    BulkDeleteSessions, SessionImport, SessionOwnerBackfill, SessionPrune, SessionRename)
from hermes_cli.web_routers._common import log as _log, http_failure
from hermes_state import is_malformed_db_error
from hermes_state_errors import is_transient_sqlite_error

list_router = APIRouter()
search_router = APIRouter()
manage_router = APIRouter()

_cron_default_profile = late("_cron_default_profile", "hermes_cli.web_server_cron")
_cron_profile_home = late("_cron_profile_home", "hermes_cli.web_server_cron")
_open_session_db_for_profile = late("_open_session_db_for_profile", "hermes_cli.web_server_sessions")

_NOT_FOUND = "Session not found"

# CRITICAL — every literal-path route on ``manage_router`` MUST be declared
# BEFORE the templated ``/api/sessions/{session_id}`` family. Starlette matches
# in registration order and ``{session_id}`` is unconstrained, so e.g.
# ``DELETE /api/sessions/empty`` would otherwise be taken as "the session with
# id 'empty'" (404, or worse, deleting the wrong row). Move the block as a unit.

# Stream-safe import: FastAPI otherwise buffers an arbitrarily large JSON body
# before SessionDB can enforce its own per-session and transaction limits.
_SESSION_IMPORT_MAX_BYTES = 25 * 1024 * 1024


async def _read_session_import_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _SESSION_IMPORT_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Session import payload is too large")
        body.extend(chunk)
    return bytes(body)


# Prune filters forwarded to SessionDB; string filters map "" -> None.
_PRUNE_STR_FILTERS = (
    "source", "title_like", "end_reason", "cwd_prefix", "model_like", "provider",
    "user_id", "chat_id", "chat_type", "branch_like")
_PRUNE_NUM_FILTERS = (
    "min_messages", "max_messages", "min_tokens", "max_tokens", "min_cost", "max_cost",
    "min_tool_calls", "max_tool_calls")


_PRUNE_ROW_KEYS = ("id", "source", "title", "model", "started_at", "last_active", "message_count")


def _prune_sessions(body: SessionPrune):
    """Delete ended sessions matching filters (mirrors `hermes sessions prune`)."""
    from hermes_cli.config import get_hermes_home
    has_window = body.started_before is not None or body.started_after is not None
    if body.older_than_days is not None and body.older_than_days < 1 and not has_window:
        raise HTTPException(status_code=400, detail="older_than_days must be >= 1")
    # Mirror the CLI: the implicit 90-day cutoff only applies to a truly bare
    # prune. Any attribute filter suppresses it unless older_than_days was
    # explicitly sent.
    attr_filters_set = any(
        getattr(body, f) is not None for f in _PRUNE_STR_FILTERS + _PRUNE_NUM_FILTERS)
    effective_older_than = body.older_than_days
    if has_window or (attr_filters_set and "older_than_days" not in body.model_fields_set):
        effective_older_than = None
    profile_home = _cron_profile_home(body.profile)[1] if body.profile else get_hermes_home()
    db = _open_session_db_for_profile(body.profile, read_only=False)
    try:
        filters = {
            "older_than_days": effective_older_than, "started_before": body.started_before,
            "started_after": body.started_after,
            "archived": None if body.include_archived else False,
            **{f: (getattr(body, f) or None) for f in _PRUNE_STR_FILTERS},
            **{f: getattr(body, f) for f in _PRUNE_NUM_FILTERS}}
        skipped_open = db.count_open_prune_matches(**filters)
        if body.dry_run:
            rows = db.list_prune_candidates(**filters)
            return {
                "ok": True,
                "removed": 0,
                "matched": len(rows),
                "skipped_open": skipped_open,
                # Rows are ordered by last activity, not creation time.
                "oldest_last_active": rows[0]["last_active"] if rows else None,
                "newest_last_active": rows[-1]["last_active"] if rows else None,
                "oldest_started_at": min(r["started_at"] for r in rows) if rows else None,
                "newest_started_at": max(r["started_at"] for r in rows) if rows else None,
                "sessions": [{k: r.get(k) for k in _PRUNE_ROW_KEYS} for r in rows]}
        sessions_dir = profile_home / "sessions"
        removed = db.prune_sessions(
            sessions_dir=sessions_dir if sessions_dir.exists() else None, **filters)
        return {"ok": True, "removed": removed, "skipped_open": skipped_open}
    finally:
        db.close()


_ACTIVE_WINDOW_S = 300


def _csv(value: Optional[str]) -> List[str]:
    """Split a comma-separated query param into stripped, non-empty items."""
    return [s.strip() for s in (value or "").split(",") if s.strip()]


def _is_active(row: dict, now: float) -> bool:
    return (
        row.get("ended_at") is None
        and (now - row.get("last_active", row.get("started_at", 0))) < _ACTIVE_WINDOW_S)


def _with_db(profile: Optional[str], fn: Callable, *, read_only: bool):
    """Open the profile's session DB, run ``fn(db)``, always close."""
    db = _open_session_db_for_profile(profile, read_only=read_only)
    try:
        return fn(db)
    finally:
        db.close()


def _serving_profile(profile: Optional[str]) -> str:
    """The profile name rows are stamped with: the requested one, else the
    serving process's own — so default-profile rows never circulate unowned."""
    return _cron_profile_home(profile)[0] if profile else _cron_default_profile()


def _resolve_session_id(db, session_id: str) -> Optional[str]:
    """Resolve *session_id*; a corrupt store (prefix scan raises "malformed") is
    reported as 503 with the actual problem instead of a misleading 404."""
    try:
        return db.resolve_session_id(session_id)
    except sqlite3.DatabaseError as exc:
        if not is_malformed_db_error(exc):
            raise
        _log.error("state.db is corrupt while resolving session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Session store is corrupt (database disk image is malformed). "
                "Sessions cannot be read until it is repaired — run "
                "`hermes doctor` for diagnosis."),
        ) from exc


# ``le=100`` on limit: an unbounded limit lets one request drag every session
# row (plus correlated-subquery preview work) out of SQLite in a single hit.
@list_router.get("/api/sessions")
def get_sessions(
    limit: int = Query(20, ge=0, le=100), offset: int = Query(0, ge=0), min_messages: int = 0,
    archived: str = "exclude", order: str = "created", source: str = None, sources: str = None,
    exclude_sources: str = None, cwd_prefix: str = None, full: bool = False,
    profile: Optional[str] = None):
    """List sessions.

    ``order=recent`` sorts by latest activity across the compression chain, so
    a long-running chat stays on page one after it auto-compresses onto a fresh
    id.  Rows omit ``system_prompt`` / ``model_config`` unless ``full=1``.
    """
    if archived not in ("exclude", "only", "include"):
        raise HTTPException(
            status_code=400, detail="archived must be one of: exclude, only, include")
    if order not in ("created", "recent"):
        raise HTTPException(status_code=400, detail="order must be one of: created, recent")
    profile_name = _cron_profile_home(profile)[0] if profile else None
    try:
        # Auto-archive is the only write on this GET path: run it on its own
        # maintenance connection, then open the listing connection read-only.
        _maybe_auto_archive_for_profile(profile)
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            min_message_count = max(0, min_messages)
            archived_only = archived == "only"
            include_archived = archived == "include"
            # Source scoping: the desktop splits recents (exclude=cron) from
            # the cron-jobs section (source=cron) into two independent lists.
            source_list = _csv(sources)
            exclude_list = _csv(exclude_sources)
            scope = dict(
                source=source or None, sources=source_list or None,
                exclude_sources=exclude_list or None, cwd_prefix=(cwd_prefix or None),
                min_message_count=min_message_count, include_archived=include_archived,
                archived_only=archived_only)
            sessions = db.list_sessions_rich(
                limit=limit,
                offset=offset,
                order_by_last_active=order == "recent",
                # Skip the system_prompt blob inside SQLite too (pairs with
                # _strip_session_list_rows below).
                compact_rows=not full,
                include_pinned=True,
                **scope)
            if sessions:
                roots = [s.get("_lineage_root_id") or s["id"] for s in sessions]
                revisions = db.get_display_revisions(roots)
                for session, root_id in zip(sessions, roots):
                    session["display_revision"] = revisions[root_id]
            total = db.session_count(exclude_children=True, **scope)
            now = time.time()
            row_profile = profile_name or _cron_default_profile()
            for s in sessions:
                s["is_active"] = _is_active(s, now)
                s["profile"] = row_profile
                s["is_default_profile"] = row_profile == "default"
                # SQLite stores the flags as 0/1; expose real JSON booleans.
                s["archived"] = bool(s.get("archived"))
                s["pinned"] = bool(s.get("pinned"))
            if not full:
                _strip_session_list_rows(sessions)
            return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}
        finally:
            db.close()
    except HTTPException:
        raise
    except sqlite3.OperationalError as exc:
        _log.exception("GET /api/sessions failed")
        # 503, not 500: the store is busy, not gone — the desktop keeps its
        # sidebar instead of reading a 500 as an authoritative empty list.
        transient = is_transient_sqlite_error(exc)
        raise HTTPException(
            status_code=503 if transient else 500,
            detail=(
                "Session store is busy (disk I/O or lock). Retry; the list was not cleared."
                if transient
                else "Internal server error"),
        ) from exc
    except Exception:
        _log.exception("GET /api/sessions failed")
        raise HTTPException(status_code=500, detail="Internal server error")


def _is_compression_edge(child: dict, parent: dict) -> bool:
    parent_ended_at = parent.get("ended_at")
    started_at = child.get("started_at")
    return (
        parent.get("end_reason") == "compression"
        and parent_ended_at is not None
        and started_at is not None
        and started_at >= parent_ended_at)


@search_router.get("/api/sessions/search")
async def search_sessions(
    q: str = "", limit: int = 20, profile: Optional[str] = None, source: str = None,
    sources: str = None, exclude_sources: str = None):
    """Search sessions by ID (first) plus FTS5 message content.

    Results are deduped by compression lineage, not raw ``session_id``:
    auto-compression rotates a chat onto a fresh id and leaves the old segment
    in the FTS index.  Branches also use ``parent_session_id`` but are real
    alternate conversations — they are NOT collapsed into the parent.
    """
    if not q or not q.strip():
        return {"results": []}
    with http_failure("GET /api/sessions/search failed", 500, detail="Search failed"):
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            safe_limit = max(1, min(int(limit or 20), 100))
            source_filter = source or None
            source_list = _csv(sources)
            include_sources = [source_filter] if source_filter else (source_list or None)
            exclude_list = _csv(exclude_sources)
            now = time.time()

            def get_session(sid):
                try:
                    return db.get_session(sid)
                except Exception:
                    return None

            # Walk parent_session_id to the compression root, memoized per
            # chain; stops at branch/delegate edges (those stay searchable).
            root_cache: dict = {}

            def compression_root(session_id: str) -> str:
                chain, cur, root = [], session_id, session_id
                while cur and cur not in chain:  # ``not in chain`` guards parent cycles
                    if cur in root_cache:
                        root = root_cache[cur]
                        break
                    chain.append(cur)
                    s = get_session(cur)
                    parent = s.get("parent_session_id") if isinstance(s, dict) else None
                    parent_session = get_session(parent) if parent else None
                    if not parent_session or not _is_compression_edge(s, parent_session):
                        root = cur
                        break
                    cur = parent
                for node in chain:
                    root_cache[node] = root
                return root

            tip_cache: dict = {}

            def lineage_tip(root_id: str) -> str:
                if root_id not in tip_cache:
                    try:
                        tip_cache[root_id] = db.get_compression_tip(root_id) or root_id
                    except Exception:
                        tip_cache[root_id] = root_id
                return tip_cache[root_id]

            # One keyspace for id-hits and content-hits, keyed by lineage root;
            # first hit wins, and ID matches run first.
            seen: dict = {}

            def add_lineage_result(raw_sid: str, payload: dict) -> None:
                if not raw_sid:
                    return
                root = compression_root(raw_sid)
                if root in seen or len(seen) >= safe_limit:
                    return
                payload = dict(payload)
                sid = lineage_tip(root)
                payload["session_id"] = sid
                payload["lineage_root"] = root
                try:
                    row = db.get_session_rich_row(sid)
                except Exception:
                    row = None
                if row:
                    last_active = row.get("last_active") or row.get("started_at")
                    payload.update({
                        "id": row.get("id") or sid,
                        "source": row.get("source"),
                        "model": row.get("model"),
                        "title": row.get("title"),
                        "started_at": row.get("started_at"),
                        "ended_at": row.get("ended_at"),
                        "last_active": last_active,
                        "is_active": (
                            row.get("ended_at") is None and (now - (last_active or 0)) < 300),
                        "message_count": row.get("message_count") or 0,
                        "tool_call_count": row.get("tool_call_count") or 0,
                        "input_tokens": row.get("input_tokens") or 0,
                        "output_tokens": row.get("output_tokens") or 0,
                        "preview": row.get("preview"),
                        "parent_session_id": row.get("parent_session_id"),
                        "archived": bool(row.get("archived"))})
                else:
                    payload["id"] = sid
                seen[root] = payload

            def hit_payload(row: dict, snippet: str, role, session_started) -> dict:
                return {
                    "snippet": snippet, "role": role, "source": row.get("source"),
                    "model": row.get("model"), "session_started": session_started}

            # Direct ID matches first (pasted ids never appear in message text).
            for row in db.search_sessions_by_id(
                q, limit=safe_limit, include_archived=True, source=source_filter,
                sources=source_list or None, exclude_sources=exclude_list or None):
                sid = row.get("id")
                preview = (row.get("preview") or "").strip()
                snippet = preview or f"Session ID: {sid}"
                add_lineage_result(sid, hit_payload(row, snippet, None, row.get("started_at")))

            # Prefix wildcards so partial words match ("nimb" -> "nimb*");
            # quoted phrases and existing wildcards are kept as-is.
            prefix_query = " ".join(
                tok if tok.startswith('"') or tok.endswith("*") else tok + "*"
                for tok in re.findall(r'"[^"]*"|\S+', q.strip()))
            # Over-fetch so lineage dedup can still surface `limit` distinct
            # conversations when several hits collapse onto one root.
            matches = db.search_messages(
                query=prefix_query, source_filter=include_sources,
                exclude_sources=exclude_list or None, limit=max(safe_limit * 5, 50),
                fields=("session_id", "role", "snippet", "source", "model", "session_started"))
            for m in matches:
                if len(seen) >= safe_limit:
                    break
                add_lineage_result(
                    m["session_id"],
                    hit_payload(m, m.get("snippet", ""), m.get("role"), m.get("session_started")))
            return {"results": list(seen.values())}
        finally:
            db.close()


@manage_router.post("/api/sessions/bulk-delete")
async def bulk_delete_sessions_endpoint(body: BulkDeleteSessions):
    """Delete every session in ``body.ids`` in one transaction (POST: many
    clients refuse a DELETE body).

    Per :meth:`SessionDB.delete_sessions`: unknown ids are skipped (``deleted``
    reports what really happened), children are orphaned, active/archived rows
    ARE deleted (hand-picked), on-disk cleanup is left to the next prune.
    """
    # Hard cap so a runaway selection can't lock the writer for long.
    if len(body.ids) > 500:
        raise HTTPException(status_code=400, detail="ids must contain at most 500 entries")
    deleted = await asyncio.to_thread(
        _with_db, body.profile, lambda db: db.delete_sessions(body.ids), read_only=False)
    return {"ok": True, "deleted": deleted}


@manage_router.post("/api/sessions/import")
async def import_sessions_endpoint(request: Request):
    """Import sessions exported from the dashboard or CLI (session rows only —
    ``/api/ops/import`` restores a whole backup archive)."""
    try:
        raw_body = await _read_session_import_body(request)
        body = SessionImport.model_validate_json(raw_body)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid session import payload") from exc

    try:
        result = await asyncio.to_thread(
            _with_db, body.profile, lambda db: db.import_sessions(body.sessions), read_only=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not result.get("ok", False):
        raise HTTPException(status_code=400, detail=result)
    return result


@manage_router.get("/api/sessions/empty/count")
async def count_empty_sessions_endpoint(profile: Optional[str] = None):
    """Count of empty, ended, non-archived sessions (the "Delete empty (N)" button)."""
    count = await asyncio.to_thread(
        _with_db, profile, lambda db: db.count_empty_sessions(), read_only=True)
    return {"count": count}


@manage_router.delete("/api/sessions/empty")
async def delete_empty_sessions_endpoint(profile: Optional[str] = None):
    """Delete every empty, ended, non-archived session in one transaction.

    "Empty" means NO ``messages`` rows at all — a rewound/compacted chat reads
    ``message_count == 0`` while its soft-archived rows are the only transcript
    copy (see :meth:`SessionDB.delete_empty_sessions`).

    * Active sessions are skipped (``ended_at IS NULL``) so a live agent isn't yanked mid-handshake. *
    Archived sessions are skipped — the user explicitly chose to keep those rows. * Children of deleted
    parents are orphaned, not cascade-deleted. See #95868.
    """
    deleted = await asyncio.to_thread(
        _with_db, profile, lambda db: db.delete_empty_sessions(), read_only=False)
    return {"ok": True, "deleted": deleted}


@manage_router.get("/api/sessions/stats")
async def get_session_stats(profile: Optional[str] = None):
    """Session-store statistics (mirrors `hermes sessions stats`)."""
    def _stats(db):
        out = {
            "total": db.session_count(include_archived=True),
            "active_store": db.session_count(include_archived=False),
            "archived": db.session_count(archived_only=True), "messages": db.message_count(),
            "by_source": {}}
        try:
            out["by_source"] = db.session_count_by_source(
                include_archived=True, exclude_children=True)
        except Exception:
            pass
        return out

    return _with_db(profile, _stats, read_only=True)


@manage_router.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str, profile: Optional[str] = None):
    def _detail(db):
        sid = _resolve_session_id(db, session_id)
        session = db.get_session(sid) if sid else None
        if not session:
            raise HTTPException(status_code=404, detail=_NOT_FOUND)
        # Always stamp the owner: unowned default-profile rows made multi-profile
        # clients resolve them to whichever gateway happened to be active.
        session["profile"] = _serving_profile(profile)
        session["is_default_profile"] = session["profile"] == "default"
        return session

    return _with_db(profile, _detail, read_only=True)


@manage_router.get("/api/sessions/{session_id}/latest-descendant")
async def get_session_latest_descendant(session_id: str, profile: Optional[str] = None):
    latest, path = await asyncio.to_thread(
        _with_db, profile, lambda db: _session_latest_descendant(session_id, db), read_only=True)
    if not latest:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return {
        "requested_session_id": path[0] if path else session_id, "session_id": latest, "path": path,
        "changed": bool(path and latest != path[0])}

def _project_for_display(messages: list) -> list:
    """Replace compaction summaries with their display-only projection."""
    from agent.compaction_display import project_compaction_message_for_display
    from agent.context_compressor import is_compaction_summary_message

    projected_messages = []
    for message in messages:
        if not is_compaction_summary_message(message):
            projected_messages.append(message)
            continue
        display_view = project_compaction_message_for_display(message)
        projected = message.copy()
        if display_view is None:
            if not projected.get("display_kind"):
                projected["display_kind"] = "hidden"
        else:
            # Keep the physical content for inspection/export compatibility;
            # Desktop consumes this display-only projection. A legacy hidden
            # wrapper must not hide a successfully recovered live ask.
            projected["display_content"] = display_view.get("content")
            projected.pop("display_kind", None)
        projected_messages.append(projected)
    return projected_messages


@manage_router.get("/api/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    profile: Optional[str] = None,
    limit: Optional[int] = Query(None, ge=0),
    offset: int = Query(0, ge=0),
    order: Optional[str] = Query(None),
    include_compacted: bool = Query(False),
    known_display_revision: Optional[str] = Query(None),
):
    # Keep the established positional function signature intact for internal
    # callers and tests. FastAPI supplies the appended query value as a string;
    # a direct call that omits it leaves the Query default object here.
    raw_known_display_revision = (
        known_display_revision if isinstance(known_display_revision, str) else None
    )
    parsed_known_display_revision: Optional[int] = None
    if raw_known_display_revision is not None:
        valid_revision = (
            bool(raw_known_display_revision)
            and len(raw_known_display_revision) <= 19
            and all("0" <= character <= "9" for character in raw_known_display_revision)
        )
        parsed_revision = int(raw_known_display_revision) if valid_revision else None
        if parsed_revision is None or parsed_revision > (1 << 63) - 1:
            raise HTTPException(
                status_code=400,
                detail="known_display_revision must be a non-negative integer",
            )
        parsed_known_display_revision = parsed_revision

    if order not in (None, "oldest", "latest"):
        raise HTTPException(
            status_code=400,
            detail="order must be one of: oldest, latest",
        )

    def _read(db):
        sid = _resolve_session_id(db, session_id)
        if not sid:
            return None
        default_page = limit is None
        latest_page = order == "latest" or (order is None and default_page)
        _limit = 500 if default_page else min(limit, 500)
        try:
            return db.get_display_message_page(
                sid,
                limit=_limit,
                offset=offset,
                latest=latest_page,
                include_compacted=include_compacted,
                known_display_revision=parsed_known_display_revision,
            )
        except KeyError:
            return None

    result = await asyncio.to_thread(_with_db, profile, _read, read_only=True)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not result["unchanged"]:
        from agent.compaction_display import suppress_redundant_compaction_projection_sources

        display_messages = suppress_redundant_compaction_projection_sources(
            result["messages"]
        )
        result["messages"] = _project_for_display(display_messages)
    return result


@manage_router.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, profile: Optional[str] = None):
    def _delete(db):
        # Already-absent is an idempotent success: the desktop optimistically
        # removes the row and RESTORES it on any error, so a 404 resurrected
        # ghost rows (transient empties racing the sidebar snapshot).
        sid = _resolve_session_id(db, session_id)
        if not sid:
            return {"ok": True, "already_absent": True}
        db.delete_session(sid)
        return {"ok": True}

    return await asyncio.to_thread(_with_db, profile, _delete, read_only=False)


@manage_router.post("/api/sessions/owner-backfill")
async def backfill_session_owner_profiles(body: SessionOwnerBackfill):
    """Stamp legacy ``profile_name = NULL`` rows with the serving-profile identity.

    A multi-connection Desktop fails closed on unowned rows.  Each ``state.db``
    belongs to exactly one profile, so this is a single-match, idempotent
    backfill (non-NULL owners are never overwritten).

    That was fine while one backend served everything, but a Desktop with registry topology (≥2 registered
    connections) fails closed on unowned rows by design — leaving every pre-campaign session unresumable
    with no migration path. Each profile's ``state.db`` belongs to exactly one profile, so stamping that
    store's own name is a single-match backfill, never a guess; the value written is the SAME
    serving-profile identity the list endpoints already stamp onto outgoing rows (``row_profile`` in
    ``get_sessions``). See #95407.
    """
    stamp = _serving_profile(body.profile)

    with http_failure(
        "POST /api/sessions/owner-backfill failed", 500, detail="Internal server error"):
        stamped = await asyncio.to_thread(
            _with_db, body.profile, lambda db: db.backfill_null_session_profiles(stamp),
            read_only=False)

    if stamped:
        _log.info(
            "owner-backfill: stamped %d legacy NULL-profile session row(s) with profile %r",
            stamped, stamp)
    return {"ok": True, "stamped": stamped, "profile": stamp}


# PATCH /api/sessions/{id} flag -> SessionDB setter, applied in this order.
_RENAME_FLAG_SETTERS = (
    ("archived", lambda db, sid, v: db.set_session_archived(sid, v)),
    ("hidden", lambda db, sid, v: db.set_session_hidden(sid, v)),
    ("pinned", lambda db, sid, v: db.set_session_pinned(sid, v)),
    ("unread", lambda db, sid, v: db.set_session_read(sid, read=not v)),
)


@manage_router.patch("/api/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, body: SessionRename):
    """Update ``title`` (empty clears) and/or the flags; ``pinned`` exempts from
    the auto-archive sweep, ``unread=False`` marks read up to now."""
    flags = [flag for flag, _ in _RENAME_FLAG_SETTERS]

    def _update(db):
        sid = _resolve_session_id(db, session_id)
        if not sid:
            raise HTTPException(status_code=404, detail=_NOT_FOUND)
        if body.title is None and all(getattr(body, f) is None for f in flags):
            raise HTTPException(
                status_code=400,
                detail="Nothing to update; provide 'title', 'archived', 'hidden', 'pinned', and/or 'unread'.",
            )
        if body.title is not None:
            try:
                db.set_session_title(sid, body.title or "")
            except ValueError as e:
                # Title too long, invalid characters, or already in use.
                raise HTTPException(status_code=400, detail=str(e))
        result = {"ok": True, "title": None}
        for flag, setter in _RENAME_FLAG_SETTERS:
            value = getattr(body, flag)
            if value is not None:
                setter(db, sid, value)
                result[flag] = bool(value)
        result["title"] = db.get_session_title(sid) or ""
        return result

    return _with_db(body.profile, _update, read_only=False)


def _compact_json(obj) -> str:
    return json.dumps(jsonable_encoder(obj), ensure_ascii=False, separators=(",", ":"))


@manage_router.get("/api/sessions/{session_id}/export")
async def export_session_endpoint(session_id: str, profile: Optional[str] = None):
    """Stream a single session (metadata + messages) as JSON."""
    def _prepare_export(db):
        sid = _resolve_session_id(db, session_id)
        return (sid, db.get_session(sid)) if sid else None

    prepared = await asyncio.to_thread(_with_db, profile, _prepare_export, read_only=True)
    if prepared is None or prepared[1] is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)

    sid, session = prepared

    def _stream_export():
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            yield _compact_json(session)[:-1] + ',"messages":['
            # Keyset pagination (id > last_seen): O(n) total over the
            # transcript, vs OFFSET's O(n²) on huge sessions.
            last_id, first = 0, True
            while True:
                messages = db.get_messages(sid, limit=500, after_id=last_id)
                for message in messages:
                    yield ("" if first else ",") + _compact_json(message)
                    first = False
                last_id = messages[-1].get("id") if len(messages) == 500 else None
                if last_id is None:  # short page, or cannot keyset without row ids
                    break
            yield "]}"
        finally:
            db.close()

    return StreamingResponse(_stream_export(), media_type="application/json")


@manage_router.post("/api/sessions/prune")
async def prune_sessions_endpoint(body: SessionPrune):
    """Delete ended sessions matching filters without blocking the event loop."""
    return await asyncio.to_thread(_prune_sessions, body)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Any  # noqa: F401,E402
from typing import Dict  # noqa: F401,E402
import logging  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
