"""Session dashboard routes.

Three routers because global route order matters: ``list_router`` (GET
/api/sessions) mounts before the profiles ``sessions_router``, ``search_router``
right after it, ``manage_router`` (mutation/detail) much later.  web_server-owned
helpers are reached via the late-binding seam so monkeypatching keeps working.
"""

import asyncio
import contextlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Callable, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from hermes_cli.session_listing import subagent_listing_scope
from hermes_cli.web_deps import late
from hermes_cli.web_server_gateway import _strip_session_list_rows
from hermes_cli.web_server_sessions import _listing_filters, _session_latest_descendant, _session_rows
from hermes_cli.web_models import (
    BulkDeleteSessions, SessionImport, SessionOwnerBackfill, SessionPrune, SessionRename)
from hermes_cli.web_routers._common import (
    CORRUPT_STORE_DETAIL, corrupt_store_as_status, corrupt_store_status, log as _log, destructive_profile, http_failure,
    is_transient_store_error,
)
from persistence.conversation.messages import TranscriptRead, TranscriptView
from persistence.conversation.sessions import SessionCountQuery, SessionOrder, SessionQuery, SessionWindow
from persistence.core.errors import CorruptStoreError, PersistenceError, SearchIndexCorrupt, StoreReplaced
from persistence.core.health import StorageState
from persistence.housekeeping.maintenance import PruneCriteria
from persistence_assembly import profile_scope, storage_health, store_location
from storage_errors import is_malformed_db_error

list_router = APIRouter()
search_router = APIRouter()
manage_router = APIRouter()

_cron_default_profile = late("_cron_default_profile", "hermes_cli.web_server_cron")
_cron_profile_home = late("_cron_profile_home", "hermes_cli.web_server_cron")
_open_session_db_for_profile = late("_open_session_db_for_profile", "hermes_cli.web_server_sessions")
_session_db_path_for_profile = late("_session_db_path_for_profile", "hermes_cli.web_server_sessions")

_NOT_FOUND = "Session not found"

# CRITICAL — every literal-path route on ``manage_router`` MUST be declared
# BEFORE the templated ``/api/sessions/{session_id}`` family. Starlette matches
# in registration order and ``{session_id}`` is unconstrained, so e.g.
# ``DELETE /api/sessions/empty`` would otherwise be taken as "the session with
# id 'empty'" (404, or worse, deleting the wrong row). Move the block as a unit.

# Stream-safe import: FastAPI otherwise buffers an arbitrarily large JSON body
# before the store can enforce its own per-session and transaction limits.
_SESSION_IMPORT_MAX_BYTES = 25 * 1024 * 1024


async def _read_session_import_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _SESSION_IMPORT_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Session import payload is too large")
        body.extend(chunk)
    return bytes(body)


# Prune filters forwarded to the store; string filters map "" -> None.
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
    with _profile_store(body.profile, read_only=body.dry_run) as store:
        source = body.source or None
        criteria = PruneCriteria(
            started_before=body.started_before, started_after=body.started_after,
            archived=None if body.include_archived else False,
            **{f: (getattr(body, f) or None) for f in _PRUNE_STR_FILTERS if f != "source"},
            **{f: getattr(body, f) for f in _PRUNE_NUM_FILTERS})
        skipped_open = store.maintenance.count_open_prune_matches(effective_older_than, source, criteria=criteria)
        if body.dry_run:
            # Same whole-lineage selection prune_sessions applies, so the preview lists what it deletes.
            rows = store.maintenance.list_prune_candidates(
                effective_older_than, source, criteria=criteria, whole_lineages=True)
            return {
                "ok": True,
                "removed": 0,
                "matched": len(rows),
                "skipped_open": skipped_open,
                # Rows are ordered by last activity, not creation time.
                "oldest_last_active": rows[0].last_active if rows else None,
                "newest_last_active": rows[-1].last_active if rows else None,
                "oldest_started_at": min(r.started_at for r in rows) if rows else None,
                "newest_started_at": max(r.started_at for r in rows) if rows else None,
                "sessions": [{k: getattr(r, "session_id" if k == "id" else k) for k in _PRUNE_ROW_KEYS}
                             for r in rows]}
        from session_retention import after_sessions_removed

        sessions_dir = profile_home / "sessions"
        outcome = store.maintenance.prune_sessions(
            effective_older_than, source, criteria=criteria, exclude_active_write_guards=True,
            media_root=_media_root())
        after_sessions_removed(store, store.scope.profile_dir, outcome.deleted_ids, reason="prune",
                               sessions_dir=sessions_dir if sessions_dir.exists() else None)
        return {"ok": True, "removed": outcome.count, "skipped_open": skipped_open}


_ACTIVE_WINDOW_S = 300


def _csv(value: Optional[str]) -> list[str]:
    """Split a comma-separated query param into stripped, non-empty items."""
    return [s.strip() for s in (value or "").split(",") if s.strip()]


def _is_active(row: dict, now: float) -> bool:
    return (
        row.get("ended_at") is None
        and (now - row.get("last_active", row.get("started_at", 0))) < _ACTIVE_WINDOW_S)


@contextlib.contextmanager
def _profile_store(profile: Optional[str], *, read_only: bool):
    """The profile's storage interface handle (a reader when ``read_only``), closed on exit."""
    store = _open_session_db_for_profile(profile, read_only=read_only)
    try:
        yield store
    finally:
        store.close()


def _media_root() -> str:
    """The retained media root a delete's retired admissions are checked against."""
    from gateway.session_ingress_media import native_inputs_root

    return str(native_inputs_root())


def _with_db(profile: Optional[str], fn: Callable, *, read_only: bool):
    """Open the profile's storage interface handle, run ``fn(store)``, always close."""
    def run():
        with _profile_store(profile, read_only=read_only) as store:
            return fn(store)
    if read_only:
        return run()
    from hermes_cli.web_server_sessions import _with_session_maintenance
    return _with_session_maintenance(profile, run)


def _serving_profile(profile: Optional[str]) -> str:
    """The profile name rows are stamped with: the requested one, else the
    serving process's own — so default-profile rows never circulate unowned."""
    return _cron_profile_home(profile)[0] if profile else _cron_default_profile()


def _resolve_session_id(store, session_id: str) -> Optional[str]:
    """Resolve *session_id*; a corrupt store (prefix scan raises "malformed") is
    reported as 503 with the actual problem instead of a misleading 404."""
    try:
        return store.sessions.resolve_id(session_id)
    except CorruptStoreError as exc:
        if isinstance(exc.__cause__, sqlite3.DatabaseError) and not is_malformed_db_error(exc.__cause__):
            raise
        _log.error("state.db is corrupt while resolving session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Session store is corrupt (database disk image is malformed). "
                "Sessions cannot be read until it is repaired — run "
                "`hermes doctor` for diagnosis."),
        ) from exc
    except StoreReplaced:
        # Same 503 payload as the analytics reads (#110054).
        with corrupt_store_as_status(store_location(store.scope)):
            raise


# ``le=100`` on limit: an unbounded limit lets one request drag every session
# row (plus correlated-subquery preview work) out of SQLite in a single hit.
@list_router.get("/api/sessions")
def get_sessions(
    limit: int = Query(20, ge=0, le=100), offset: int = Query(0, ge=0), min_messages: int = 0,
    archived: str = "exclude", order: str = "created", source: str | None = None, sources: str | None = None,
    exclude_sources: str | None = None, cwd_prefix: str | None = None, full: bool = False,
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
        store = _open_session_db_for_profile(profile, read_only=True)
        try:
            home = _session_db_path_for_profile(profile).parent
            min_message_count = max(0, min_messages)
            archived_only = archived == "only"
            include_archived = archived == "include"
            # Source scoping: the desktop splits recents (exclude=cron) from
            # the cron-jobs section (source=cron) into two independent lists.
            source_list = _csv(sources)
            include_subagents, exclude_list = subagent_listing_scope(
                home, source=source or None, sources=source_list or None,
                exclude_sources=_csv(exclude_sources) or None)
            scope = dict(
                source=source or None, sources=source_list or None,
                exclude_sources=exclude_list or None, cwd_prefix=(cwd_prefix or None),
                min_message_count=min_message_count, include_archived=include_archived,
                archived_only=archived_only, include_subagents=include_subagents)
            sessions = _session_rows(
                store,
                limit=limit,
                offset=offset,
                order_by_last_active=order == "recent",
                # Skip the system_prompt blob inside SQLite too (pairs with
                # _strip_session_list_rows below).
                compact_rows=not full,
                include_pinned=True,
                **scope)
            total = store.sessions.count_sessions(SessionCountQuery(**_listing_filters(**scope), exclude_children=True))
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
            # ``storage`` tells an empty page apart from an unreadable store (#72046); same
            # ``{profile: "corrupt"}`` shape as the /api/profiles/sessions* lists.
            health = storage_health(profile_scope(home=home))
            storage = {row_profile: StorageState.CORRUPT.value} if health.storage_state() is StorageState.CORRUPT else {}
            return {"sessions": sessions, "total": total, "limit": limit, "offset": offset,
                    "storage": storage}
        finally:
            store.close()
    except HTTPException:
        raise
    except StoreReplaced as exc:
        # RuntimeError family, not sqlite3: same 503 payload as the analytics reads (#110054).
        raise corrupt_store_status(_session_db_path_for_profile(profile), exc) from exc
    except PersistenceError as exc:
        cause = exc.__cause__ if isinstance(exc.__cause__, sqlite3.Error) else None
        if is_transient_store_error(exc) or isinstance(cause, sqlite3.OperationalError):
            _log.exception("GET /api/sessions failed")
            # 503, not 500: the store is busy, not gone — the desktop keeps its
            # sidebar instead of reading a 500 as an authoritative empty list.
            transient = is_transient_store_error(exc)
            raise HTTPException(
                status_code=503 if transient else 500,
                detail=(
                    "Session store is busy (disk I/O or lock). Retry; the list was not cleared."
                    if transient
                    else "Internal server error"),
            ) from exc
        # A damaged store is unavailable, not empty and not an internal error (#72046). Damage SQLite scopes to
        # the search index counts only when it reads as a malformed image.
        db_path = _session_db_path_for_profile(profile)
        if not (isinstance(exc, CorruptStoreError) and (
                cause is None or not isinstance(exc, SearchIndexCorrupt) or is_malformed_db_error(cause))):
            _log.exception("GET /api/sessions failed")
            raise HTTPException(status_code=500, detail="Internal server error") from exc
        _log.error("GET /api/sessions: state.db at %s is corrupt: %s", db_path, exc)
        raise HTTPException(status_code=503, detail=dict(CORRUPT_STORE_DETAIL)) from exc
    except Exception:
        _log.exception("GET /api/sessions failed")
        raise HTTPException(status_code=500, detail="Internal server error")


def _is_compression_edge(child, parent) -> bool:
    parent_ended_at = parent.ended_at
    started_at = child.started_at
    return (
        parent.end_reason == "compression"
        and parent_ended_at is not None
        and started_at is not None
        and started_at >= parent_ended_at)


@search_router.get("/api/sessions/search")
async def search_sessions(
    q: str = "", limit: int = 20, profile: Optional[str] = None, source: str | None = None,
    sources: str | None = None, exclude_sources: str | None = None):
    """Search sessions by ID (first) plus FTS5 message content.

    Results are deduped by compression lineage, not raw ``session_id``:
    auto-compression rotates a chat onto a fresh id and leaves the old segment
    in the FTS index.  Branches also use ``parent_session_id`` but are real
    alternate conversations — they are NOT collapsed into the parent.
    """
    if not q or not q.strip():
        return {"results": []}
    with http_failure("GET /api/sessions/search failed", 500, detail="Search failed"):
        row_profile = _serving_profile(profile)

        def _search(store):
            safe_limit = max(1, min(int(limit or 20), 100))
            source_filter = source or None
            source_list = _csv(sources)
            include_sources = [source_filter] if source_filter else (source_list or None)
            exclude_list = _csv(exclude_sources)
            now = time.time()

            def get_session(sid):
                try:
                    return store.sessions.get(sid)
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
                    parent = s.parent_session_id if s is not None else None
                    parent_session = get_session(parent) if parent else None
                    if not parent_session or not _is_compression_edge(s, parent_session):
                        root = cur
                        break
                    cur = parent
                for node in chain:
                    root_cache[node] = root
                return root

            tip_cache: dict = {}

            def lineage_tip(session_id: str) -> str:
                # Resolve the tip from the MATCHED id, never from the lineage
                # root: the forward chain walk is defensively bounded, so a
                # lineage deeper than the bound truncates to a stale mid id
                # when started at the root. Resuming from the matched id is
                # what the CLI does and always reaches the live tip (#125041).
                if session_id not in tip_cache:
                    try:
                        tip_cache[session_id] = store.lineage.tip(session_id)
                    except Exception:
                        tip_cache[session_id] = session_id
                return tip_cache[session_id]

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
                sid = lineage_tip(raw_sid)
                payload["session_id"] = sid
                payload["lineage_root"] = root
                payload["profile"] = row_profile
                payload["is_default_profile"] = row_profile == "default"
                try:
                    entry = store.sessions.summary(sid)
                except Exception:
                    entry = None
                if entry:
                    row = entry.session
                    last_active = entry.last_active or row.started_at
                    payload.update({
                        "id": row.session_id or sid,
                        "source": row.source,
                        "model": row.model,
                        "title": row.title,
                        "started_at": row.started_at,
                        "ended_at": row.ended_at,
                        "last_active": last_active,
                        "is_active": (
                            row.ended_at is None and (now - (last_active or 0)) < 300),
                        "message_count": row.message_count or 0,
                        "tool_call_count": row.tool_call_count or 0,
                        "input_tokens": row.input_tokens or 0,
                        "output_tokens": row.output_tokens or 0,
                        "preview": entry.preview or None,
                        "parent_session_id": row.parent_session_id,
                        "profile": _serving_profile(profile),
                        "archived": row.archived})
                else:
                    payload["id"] = sid
                seen[root] = payload

            def hit_payload(snippet: str, role, source, model, session_started, last_active=None) -> dict:
                # `last_active` rides only on id-match rows (sessions table); FTS
                # hits have no row recency and leave it null so the desktop can
                # fall back to session_started instead of inventing one.
                return {
                    "snippet": snippet, "role": role, "source": source,
                    "model": model, "session_started": session_started,
                    "last_active": last_active}

            # Direct ID matches first (pasted ids never appear in message text).
            for entry in store.sessions.search_by_id(
                q, limit=safe_limit, include_archived=True, sources=include_sources,
                exclude_sources=exclude_list):
                row = entry.session
                sid = row.session_id
                preview = entry.preview.strip()
                snippet = preview or f"Session ID: {sid}"
                add_lineage_result(sid, hit_payload(
                    snippet, None, row.source, row.model, row.started_at, entry.last_active))

            # Prefix wildcards so partial words match ("nimb" -> "nimb*");
            # quoted phrases and existing wildcards are kept as-is.
            prefix_query = " ".join(
                tok if tok.startswith('"') or tok.endswith("*") else tok + "*"
                for tok in re.findall(r'"[^"]*"|\S+', q.strip()))
            # Over-fetch so lineage dedup can still surface `limit` distinct
            # conversations when several hits collapse onto one root.
            matches = store.search.search_messages(
                prefix_query, source_filter=include_sources,
                exclude_sources=exclude_list or None, limit=max(safe_limit * 5, 50),
                fields=("session_id", "role", "snippet", "source", "model", "session_started"))
            for m in matches:
                if len(seen) >= safe_limit:
                    break
                add_lineage_result(
                    m.session_id,
                    hit_payload(m.snippet, m.role, m.source, m.model, m.session_started))

            # Title matches fill any remaining slots (#66242): the FTS index
            # only covers message content, so a term that lives solely in a
            # manually-set sessions.title would otherwise return nothing. The
            # DB layer already knows how to LIKE-match titles across the whole
            # compression chain (the listing's ``search_text``, the same
            # helper the sidebar listing uses), so reuse it rather than adding
            # a second title query path here. Best-effort: an old/odd store
            # that rejects the call just skips the lane.
            if len(seen) < safe_limit:
                try:
                    title_rows = store.sessions.list_sessions(
                        SessionQuery(search_text=q.strip(), include_archived=True, order=SessionOrder.LAST_ACTIVE,
                                     sources=tuple(include_sources) if include_sources else None,
                                     exclude_sources=tuple(exclude_list)),
                        SessionWindow(limit=safe_limit))
                except Exception:  # health: allow BLE001 -- best-effort supplement lane: an old/odd store that rejects the search_query read must not fail the id+content results already collected
                    _log.debug("Title-match supplement skipped for %r", q[: 200])
                    title_rows = []
                for entry in title_rows:
                    if len(seen) >= safe_limit:
                        break
                    row = entry.session
                    sid = row.session_id
                    if not sid:
                        continue
                    preview = entry.preview.strip()
                    add_lineage_result(
                        sid, hit_payload(preview or f"Session title matched: {q.strip()}",
                                         None, row.source, row.model, row.started_at, entry.last_active))
            return {"results": list(seen.values())}

        # FTS over a large state.db is the slowest read here; keep it off the loop (#60747).
        return await asyncio.to_thread(_with_db, profile, _search, read_only=True)


@manage_router.post("/api/sessions/bulk-delete")
async def bulk_delete_sessions_endpoint(body: BulkDeleteSessions):
    """Delete every session in ``body.ids`` in one transaction (POST: many
    clients refuse a DELETE body).

    Per ``SessionRepository.delete_many``: unknown ids are skipped (``deleted``
    reports what really happened), children are orphaned, active/archived rows
    ARE deleted (hand-picked), on-disk cleanup is left to the next prune.

    Compression chains are deleted whole: the sessions list shows one row per
    logical conversation carrying the chain *tip's* id, so deleting only that
    row would leave the root to resurface as the previous chain link on the
    next reload (#57543). ``deleted`` still counts the selected rows, not the
    expanded chain links.
    """
    # Hard cap so a runaway selection can't lock the writer for long.
    if len(body.ids) > 500:
        raise HTTPException(status_code=400, detail="ids must contain at most 500 entries")
    profile = destructive_profile(body.profile, "POST /api/sessions/bulk-delete")

    def _delete(store):
        from session_retention import after_sessions_removed

        unique_ids = list({sid for sid in body.ids or () if isinstance(sid, str) and sid})
        if not unique_ids:
            return 0, []
        outcome = store.sessions.delete_many(
            body.ids, skip_guarded=True, include_compression_chain=True, media_root=_media_root())
        after_sessions_removed(store, store.scope.profile_dir, outcome.removed_ids, reason="delete",
                               primary=unique_ids[0] if len(unique_ids) == 1 else None)
        return outcome.deleted, list(outcome.skipped_guarded)

    # skipped: rows a live turn/compression still owns; the UI must keep them listed
    deleted, skipped = await asyncio.to_thread(_with_db, profile, _delete, read_only=False)
    return {"ok": True, "deleted": deleted, "skipped_active": skipped}


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

    from hermes_cli.web_server_sessions import _mutate_session_request
    anchor = body.sessions[0].get('id', '') if body.sessions else ''
    return await _mutate_session_request(request, body.profile, anchor,
        request_id=body.request_id, expected_revision=body.expected_revision,
        operation='import', payload={'sessions': body.sessions})


@manage_router.get("/api/sessions/empty/count")
async def count_empty_sessions_endpoint(profile: Optional[str] = None):
    """Count of empty, ended, non-archived sessions (the "Delete empty (N)" button)."""
    count = await asyncio.to_thread(
        _with_db, profile, lambda store: store.sessions.count_empty(), read_only=True)
    return {"count": count}


@manage_router.delete("/api/sessions/empty")
async def delete_empty_sessions_endpoint(profile: Optional[str] = None):
    """Delete every empty, ended, non-archived session in one transaction.

    "Empty" means NO ``messages`` rows at all — a rewound/compacted chat reads
    ``message_count == 0`` while its soft-archived rows are the only transcript
    copy (see ``SessionRepository.delete_empty``).

    * Active sessions are skipped (``ended_at IS NULL``) so a live agent isn't yanked mid-handshake. *
    Archived sessions are skipped — the user explicitly chose to keep those rows. * Children of deleted
    parents are orphaned, not cascade-deleted. See #95868.
    """
    profile = destructive_profile(profile, "DELETE /api/sessions/empty")

    def _delete_empty(store):
        from session_retention import after_sessions_removed

        removed = store.sessions.delete_empty(media_root=_media_root())
        after_sessions_removed(store, store.scope.profile_dir, removed, reason="empty_sweep",
                               sessions_dir=_session_files_dir(profile))
        return len(removed)

    deleted = await asyncio.to_thread(_with_db, profile, _delete_empty, read_only=False)
    return {"ok": True, "deleted": deleted}


@manage_router.get("/api/sessions/stats")
async def get_session_stats(profile: Optional[str] = None):
    """Session-store statistics (mirrors `hermes sessions stats`)."""
    def _stats(store):
        count = store.sessions.count_sessions
        out = {
            # exclude_children=True: the dashboard Sessions page pairs these counts with
            # listing rows, which collapse compression chains to one row carrying
            # the tip's id — the raw row count is inflated by the hidden chain links (#54298).
            "total": count(SessionCountQuery(include_archived=True, exclude_children=True)),
            "active_store": count(SessionCountQuery(include_archived=False, exclude_children=True)),
            "archived": count(SessionCountQuery(archived_only=True, exclude_children=True)),
            "messages": store.messages.count_all(),
            "by_source": {}}
        try:
            out["by_source"] = dict(store.sessions.count_by_source(
                include_archived=True, exclude_children=True))
        except Exception:
            pass
        return out

    return await asyncio.to_thread(_with_db, profile, _stats, read_only=True)


@manage_router.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str, profile: Optional[str] = None):
    def _detail(store):
        sid = _resolve_session_id(store, session_id)
        record = store.sessions.get(sid) if sid else None
        if record is None:
            raise HTTPException(status_code=404, detail=_NOT_FOUND)
        session = record.as_session_row()
        # Always stamp the owner: unowned default-profile rows made multi-profile
        # clients resolve them to whichever gateway happened to be active.
        session["profile"] = _serving_profile(profile)
        session["is_default_profile"] = session["profile"] == "default"
        # A cron run's liveness is scheduler ownership, not the 300s activity
        # window (#88443): a run inside a long tool call is still owned. A
        # unified-runtime run's local id names no job, so stamp its identity first.
        from hermes_cli.web_routers.cron import cron_run_scheduler_owned

        if session.get("source") == "cron" and (identity := store.sessions.cron_run_identity(sid)) is not None:
            session.update(cron_job_id=identity.cron_job_id, cron_execution_id=identity.cron_execution_id)
        owned = cron_run_scheduler_owned(session, profile)
        if owned is not None:
            session["scheduler_owned"] = owned
        return session

    return await asyncio.to_thread(_with_db, profile, _detail, read_only=True)


@manage_router.get("/api/sessions/{session_id}/mutation-snapshot")
async def get_session_mutation_snapshot(session_id: str, request: Request, profile: Optional[str] = None):
    from hermes_cli.web_server_sessions import _session_mutation_context
    authority, _actor = _session_mutation_context(request, profile)
    if authority is None:  # standalone serve: the counters the offline receipt checks
        record = await asyncio.to_thread(
            _with_db, profile, lambda store: store.sessions.get(session_id), read_only=True)
    else:
        from gateway.session_persistence import authority_persistence
        record = authority_persistence(authority).sessions.get(session_id)
    row = None if record is None else {
        'runtime_revision': record.runtime_revision, 'runtime_generation': record.runtime_generation}
    # An absent import anchor has revision zero by the owner's storage contract,
    # not by a client guessing after a failed or stale detail request.
    return {'session_id': session_id, 'exists': row is not None,
            'runtime_revision': row['runtime_revision'] if row else 0,
            'runtime_generation': row['runtime_generation'] if row else None}


@manage_router.get("/api/sessions/{session_id}/latest-descendant")
async def get_session_latest_descendant(session_id: str, profile: Optional[str] = None):
    latest, path = await asyncio.to_thread(
        _with_db, profile, lambda store: _session_latest_descendant(session_id, store), read_only=True)
    if not latest:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return {
        "requested_session_id": path[0] if path else session_id, "session_id": latest, "path": path,
        "changed": bool(path and latest != path[0])}


def _stored_tool_call_labels(message: dict) -> dict:
    from agent.display import tool_labels_for_call
    from tools.tool_labels import BRIDGE_TOOL_NAMES

    out = {}
    for call in message.get("tool_calls") or ():
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        call_id, name = str(call.get("id") or ""), str(fn.get("name") or "")
        if not call_id or name not in BRIDGE_TOOL_NAMES:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            args = {}
        labels = [label.as_payload() for label in tool_labels_for_call(name, args if isinstance(args, dict) else {})]
        if labels:
            out[call_id] = labels
    return out


def _with_tool_call_labels(message: dict) -> dict:
    labels = _stored_tool_call_labels(message)
    return {**message, "tool_call_labels": labels} if labels else message


def _history_profile_home(profile):
    if profile:
        return _cron_profile_home(profile)[1]
    # An omitted profile reads this process's DB (including custom HERMES_HOME),
    # not necessarily the registered default profile used by cron routes.
    from hermes_cli.config import get_hermes_home

    return get_hermes_home()


def _session_files_dir(profile) -> Path:
    """Transcript dir of the profile whose store a delete targets: a delete only unlinks the
    session's on-disk artifacts when handed this, and a row-only delete leaves the
    (secret-bearing) ``session_<id>.json`` snapshots and ``request_dump_<id>_*.json`` readable after
    the user removed the session (#55088, #60207)."""
    return _history_profile_home(profile) / "sessions"


def _is_untyped_scaffold_notice(message) -> bool:
    """A ``[System: …]`` role=user row persisted without a ``display_kind``.

    ``[System:`` is a reserved gateway-notice namespace — it must never render as a user
    bubble (the gateway's own history projection drops these rows outright) — but recovery
    scaffolding written before typing existed carries no kind. Rows WITH a kind
    (``model_switch``, …) are timeline entries and keep flowing.
    """
    if not isinstance(message, dict) or message.get("role") != "user" or message.get("display_kind"):
        return False
    content = message.get("content")
    return isinstance(content, str) and content.lstrip().startswith("[System:")


def _project_for_display(messages: list, *, home=None, inline_images: bool = True) -> list:
    """Replace compaction summaries with their display-only projection and hide untyped
    gateway-scaffold notices.

    Recovery scaffolding (e.g. the stream-timeout nudge appended when a tool call's stream
    is cut) persists as a ``[System: …]`` ``role=user`` row with no ``display_kind``. This
    projection feeds the Desktop's transcript prefetch, which addresses VISIBLE user rows by
    durable row id — and the gateway truncation resolver refuses scaffold rows fail-closed,
    so a shipped scaffold row can never resolve as a rewind/regenerate target and dead-ends
    every retry (``refusing truncation without fallback``). Hide them the same way the
    Desktop collapses other display-only rows; typed notices stay for the timeline.
    """
    from agent.compaction_display import project_compaction_message_for_display
    from session_rules.turns import is_compaction_summary_message, withheld_prompt_text
    from agent.conversation_compression import _extract_steer_text_from_message
    from agent.history_commentary import project_history_commentary
    from agent.prompt_builder import STEER_DISPLAY_KIND
    from agent.turn_failure_copy import CONTENT_REFUSED_DISPLAY_KIND, untyped_failed_turn_display_kind

    # inline_images=False (#116511): render content through the gateway's ``_coerce_message_text``
    # projection so a data-URI image part becomes ``[image]`` — the same branch session.resume's
    # ``inline_images=false`` uses, kilobytes instead of re-transmitting every stored attachment.
    coerce = None
    if not inline_images:
        from tui_gateway.session_history import _coerce_message_text

        def coerce(message: dict) -> dict:
            if message.get("content") is not None:
                return {**message, "content": _coerce_message_text(message["content"], image_urls=False)}
            return message

    projected_messages = []
    for message in messages:
        message = _with_tool_call_labels(message)
        if coerce is not None:
            message = coerce(message)
        # Same read-side typing as session.resume (tui_gateway/session_history.py).
        failed_turn = not message.get("display_kind") and untyped_failed_turn_display_kind(
            message.get("role"), message.get("content"))
        if failed_turn:
            message = {**message, "display_kind": failed_turn}
        if _is_untyped_scaffold_notice(message):
            projected = message.copy()
            projected["display_kind"] = "hidden"
            projected_messages.append(projected)
            continue
        # Mid-turn steer: the user's own words, not the model-facing marker (same as session.resume).
        if message.get("role") == "user" and message.get("display_kind") == STEER_DISPLAY_KIND and (
                steer_text := _extract_steer_text_from_message(message)):
            message = {**message, "display_content": steer_text}
        # A pre_llm_call veto stores a placeholder; the transcript shows what the user typed.
        if message.get("role") == "user" and (typed := withheld_prompt_text(message)) is not None:
            message = {**message, "display_content": typed}
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
            # A refused ask embedded in a carrier keeps its content_refused tag so the reopened
            # transcript still renders the greyed refused block (#149708 review).
            if projected.get("display_kind") != CONTENT_REFUSED_DISPLAY_KIND:
                projected.pop("display_kind", None)
        projected_messages.append(projected)
    return project_history_commentary(projected_messages, home=home)


@manage_router.get("/api/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str, profile: Optional[str] = None, limit: Optional[int] = Query(None, ge=0),
    offset: int = Query(0, ge=0), order: Optional[str] = Query(None),
    include_compacted: bool = Query(False), inline_images: bool = Query(True)):
    if order not in (None, "oldest", "latest"):
        raise HTTPException(status_code=400, detail="order must be one of: oldest, latest")

    def _read(store):
        sid = _resolve_session_id(store, session_id)
        if not sid:
            return None
        sid = store.lineage.resolve_resume_target(sid)
        # Always page (an omitted limit used to load whole transcripts). Explicit
        # pagination anchors at the start; the default view is the latest page.
        default_page = limit is None
        latest_page = order == "latest" or (order is None and default_page)
        _limit = 500 if default_page else min(limit, 500)
        # Include compression-ancestor messages so the REST transcript
        # matches the gateway's session.resume (which uses
        # include_ancestors=True). Without this, the desktop's REST
        # prefetch only shows the child continuation's messages after a
        # compression rotation, hiding the pre-compaction transcript
        # (#51058).
        rows = store.messages.read_transcript(sid, TranscriptRead(
            view=TranscriptView.DISPLAY if include_compacted else TranscriptView.ACTIVE, limit=_limit,
            offset=offset, latest=latest_page, include_lineage=True))
        return sid, _limit, [row.as_wire_row() for row in rows]

    result = await asyncio.to_thread(_with_db, profile, _read, read_only=True)
    if result is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    sid, _limit, messages = result
    projected_messages = await asyncio.to_thread(
        _project_for_display, messages, home=_history_profile_home(profile),
        inline_images=inline_images)
    return {
        "session_id": sid,
        # The same stamp list rows carry, so the Desktop keys a page under the
        # owner it already routes the session by.
        "profile": _serving_profile(profile),
        "messages": projected_messages,
        "pagination": {
            "limit": _limit, "offset": offset,
            "order": order or ("latest" if limit is None else "oldest"),
            "returned": len(projected_messages)}}


def _timeline_session_id(store, session_id: str, owner: str) -> str:
    # Durable jump addresses are exact ids, never title/prefix guesses. A NULL
    # legacy owner belongs to this profile's store, just like /messages pages.
    def owned(sid):
        row = store.sessions.get(sid)
        return row is not None and row.profile_name in (None, owner)

    if not owned(session_id):
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    sid = store.lineage.resolve_resume_target(session_id)
    if not owned(sid):
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return sid


@manage_router.get("/api/sessions/{session_id}/timeline")
async def get_session_timeline(
    session_id: str, profile: Optional[str] = None,
    limit: int = Query(500, ge=1, le=500), after_row_id: int = Query(0, ge=0),
):
    """Prompt metadata only, including compacted display history (never rewind rows).

    ``next_cursor`` is a stable logical first-row id; pass it as ``after_row_id``.
    Entry ``row_id`` addresses the current representative for /messages/around.
    """
    owner = _serving_profile(profile)

    def _read(store):
        sid = _timeline_session_id(store, session_id, owner)
        timeline = store.messages.read_session_timeline(sid, limit=limit, after_row_id=after_row_id)
        return {"session_id": sid, "profile": owner,
                "entries": [{"row_id": e.row_id, "preview": e.preview, "timestamp": e.timestamp}
                            for e in timeline.entries],
                "pagination": {"limit": timeline.limit, "after_row_id": timeline.after_row_id,
                               "returned": timeline.returned, "total": timeline.total,
                               "has_more": timeline.has_more, "next_cursor": timeline.next_cursor}}

    return await asyncio.to_thread(_with_db, profile, _read, read_only=True)


@manage_router.get("/api/sessions/{session_id}/messages/around")
async def get_session_messages_around(
    session_id: str, row_id: int = Query(..., ge=1), profile: Optional[str] = None,
    limit: int = Query(120, ge=1, le=120),
):
    """Bounded display page starting at a timeline prompt; no intervening payloads."""
    owner = _serving_profile(profile)

    def _read(store):
        sid = _timeline_session_id(store, session_id, owner)
        page = store.messages.read_prompt_page(sid, row_id, limit=limit)
        if page is None:
            raise HTTPException(status_code=404, detail="Prompt not found")
        return {"session_id": sid, "profile": owner,
                "messages": [row.as_wire_row() for row in page.messages],
                "pagination": {"row_id": page.row_id, "limit": page.limit, "returned": page.returned,
                               "order": "oldest", "offset": page.offset, "total": page.total,
                               "has_older": page.has_older, "has_newer": page.has_newer}}

    result = await asyncio.to_thread(_with_db, profile, _read, read_only=True)
    result["messages"] = await asyncio.to_thread(
        _project_for_display, result["messages"], home=_history_profile_home(profile))
    return result


@manage_router.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, request: Request, profile: Optional[str] = None,
                                  request_id: Optional[str] = None, expected_revision: Optional[int] = None,
                                  expected_generation: Optional[int] = None):
    from hermes_cli.web_server_sessions import _mutate_session_request
    from session_retention import remove_session_files
    try:
        result = await _mutate_session_request(request, profile, session_id,
            request_id=request_id, expected_revision=expected_revision,
            expected_generation=expected_generation, operation='delete', payload={})
    except HTTPException as exc:
        # Already-absent is an idempotent success: the desktop optimistically removes the row
        # and RESTORES it on any error, so a 404 resurrected ghost rows (a row deleted by
        # another client between the sidebar snapshot and this DELETE). An exact retry of the
        # caller's own committed request never reaches here: mutate_session checks the receipt
        # before resolving the id, so it replays that receipt (or 409s on a changed digest).
        # The authority also answers ``not_found`` for an EXISTING cold row it cannot restore
        # (e.g. an API-server row without its binding); that row was not deleted, so only a
        # genuinely absent id is the idempotent success.
        if exc.status_code == 404 and exc.detail == 'not_found' and await asyncio.to_thread(
                _with_db, profile, lambda store: store.sessions.get(session_id) is None, read_only=True):
            return {"ok": True, "already_absent": True}
        raise
    # The canonical delete retires rows only: scrub each deleted session's on-disk artifacts
    # (secret-bearing ``session_<id>.json`` snapshots, ``request_dump_<id>_*.json``) as the CLI
    # delete path does (#60207). Idempotent, so an exact retry of the receipt repeats it safely.
    sessions_dir = _session_files_dir(profile)
    for sid in result.get('deleted_ids') or ():
        await asyncio.to_thread(remove_session_files, sessions_dir, str(sid))
    return result


def _owner_store_for(request: Request, profile: Optional[str]):
    """The live owner's own storage handle for *profile* when a composed gateway serves it, else None."""
    runner = getattr(request.app.state, "gateway_runner", None)
    if runner is None:
        return None
    from gateway.session_authorities import authority_for_home
    from gateway.session_persistence import authority_persistence
    home = Path(_cron_profile_home(profile)[1]) if profile else profile_scope().profile_dir
    authority = authority_for_home(runner, home)
    return None if authority is None else authority_persistence(authority)


@manage_router.post("/api/sessions/owner-backfill")
async def backfill_session_owner_profiles(body: SessionOwnerBackfill, request: Request):
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
        owner_store = _owner_store_for(request, body.profile)
        if owner_store is not None:
            # A live gateway owns this store: it is the only writer (the offline maintenance lock
            # would 409 every Desktop boot), so the idempotent stamp runs on its own connection.
            stamped = await asyncio.to_thread(owner_store.sessions.backfill_session_owners, stamp=stamp)
        else:
            stamped = await asyncio.to_thread(
                _with_db, body.profile, lambda store: store.sessions.backfill_session_owners(stamp=stamp),
                read_only=False)

    if stamped:
        _log.info(
            "owner-backfill: stamped %d legacy NULL-profile session row(s) with profile %r",
            stamped, stamp)
    return {"ok": True, "stamped": stamped, "profile": stamp}


@manage_router.patch("/api/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, body: SessionRename, request: Request):
    """Title and sidebar flags share one authoritative revision and receipt."""
    from hermes_cli.web_server_sessions import _mutate_session_request
    payload = {key: getattr(body, key) for key in ('title', 'archived', 'hidden', 'pinned', 'unread')
               if getattr(body, key) is not None}
    return await _mutate_session_request(request, body.profile, session_id,
        request_id=body.request_id, expected_revision=body.expected_revision,
        expected_generation=body.expected_generation, operation='sidebar', payload=payload)


def _compact_json(obj) -> str:
    return json.dumps(jsonable_encoder(obj), ensure_ascii=False, separators=(",", ":"))


@manage_router.get("/api/sessions/{session_id}/export")
async def export_session_endpoint(session_id: str, profile: Optional[str] = None):
    """Stream a single session (metadata + messages) as JSON."""
    def _prepare_export(store):
        sid = _resolve_session_id(store, session_id)
        record = store.sessions.get(sid) if sid else None
        return (sid, None if record is None else record.as_session_row()) if sid else None

    prepared = await asyncio.to_thread(_with_db, profile, _prepare_export, read_only=True)
    if prepared is None or prepared[1] is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)

    sid, session = prepared

    def _stream_export():
        with _profile_store(profile, read_only=True) as store:
            yield _compact_json(session)[:-1] + ',"messages":['
            # Keyset pagination (id > last_seen): O(n) total over the
            # transcript, vs OFFSET's O(n²) on huge sessions. Every row with its
            # active/compacted flags, so re-importing restores compacted history as archived.
            last_id, first = 0, True
            while True:
                messages = [row.as_wire_row() for row in store.messages.read_transcript(
                    sid, TranscriptRead(view=TranscriptView.ALL, limit=500, after_id=last_id))]
                for message in messages:
                    yield ("" if first else ",") + _compact_json(message)
                    first = False
                last_id = messages[-1].get("id") if len(messages) == 500 else None
                if last_id is None:  # short page, or cannot keyset without row ids
                    break
            yield "]}"

    return StreamingResponse(_stream_export(), media_type="application/json")


@manage_router.post("/api/sessions/prune")
async def prune_sessions_endpoint(body: SessionPrune):
    """Delete ended sessions matching filters without blocking the event loop."""
    if body.dry_run:
        return await asyncio.to_thread(_prune_sessions, body)
    # Same destructive rule as the rest of the family; a dry run deletes nothing, so it
    # keeps working unnamed (it is the preview the confirm dialog reads).
    body = body.model_copy(update={
        "profile": destructive_profile(body.profile, "POST /api/sessions/prune")})
    from hermes_cli.web_server_sessions import _with_session_maintenance
    return await asyncio.to_thread(_with_session_maintenance, body.profile, _prune_sessions, body)
