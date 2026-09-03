"""Session dashboard routes (extracted verbatim from web_server.py).

Three routers because the original registration points are far apart and
global route order matters: ``list_router`` (GET /api/sessions) was registered
before the profiles ``sessions_router`` include, ``search_router``
(GET /api/sessions/search) right after it, and ``manage_router`` (the
mutation/detail endpoints) thousands of lines later - each is mounted at its
original registration point so the app's route table is byte-identical.

Handler bodies are byte-identical; web_server-owned helpers are reached via
the late-binding seam in :mod:`hermes_cli.web_deps` so tests that
``monkeypatch.setattr(web_server, "_helper", ...)`` keep working.
"""

import asyncio  # noqa: F401 — used by handlers
import json
import logging
import sqlite3
import time  # noqa: F401
from typing import Any, Dict, List, Optional  # noqa: F401

from fastapi import APIRouter, HTTPException, Query, Request  # noqa: F401
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from hermes_cli.web_deps import late
from hermes_cli.web_models import (
    BulkDeleteSessions,
    SessionImport,
    SessionOwnerBackfill,
    SessionPrune,
    SessionRename,
)
from hermes_state import is_malformed_db_error, is_transient_sqlite_error

# Same logger the handlers used before extraction (identical logger object).
_log = logging.getLogger("hermes_cli.web_server")

list_router = APIRouter()
search_router = APIRouter()
manage_router = APIRouter()

# Late-bound web_server helpers (resolved at call time; cycle-safe,
# monkeypatch-transparent).
_cron_default_profile = late("_cron_default_profile")
_cron_profile_home = late("_cron_profile_home")
_import_sessions_for_profile = late("_import_sessions_for_profile")
_maybe_auto_archive_for_profile = late("_maybe_auto_archive_for_profile")
_open_session_db_for_profile = late("_open_session_db_for_profile")
_prune_sessions = late("_prune_sessions")
_read_session_import_body = late("_read_session_import_body")
_session_latest_descendant = late("_session_latest_descendant")
_strip_session_list_rows = late("_strip_session_list_rows")


def _resolve_session_id(db, session_id: str) -> Optional[str]:
    """Resolve *session_id*, distinguishing "absent" from "unreadable".

    A corrupt ``state.db`` does not raise on every read. The exact-match
    lookup goes through the ``sessions`` primary-key index, so a damaged
    index simply *misses* and returns None — indistinguishable from a
    session that was never there. Only the prefix fallback scans the base
    b-tree and raises ``database disk image is malformed``.

    Both outcomes used to end at ``404 Session not found``: one silently, one
    as an unhandled 500 that the Desktop surfaced as "session unavailable".
    Reporting a corrupt store as an empty one cost a day of looking at
    session logic during the 2026-08-31 incident, so classify it here and say
    what is actually wrong.
    """
    try:
        return db.resolve_session_id(session_id)
    except sqlite3.DatabaseError as exc:
        if not is_malformed_db_error(exc):
            raise
        _log.error(
            "state.db is corrupt while resolving session %s: %s", session_id, exc
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Session store is corrupt (database disk image is malformed). "
                "Sessions cannot be read until it is repaired — run "
                "`hermes doctor` for diagnosis."
            ),
        ) from exc


@list_router.get("/api/sessions")
def get_sessions(
    # ``le=100`` caps the page size (idea from #39200): an unbounded limit
    # lets one request drag every session row (plus correlated-subquery
    # preview work) out of SQLite in a single hit.
    limit: int = Query(20, ge=0, le=100),
    offset: int = Query(0, ge=0),
    min_messages: int = 0,
    archived: str = "exclude",
    order: str = "created",
    source: str = None,
    sources: str = None,
    exclude_sources: str = None,
    cwd_prefix: str = None,
    full: bool = False,
    profile: Optional[str] = None,
):
    """List sessions.

    ``archived`` controls how soft-archived sessions are treated:
    ``exclude`` (default) hides them, ``only`` returns just the archived ones
    (used by the desktop "Archived sessions" settings panel), and ``include``
    returns both.

    ``order`` controls pagination order: ``created`` (default, by original
    start time) or ``recent`` (by latest activity across the compression
    chain). ``recent`` keeps a long-running conversation on the first page
    after it auto-compresses into a fresh continuation id.

    Rows omit ``system_prompt``/``model_config`` (the payload-dominating
    fields no list UI reads) unless ``full=1`` is passed.
    """
    if archived not in ("exclude", "only", "include"):
        raise HTTPException(
            status_code=400,
            detail="archived must be one of: exclude, only, include",
        )
    if order not in ("created", "recent"):
        raise HTTPException(
            status_code=400,
            detail="order must be one of: created, recent",
        )
    profile_name: Optional[str] = None
    if profile:
        profile_name, _ = _cron_profile_home(profile)
    try:
        # Auto-archive is the only configured write on this GET path. Run it
        # through a dedicated maintenance connection, close that writer, then
        # open the listing connection read-only.
        _maybe_auto_archive_for_profile(profile)
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            min_message_count = max(0, min_messages)
            archived_only = archived == "only"
            include_archived = archived == "include"
            # Optional source scoping: ``source`` includes a single class,
            # ``sources`` includes any of several comma-separated classes, and
            # ``exclude_sources`` (comma-separated) drops classes. The desktop
            # uses these to split recents (exclude=cron) from the cron-jobs
            # section (source=cron) into two independent lists.
            source_list = [s.strip() for s in (sources or "").split(",") if s.strip()]
            exclude_list = [s.strip() for s in (exclude_sources or "").split(",") if s.strip()]
            sessions = db.list_sessions_rich(
                source=source or None,
                sources=source_list or None,
                exclude_sources=exclude_list or None,
                cwd_prefix=(cwd_prefix or None),
                limit=limit,
                offset=offset,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                order_by_last_active=order == "recent",
                # SQL-level projection: when the caller didn't ask for full
                # rows, skip the system_prompt blob inside SQLite too (pairs
                # with the API-level _strip_session_list_rows below).
                compact_rows=not full,
                include_pinned=True,
            )
            total = db.session_count(
                source=source or None,
                sources=source_list or None,
                cwd_prefix=(cwd_prefix or None),
                exclude_sources=exclude_list or None,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                exclude_children=True,
            )
            now = time.time()
            # Same ownership contract as get_session_detail: rows are stamped
            # with the serving profile even when the request wasn't explicitly
            # scoped, so default-profile rows never circulate unowned.
            row_profile = profile_name or _cron_default_profile()
            for s in sessions:
                s["is_active"] = (
                    s.get("ended_at") is None
                    and (now - s.get("last_active", s.get("started_at", 0))) < 300
                )
                s["profile"] = row_profile
                s["is_default_profile"] = row_profile == "default"
                # SQLite stores the flag as 0/1; expose a real JSON boolean.
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
        # 503, not 500: the store is busy, not gone. The desktop keeps the
        # sidebar it already has instead of reading a 500 as an authoritative
        # empty list. Retrying the OPEN here is deliberately not done — the
        # bounded retry lives in SessionDB's read-only constructor, so every
        # read-only opener gets it, not just this route.
        transient = is_transient_sqlite_error(exc)
        raise HTTPException(
            status_code=503 if transient else 500,
            detail=(
                "Session store is busy (disk I/O or lock). Retry; the list was not cleared."
                if transient
                else "Internal server error"
            ),
        ) from exc
    except Exception:
        _log.exception("GET /api/sessions failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@search_router.get("/api/sessions/search")
async def search_sessions(
    q: str = "",
    limit: int = 20,
    profile: Optional[str] = None,
    source: str = None,
    sources: str = None,
    exclude_sources: str = None,
):
    """Search sessions by ID plus full-text message content using FTS5.

    Direct session-id matches are surfaced first, then FTS message-content
    matches. Results are deduped by compression lineage, not by raw
    ``session_id``. Auto-compression rotates a conversation onto a fresh
    session id (and leaves the old segment's messages in the FTS index), so one
    logical chat can own many ``sessions`` rows that all match the same query.
    Branches also use ``parent_session_id``, but they are real alternate
    conversations; don't collapse branch-specific hits back into the parent.
    """
    if not q or not q.strip():
        return {"results": []}
    try:
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            safe_limit = max(1, min(int(limit or 20), 100))
            source_filter = source or None
            source_list = [s.strip() for s in (sources or "").split(",") if s.strip()]
            include_sources = [source_filter] if source_filter else (source_list or None)
            exclude_list = [s.strip() for s in (exclude_sources or "").split(",") if s.strip()]
            now = time.time()

            # Walk parent_session_id to the compression root, memoized so a
            # chain of compression segments only costs one walk. We deliberately
            # stop at branch/delegate edges: those sessions may diverge from the
            # parent and should remain searchable on their own.
            root_cache: dict = {}

            def compression_root(session_id: str) -> str:
                if not session_id:
                    return session_id
                if session_id in root_cache:
                    return root_cache[session_id]
                chain = []
                cur = session_id
                visited = set()
                root = session_id
                while cur and cur not in visited:
                    visited.add(cur)
                    chain.append(cur)
                    if cur in root_cache:
                        root = root_cache[cur]
                        break
                    try:
                        s = db.get_session(cur)
                    except Exception:
                        s = None
                    if not s:
                        root = cur
                        break
                    parent = s.get("parent_session_id") if isinstance(s, dict) else None
                    if not parent:
                        root = cur
                        break
                    try:
                        parent_session = db.get_session(parent)
                    except Exception:
                        parent_session = None
                    if not parent_session:
                        root = cur
                        break
                    parent_ended_at = parent_session.get("ended_at")
                    started_at = s.get("started_at")
                    is_compression_edge = (
                        parent_session.get("end_reason") == "compression"
                        and parent_ended_at is not None
                        and started_at is not None
                        and started_at >= parent_ended_at
                    )
                    if not is_compression_edge:
                        root = cur
                        break
                    cur = parent
                for node in chain:
                    root_cache[node] = root
                return root

            tip_cache: dict = {}

            def lineage_tip(root_id: str) -> str:
                if root_id in tip_cache:
                    return tip_cache[root_id]
                tip = root_id
                try:
                    resolved = db.get_compression_tip(root_id)
                    if resolved:
                        tip = resolved
                except Exception:
                    pass
                tip_cache[root_id] = tip
                return tip

            # Both ID matches and content matches share one keyspace, keyed by
            # compression lineage root, so an id-hit and a content-hit on the
            # same logical conversation collapse to a single result. The first
            # hit for a lineage wins; ID matches run first and take priority.
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
                    payload.update(
                        {
                            "id": row.get("id") or sid,
                            "source": row.get("source"),
                            "model": row.get("model"),
                            "title": row.get("title"),
                            "started_at": row.get("started_at"),
                            "ended_at": row.get("ended_at"),
                            "last_active": row.get("last_active") or row.get("started_at"),
                            "is_active": (
                                row.get("ended_at") is None
                                and (now - (row.get("last_active") or row.get("started_at") or 0)) < 300
                            ),
                            "message_count": row.get("message_count") or 0,
                            "tool_call_count": row.get("tool_call_count") or 0,
                            "input_tokens": row.get("input_tokens") or 0,
                            "output_tokens": row.get("output_tokens") or 0,
                            "preview": row.get("preview"),
                            "parent_session_id": row.get("parent_session_id"),
                            "archived": bool(row.get("archived")),
                        }
                    )
                else:
                    payload["id"] = sid
                seen[root] = payload

            # Direct ID matches first: users often paste a session id from CLI,
            # logs, or another Hermes surface. FTS can't find those unless the
            # id happens to appear in message text. search_sessions_by_id is
            # SQL-bounded, so this stays cheap even with thousands of sessions.
            for row in db.search_sessions_by_id(
                q,
                limit=safe_limit,
                include_archived=True,
                source=source_filter,
                sources=source_list or None,
                exclude_sources=exclude_list or None,
            ):
                sid = row.get("id")
                preview = (row.get("preview") or "").strip()
                snippet = preview or f"Session ID: {sid}"
                add_lineage_result(
                    sid,
                    {
                        "snippet": snippet,
                        "role": None,
                        "source": row.get("source"),
                        "model": row.get("model"),
                        "session_started": row.get("started_at"),
                    },
                )

            # Auto-add prefix wildcards so partial words match
            # e.g. "nimb" → "nimb*" matches "nimby"
            # Preserve quoted phrases and existing wildcards as-is
            import re
            terms = []
            for token in re.findall(r'"[^"]*"|\S+', q.strip()):
                if token.startswith('"') or token.endswith("*"):
                    terms.append(token)
                else:
                    terms.append(token + "*")
            prefix_query = " ".join(terms)
            # Over-fetch so lineage dedup can still surface `limit` distinct
            # conversations even when several hits collapse onto one root.
            fetch_limit = max(safe_limit * 5, 50)
            matches = db.search_messages(
                query=prefix_query,
                source_filter=include_sources,
                exclude_sources=exclude_list or None,
                limit=fetch_limit,
                fields=(
                    "session_id",
                    "role",
                    "snippet",
                    "source",
                    "model",
                    "session_started",
                ),
            )

            for m in matches:
                if len(seen) >= safe_limit:
                    break
                add_lineage_result(
                    m["session_id"],
                    {
                        "snippet": m.get("snippet", ""),
                        "role": m.get("role"),
                        "source": m.get("source"),
                        "model": m.get("model"),
                        "session_started": m.get("session_started"),
                    },
                )
            return {"results": list(seen.values())}
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception:
        _log.exception("GET /api/sessions/search failed")
        raise HTTPException(status_code=500, detail="Search failed")


@manage_router.post("/api/sessions/bulk-delete")
async def bulk_delete_sessions_endpoint(body: BulkDeleteSessions):
    """Delete every session in ``body.ids`` in a single DB transaction.

    Backs the dashboard's bulk-select-and-delete flow on the sessions
    page. POST (not DELETE) because most HTTP clients refuse to send a
    request body on DELETE and a body is the natural shape for a list
    of IDs — Starlette accepts both, but POSTing a list keeps proxies,
    curl, and the browser ``fetch`` API consistent.

    Per-row contract matches :meth:`SessionDB.delete_sessions`:

    * Unknown IDs are silently skipped (the response ``deleted`` count
      reflects what really happened, not the input length). This is
      deliberate — UI selection state can race against another tab's
      delete, and we'd rather succeed-on-the-rest than fail-the-whole-
      batch.
    * Children of every deleted parent are orphaned, not cascade-
      deleted.
    * Active and archived sessions ARE deleted when explicitly
      selected — unlike ``DELETE /api/sessions/empty``, the user
      hand-picked the rows so we trust the selection.
    * Like the other session-delete endpoints, this does NOT pass a
      ``sessions_dir`` through; on-disk transcript / request-dump
      cleanup runs at the CLI/agent layer on the next prune pass.

    The response carries the actual deleted count, so the dashboard
    can surface it in a toast. The IDs that were removed are not
    echoed back because the client already knows what it asked to
    delete (unknown IDs are silently skipped — see contract above)
    and can prune its in-memory list directly from the request.
    """
    # Enforce a hard cap so a runaway/typo'd selection can't lock the
    # DB writer for an extended window. The dashboard pages 20 rows
    # at a time; 500 covers a "select all on every page in a
    # reasonable scrollback" worst case without opening the door to
    # multi-thousand-row transactions.
    if len(body.ids) > 500:
        raise HTTPException(
            status_code=400,
            detail="ids must contain at most 500 entries",
        )
    def _delete() -> int:
        db = _open_session_db_for_profile(body.profile, read_only=False)
        try:
            return db.delete_sessions(body.ids)
        finally:
            db.close()

    deleted = await asyncio.to_thread(_delete)
    return {"ok": True, "deleted": deleted}


@manage_router.post("/api/sessions/import")
async def import_sessions_endpoint(request: Request):
    """Import one or more sessions exported from the dashboard or CLI.

    This is intentionally separate from ``/api/ops/import``: that endpoint
    restores a whole Hermes backup archive, while this endpoint is scoped to
    session rows/messages and is safe to use from the Sessions page.
    """
    try:
        raw_body = await _read_session_import_body(request)
        body = SessionImport.model_validate_json(raw_body)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid session import payload") from exc

    try:
        result = await asyncio.to_thread(_import_sessions_for_profile, body.profile, body.sessions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not result.get("ok", False):
        raise HTTPException(status_code=400, detail=result)
    return result


@manage_router.get("/api/sessions/empty/count")
async def count_empty_sessions_endpoint(profile: Optional[str] = None):
    """Return the number of empty, ended, non-archived sessions.

    Drives the dashboard's "Delete empty (N)" button — when N is 0 the
    UI hides the affordance so users aren't presented with a button
    that does nothing. Cheap, single-COUNT query.
    """
    def _count() -> int:
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            return db.count_empty_sessions()
        finally:
            db.close()

    return {"count": await asyncio.to_thread(_count)}


@manage_router.delete("/api/sessions/empty")
async def delete_empty_sessions_endpoint(profile: Optional[str] = None):
    """Delete every empty, ended, non-archived session in a single
    transaction.

    Safety contract mirrors :meth:`SessionDB.delete_empty_sessions`:

    * "Empty" means the session owns no rows in ``messages`` at all — not
      merely ``message_count == 0``. A rewound or in-place-compacted chat
      keeps its dropped turns as soft-archived (``active = 0``) rows while
      the counter reads zero, and those rows are the only recoverable copy
      of the transcript (#95868).
    * Active sessions are skipped (``ended_at IS NULL``) so a live
      agent isn't yanked mid-handshake.
    * Archived sessions are skipped — the user explicitly chose to
      keep those rows.
    * Children of deleted parents are orphaned, not cascade-deleted.

    Like the single-session ``DELETE /api/sessions/{id}`` endpoint
    below, this doesn't pass a ``sessions_dir`` through — the on-disk
    transcript / request-dump cleanup is wired at the CLI/agent layer
    but the web server historically leaves file cleanup to the next
    prune-on-startup pass. Matching that pre-existing trade-off keeps
    the two delete endpoints' DB-vs-disk behaviour consistent.
    """
    def _delete() -> int:
        db = _open_session_db_for_profile(profile, read_only=False)
        try:
            return db.delete_empty_sessions()
        finally:
            db.close()

    deleted = await asyncio.to_thread(_delete)
    return {"ok": True, "deleted": deleted}


@manage_router.get("/api/sessions/stats")
async def get_session_stats(profile: Optional[str] = None):
    """Session-store statistics for the Sessions page (mirrors `hermes sessions stats`).

    Registered before ``/api/sessions/{session_id}`` so the literal ``stats``
    path isn't captured as a session id by the parameterized route.
    """
    db = _open_session_db_for_profile(profile, read_only=True)
    try:
        total = db.session_count(include_archived=True)
        active_store = db.session_count(include_archived=False)
        archived = db.session_count(archived_only=True)
        messages = db.message_count()
        by_source: Dict[str, int] = {}
        try:
            by_source = db.session_count_by_source(
                include_archived=True,
                exclude_children=True,
            )
        except Exception:
            pass
        return {
            "total": total,
            "active_store": active_store,
            "archived": archived,
            "messages": messages,
            "by_source": by_source,
        }
    finally:
        db.close()


@manage_router.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str, profile: Optional[str] = None):
    db = _open_session_db_for_profile(profile, read_only=True)
    try:
        sid = _resolve_session_id(db, session_id)
        session = db.get_session(sid) if sid else None
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        # Always stamp the owning profile — the serving profile is known even
        # when the request carries no ``?profile=`` (it's this process's own
        # profile). Stamping only on explicit ``?profile=`` left rows for the
        # default/primary profile systematically unowned, so multi-profile
        # clients resolved them to whichever gateway happened to be active
        # (cross-profile open asymmetry, #67603 family).
        session["profile"] = (
            _cron_profile_home(profile)[0] if profile else _cron_default_profile()
        )
        session["is_default_profile"] = session["profile"] == "default"
        return session
    finally:
        db.close()


@manage_router.get("/api/sessions/{session_id}/latest-descendant")
async def get_session_latest_descendant(
    session_id: str,
    profile: Optional[str] = None,
):
    def _lookup():
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            return _session_latest_descendant(session_id, db)
        finally:
            db.close()

    latest, path = await asyncio.to_thread(_lookup)
    if not latest:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "requested_session_id": path[0] if path else session_id,
        "session_id": latest,
        "path": path,
        "changed": bool(path and latest != path[0]),
    }


@manage_router.get("/api/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    profile: Optional[str] = None,
    limit: Optional[int] = Query(None, ge=0),
    offset: int = Query(0, ge=0),
    order: Optional[str] = Query(None),
    include_compacted: bool = Query(False),
    include_media: bool = Query(False),
):
    if order not in (None, "oldest", "latest"):
        raise HTTPException(
            status_code=400,
            detail="order must be one of: oldest, latest",
        )
    # Direct (non-FastAPI) callers omitting ``include_media`` would otherwise
    # see the Query() descriptor object, which is truthy — normalize it so the
    # opt-in default holds regardless of the calling convention.
    if not isinstance(include_media, bool):
        include_media = False

    def _read():
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            sid = _resolve_session_id(db, session_id)
            if not sid:
                return None
            sid = db.resolve_resume_session_id(sid)
            # Always page this endpoint. An omitted limit used to load an
            # entire transcript, which can be hundreds of thousands of rows
            # for a runaway session and exhaust the dashboard process. Keep
            # explicit pagination anchored at the start, while the default
            # dashboard view returns the latest page in chronological order.
            default_page = limit is None
            latest_page = order == "latest" or (order is None and default_page)
            _limit = 500 if default_page else min(limit, 500)
            return sid, _limit, db.get_messages(
                sid,
                limit=_limit,
                offset=offset,
                latest=latest_page,
                include_compacted=include_compacted,
            )
        finally:
            db.close()

    result = await asyncio.to_thread(_read)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    sid, _limit, messages = result
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
    if include_media:
        # History media projection (D5): per-message refs derived server-side
        # from stored data, bounded to the page just read. Stamped onto the
        # same row objects the response returns; every message gets the key
        # under the opt-in (empty list when it tags nothing) so the consumer
        # shape is uniform. Default calls are untouched.
        from hermes_cli.session_media_projection import build_media_refs_for_messages

        projection = build_media_refs_for_messages(projected_messages)
        for index, message in enumerate(projected_messages):
            message["media"] = projection.get(index, [])
    return {
        "session_id": sid,
        "messages": projected_messages,
        "pagination": {
            "limit": _limit,
            "offset": offset,
            "order": order or ("latest" if limit is None else "oldest"),
            "returned": len(projected_messages),
        },
    }


@manage_router.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, profile: Optional[str] = None):
    # ``profile`` deletes a session belonging to another (local) profile by
    # opening its state.db directly. Remote profiles never reach here — the
    # desktop routes their DELETE to the remote backend. Omit for current/default.
    def _delete():
        db = _open_session_db_for_profile(profile, read_only=False)
        try:
            # Resolve exact ids / unique prefixes like every other session endpoint
            # (detail, messages, rename, export all do). A session that no longer
            # exists is an idempotent success: DELETE's contract is "ensure it's
            # gone", and the desktop optimistically removes the row then RESTORES it
            # on any error — so a 404 on an already-absent row resurrected a ghost
            # row and surfaced "session not found". /goal + auto-compression churn
            # leaves transient empty rows (reaped by empty-session hygiene) that
            # race the sidebar snapshot, which is exactly when this fired. Mirrors
            # the bulk-delete endpoint, which already treats ghost ids as success.
            sid = _resolve_session_id(db, session_id)
            if not sid:
                return {"ok": True, "already_absent": True}
            db.delete_session(sid)
            return {"ok": True}
        finally:
            db.close()

    return await asyncio.to_thread(_delete)


@manage_router.post("/api/sessions/owner-backfill")
async def backfill_session_owner_profiles(body: SessionOwnerBackfill):
    """Stamp legacy ``profile_name = NULL`` session rows with this store's own
    serving-profile identity (#94724 legacy-session migration).

    Pre-#95407 rows never recorded an owning profile. That was fine while one
    backend served everything, but a Desktop with registry topology (≥2
    registered connections) fails closed on unowned rows by design — leaving
    every pre-campaign session unresumable with no migration path. Each
    profile's ``state.db`` belongs to exactly one profile, so stamping that
    store's own name is a single-match backfill, never a guess; the value
    written is the SAME serving-profile identity the list endpoints already
    stamp onto outgoing rows (``row_profile`` in ``get_sessions``). Idempotent
    and one-shot-per-row: non-NULL owners are never overwritten and a second
    call reports 0.
    """
    profile_name: Optional[str] = None
    if body.profile:
        profile_name, _ = _cron_profile_home(body.profile)
    stamp = profile_name or _cron_default_profile()

    def _backfill():
        db = _open_session_db_for_profile(body.profile, read_only=False)
        try:
            return db.backfill_null_session_profiles(stamp)
        finally:
            db.close()

    try:
        stamped = await asyncio.to_thread(_backfill)
    except HTTPException:
        raise
    except Exception:
        _log.exception("POST /api/sessions/owner-backfill failed")
        raise HTTPException(status_code=500, detail="Internal server error")

    if stamped:
        _log.info(
            "owner-backfill: stamped %d legacy NULL-profile session row(s) with profile %r",
            stamped,
            stamp,
        )
    return {"ok": True, "stamped": stamped, "profile": stamp}


@manage_router.patch("/api/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, body: SessionRename):
    """Update a session: rename, archive, hide, pin, and/or mark read/unread.

    ``title`` renames (empty/null clears the title); ``archived`` soft-hides or
    restores the session; ``hidden`` controls generic list visibility;
    ``pinned`` sets the durable keep flag (exempts the
    session from the auto-archive sweep); ``unread`` toggles the read-state
    watermark (True = explicitly unread, False = read up to now — see
    ``SessionDB.set_session_read``). Any field may be omitted. ``profile``
    targets another profile's session.
    """
    db = _open_session_db_for_profile(body.profile, read_only=False)
    try:
        sid = _resolve_session_id(db, session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        if (
            body.title is None
            and body.archived is None
            and body.hidden is None
            and body.pinned is None
            and body.unread is None
        ):
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
        if body.archived is not None:
            db.set_session_archived(sid, body.archived)
        if body.hidden is not None:
            db.set_session_hidden(sid, body.hidden)
        if body.pinned is not None:
            db.set_session_pinned(sid, body.pinned)
        if body.unread is not None:
            db.set_session_read(sid, read=not body.unread)
        result = {"ok": True, "title": db.get_session_title(sid) or ""}
        if body.archived is not None:
            result["archived"] = bool(body.archived)
        if body.hidden is not None:
            result["hidden"] = bool(body.hidden)
        if body.pinned is not None:
            result["pinned"] = bool(body.pinned)
        if body.unread is not None:
            result["unread"] = bool(body.unread)
        return result
    finally:
        db.close()


@manage_router.get("/api/sessions/{session_id}/export")
async def export_session_endpoint(session_id: str, profile: Optional[str] = None):
    """Stream a single session (metadata + messages) as JSON."""
    def _prepare_export():
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            sid = _resolve_session_id(db, session_id)
            return (sid, db.get_session(sid)) if sid else None
        finally:
            db.close()

    prepared = await asyncio.to_thread(_prepare_export)
    if prepared is None or prepared[1] is None:
        raise HTTPException(status_code=404, detail="Session not found")

    sid, session = prepared

    def _stream_export():
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            metadata = json.dumps(
                jsonable_encoder(session),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            yield metadata[:-1] + ',"messages":['

            # Keyset pagination (id > last_seen): O(n) total over the
            # transcript, vs OFFSET's O(n²) on huge sessions.
            last_id = None
            first = True
            while True:
                messages = db.get_messages(
                    sid,
                    limit=500,
                    after_id=last_id if last_id is not None else 0,
                )
                for message in messages:
                    if not first:
                        yield ","
                    yield json.dumps(
                        jsonable_encoder(message),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    first = False
                if len(messages) < 500:
                    break
                last_id = messages[-1].get("id")
                if last_id is None:
                    break  # defensive: cannot keyset without row ids

            yield "]}"
        finally:
            db.close()

    return StreamingResponse(
        _stream_export(),
        media_type="application/json",
    )


@manage_router.post("/api/sessions/prune")
async def prune_sessions_endpoint(body: SessionPrune):
    """Delete ended sessions matching filters without blocking the event loop."""
    return await asyncio.to_thread(_prune_sessions, body)
