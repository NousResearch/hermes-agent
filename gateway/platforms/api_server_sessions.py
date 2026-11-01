"""Read-only gateway session metadata, including scheduler admission identity."""

import asyncio
import logging

from aiohttp import web

from gateway.platforms.api_server_bot_chat import reset_local_title_owner


def _attach_cron_identity(db, sessions):
    for session in sessions:
        if session.get("source") == "cron":
            session.update(db.cron_run_identity(session["id"]) or {})


async def list_sessions(adapter, request):
    """GET /api/sessions — list persisted Hermes sessions."""
    from gateway.platforms.api_server import _coerce_request_bool
    db = await adapter._ensure_session_db_async()
    if db is None:
        return adapter._session_db_unavailable()
    limit = adapter._parse_nonnegative_int(request.query.get("limit"), default=50, maximum=200)
    offset = adapter._parse_nonnegative_int(request.query.get("offset"), default=0, maximum=1_000_000)
    source = request.query.get("source") or None
    include_children = _coerce_request_bool(request.query.get("include_children"), default=False)
    # Exact-title lookup (`hermes peer dm` -> canonical "Bot Chat"). include_hidden is honored
    # ONLY with a title filter: a blanket hidden listing stays off this client surface.
    title_filter = (request.query.get("title") or "").strip() or None
    include_hidden = bool(title_filter) and _coerce_request_bool(
        request.query.get("include_hidden"), default=False)

    async def _list() -> list:
        # include_pinned back-fills pins past the recency window; search_query pushes the
        # title needle into SQL (substring) so a hidden/old row is found, exact match below.
        rows = await asyncio.to_thread(
            db.list_sessions_rich, source=source, limit=limit, offset=offset,
            include_children=include_children, order_by_last_active=True, include_pinned=True,
            search_query=title_filter, include_hidden=include_hidden)
        if title_filter:
            rows = [s for s in rows if (s.get("title") or "").strip() == title_filter]
        return rows

    sessions = await _list() or (await reset_local_title_owner(db, title_filter, include_hidden, source))
    if title_filter and not sessions:
        # A canonical Bot Chat auto-archived by the orphan reaper would make `hermes peer dm`
        # mint transient sessions: resurrect and re-list; deliberate archives stay put.
        try:
            # Recoverable-archive resurrection (#92687): a canonical Bot Chat archived by the ws-orphan
            # reaper / older agent cleanup is invisible to list_sessions_rich (include_archived=False),
            # which would fail `hermes peer dm` resolution and mint transient sessions — same accident
            # the tui_gateway lookups heal.
            from tools.bot_mode_probe import BOT_CHAT_TITLE

            def _resurrect() -> bool:
                # Lookup + unarchive (a WRITE with the full write patience) as one worker-thread
                # hop: a contended lock parks this thread, never the event loop (#113772).
                stale = db.get_session_by_title(title_filter)
                return bool(stale and stale.get("archived") and db.unarchive_recoverable_session(stale["id"]))

            if title_filter == BOT_CHAT_TITLE and await asyncio.to_thread(_resurrect):
                sessions = await _list()
        except Exception:
            logging.getLogger(__name__).debug("Bot Chat archive recovery failed", exc_info=True)
    await asyncio.to_thread(_attach_cron_identity, db, sessions)
    # Back-filled pins arrive PAST the limit: counting them would report a page that doesn't exist.
    windowed = sum(1 for s in sessions if not s.get("pinned"))
    return web.json_response({
        "object": "list", "data": [adapter._session_response(s) for s in sessions],
        "limit": limit, "offset": offset, "has_more": windowed >= limit})


async def get_session(adapter, request):
    session, err = await adapter._get_existing_session_or_404(request.match_info["session_id"])
    if err:
        return err
    if session.get("source") == "cron":
        db = await adapter._ensure_session_db_async()
        await asyncio.to_thread(_attach_cron_identity, db, [session])
    return web.json_response({"object": "hermes.session", "session": adapter._session_response(session)})
