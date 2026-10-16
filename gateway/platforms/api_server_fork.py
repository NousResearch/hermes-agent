"""``POST /api/sessions/{session_id}/fork`` for the api_server adapter (split out of api_server.py).

CLI ``/branch`` semantics: the child is created first, then the source is ended as branched
(#11030). A temporary chat is refused: its transcript must never land in a saved session.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from contextlib import suppress

from aiohttp import web


async def handle_fork_session(adapter, request: web.Request) -> web.Response:
    """POST /api/sessions/{session_id}/fork — branch via current SessionDB primitives."""
    from gateway.platforms.api_server import _error_response
    source_id = request.match_info["session_id"]
    source, err = await adapter._get_existing_session_or_404(source_id)
    if err:
        return err
    db = await adapter._ensure_session_db_async()
    if await asyncio.to_thread(db.is_session_temporary, source_id):
        # A temporary chat's transcript must never be copied into a saved, listed session.
        return _error_response("temporary chats can't be branched", 409, code="temporary_session_cannot_branch")
    body, err = await adapter._read_json_body(request)
    if err:
        return err
    fork_id = str(body.get("id") or body.get("session_id") or f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}").strip()
    if not fork_id or re.search(r'[\r\n\x00]', fork_id):
        return _error_response("Invalid session ID", 400, code="invalid_session_id")
    if await asyncio.to_thread(db.get_session, fork_id):
        return _error_response(f"Session already exists: {fork_id}", 409, code="session_exists")

    # CLI /branch semantics: create the child, then end the original as branched (child first, so
    # a failed create never leaves the source ended with no fork, #11030).
    # ``_branched_from`` is the durable branch marker (same as CLI /branch): with the child created
    # first, the timestamp fallback in _BRANCH_CHILD_SQL (child.started_at >= parent.ended_at) no
    # longer holds, and an unmarked child would vanish from default session listings.
    await asyncio.to_thread(
        db.create_session, fork_id, "api_server", model=source.get("model"),
        system_prompt=source.get("system_prompt"), parent_session_id=source_id,
        model_config={"_branched_from": source_id})
    await asyncio.to_thread(db.end_session, source_id, "branched")
    messages = await asyncio.to_thread(db.get_messages, source_id)
    await asyncio.to_thread(db.replace_messages, fork_id, messages)
    title = body.get("title")
    if title is None:
        base = source.get("title") or "fork"
        title = f"{base} fork"
        with suppress(Exception):
            title = await asyncio.to_thread(db.get_next_title_in_lineage, base)
    try:
        await asyncio.to_thread(db.set_session_title, fork_id, str(title))
    except ValueError as exc:
        return _error_response(str(exc), 400, code="invalid_title")
    fork = await asyncio.to_thread(db.get_session, fork_id) or {"id": fork_id, "parent_session_id": source_id}
    return web.json_response({"object": "hermes.session", "session": adapter._session_response(fork)}, status=201)
