"""Manual compaction and the context meter for ``/api/sessions/{session_id}``.

``POST /api/sessions/{session_id}/compress`` is the API twin of ``/compress``: it runs the
shared manual core (``agent.conversation_compression_manual.compress_now``) on a throwaway
agent built like an API turn's, and ``_compress_context``'s own commit is the only write (in
place, or a rotated continuation whose id the response returns). The conversation's
cross-process turn lease is held for the whole attempt, so a compaction never interleaves
with a running turn: a held lease is refused with 409 instead of waited on.

``with_context_meter`` adds ``context_tokens`` / ``context_limit`` to a session read.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Callable, Optional

from aiohttp import web

from agent.conversation_compression import finalize_context_engine_compression_notification
from agent.conversation_compression_manual import MIN_MESSAGES, CompressRequest, compress_now
from gateway.session_mutation_compress import _request_tokens
from hermes_state_pidns import holder_namespace_token

logger = logging.getLogger("gateway.platforms.api_server")

# A summary call plus commit; a holder that dies is reclaimed by PID long before this.
_COMPRESS_LEASE_TTL_S = 900.0
# Same budget as the dashboard's /api/model/info probe: an unreachable base_url must not hold a read.
_CONTEXT_LIMIT_BUDGET_S = 5.0
_API_SESSION_SOURCE = "api_server"


def _http_routes(adapter, *, error_response: Callable[..., web.Response]) -> list[tuple[str, str, Any]]:
    async def compress(request: web.Request) -> web.Response:
        return await handle_session_compress(adapter, request, error_response=error_response)
    return [("POST", "/api/sessions/{session_id}/compress", compress)]


def _parse_request(body: dict[str, Any]) -> CompressRequest | str:
    """Body → request, or the 400 message. ``instructions`` / ``focus_topic`` is free text for the
    summarizer, never re-parsed as ``/compress`` flags (API text must not turn into ``here 2``)."""
    focus = body.get("instructions", body.get("focus_topic"))
    if focus is not None and not isinstance(focus, str):
        return "instructions must be a string"
    preview = body.get("preview", False)
    if not isinstance(preview, bool):
        return "preview must be a boolean"
    return CompressRequest(preview=preview, focus_topic=(focus or "").strip() or None)


def _payload(status: str, session_id: str, before_tokens: int, after_tokens: int,
             removed: int = 0, summary: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    out = {"object": "hermes.session.compress", "status": status, "session_id": session_id,
           "before_tokens": before_tokens, "after_tokens": after_tokens, "messages_removed": removed}
    if summary:
        out["summary"] = {k: summary.get(k) for k in ("headline", "token_line", "note")}
    return out


async def handle_session_compress(adapter, request: web.Request, *, error_response) -> web.Response:
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    session, err = await adapter._get_existing_session_or_404(request.match_info["session_id"])
    if err:
        return err
    if session.get("source") != _API_SESSION_SOURCE:
        # Another platform's live agent caches this history; compacting under it would fork the two.
        return error_response("Only API-server sessions can be compacted here", 409,
                              code="session_platform_owned")
    body, err = await adapter._read_json_body(request) if request.can_read_body else ({}, None)
    if err:
        return err
    compress_request = _parse_request(body)
    if isinstance(compress_request, str):
        return error_response(compress_request, 400, code="invalid_compress_request")
    db = await adapter._ensure_session_db_async()
    if db is None:
        return adapter._session_db_unavailable()
    # to_thread copies the request's context, so the profile scope the prefix middleware bound
    # (home, config, credentials) is the one the compaction agent resolves under.
    status, payload = await asyncio.to_thread(_compress_session, adapter, db, session["id"], compress_request)
    if status == 200:
        return web.json_response(payload)
    return error_response(payload["message"], status, code=payload["code"])


def _compress_session(adapter, db, session_id: str, request: CompressRequest) -> tuple[int, dict[str, Any]]:
    """Admit the conversation's turn lease, compact, release. ``(http_status, body)``."""
    target = db.get_compression_tip(session_id) or session_id
    holder = f"pid={os.getpid()}{holder_namespace_token()}:api-compress={uuid.uuid4().hex}"
    if not db.try_acquire_session_turn_lease(target, holder, ttl_seconds=_COMPRESS_LEASE_TTL_S):
        return 409, {"message": "A turn is running on this session; retry when it finishes",
                     "code": "session_active_turn"}
    try:
        return _compress_admitted(adapter, db, target, request)
    finally:
        db.release_session_turn_lease(target, holder)


def _compress_admitted(adapter, db, target: str, request: CompressRequest) -> tuple[int, dict[str, Any]]:
    # Full transcript (tool results included) like gateway /compress: tool-result pruning needs them.
    history = [m for m in db.get_messages_as_conversation(target, repair_alternation=True)
               if m.get("role") in {"user", "assistant", "tool"}]
    row = db.get_session(target)
    if len(history) < MIN_MESSAGES:
        tokens = _request_tokens(None, history, row)
        return 200, _payload("nothing_to_do", target, tokens, tokens)
    agent = _build_compression_agent(adapter, target, row)
    try:
        if str(getattr(agent, "api_mode", "") or "").lower() == "codex_app_server":
            return 409, {"message": "This runtime compacts its server-side thread; use /compress in its chat",
                         "code": "compression_unsupported"}
        result = compress_now(agent, history, request, system_message="", skip_without_window=True,
                              task_id=target)
        if result.status == "lock_skipped":
            return 409, {"message": "Another compaction of this session is in progress",
                         "code": "session_compression_in_progress"}
        if result.status != "compressed":
            return 200, _payload(result.status, target, result.before_tokens, result.after_tokens)
        # _compress_context already committed (in place, or a published continuation).
        finalize_context_engine_compression_notification(agent, committed=True)
        return 200, _payload("compressed", str(agent.session_id or target), result.before_tokens,
                             result.after_tokens, result.removed, result.summary)
    finally:
        finalize_context_engine_compression_notification(agent, committed=False)
        adapter._memory_sessions.checkin(agent)


def _build_compression_agent(adapter, session_id: str, row: Optional[dict[str, Any]]):
    """The API turn's agent (same runtime precedence), seeded with the stored prompt the next turn
    sends, so the commit boundary does not rebuild it and the cached prefix survives."""
    from gateway.run import _seed_hygiene_system_prompt
    agent = adapter._create_agent(session_id=session_id, session_model=adapter._stored_session_model(row))
    _seed_hygiene_system_prompt(agent, row)
    agent._print_fn = lambda *a, **kw: None
    agent._end_session_on_close = False
    return agent


async def with_context_meter(adapter, session: dict[str, Any]) -> dict[str, Any]:
    """``session`` plus ``context_tokens`` (estimated next-request size: stored prompt, pinned tools
    and the live transcript) and ``context_limit`` (the window the session's model resolves to).
    A limit that cannot be resolved inside the budget is omitted, not guessed."""
    db = await adapter._ensure_session_db_async()
    if db is None:
        return session
    target = await asyncio.to_thread(db.get_compression_tip, session["id"]) or session["id"]
    row = session if target == session["id"] else (await asyncio.to_thread(db.get_session, target) or session)
    messages = await asyncio.to_thread(db.get_messages_as_conversation, target)
    meter = {"context_tokens": _request_tokens(None, messages, row)}
    model = adapter._stored_session_model(session)
    from gateway.run import _resolve_gateway_model_context
    try:
        resolved = await asyncio.wait_for(
            asyncio.to_thread(_resolve_gateway_model_context, model), _CONTEXT_LIMIT_BUDGET_S)
        meter["context_limit"] = resolved.context_length
    except TimeoutError:
        logger.warning("context limit for session %s not resolved within %.0fs", session["id"],
                       _CONTEXT_LIMIT_BUDGET_S)
    return {**session, **meter}
