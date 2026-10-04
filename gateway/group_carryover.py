"""Group carry-over: fork a group chat's session seeded with a synopsis the user approved in a DM.

Runs on the gateway loop (scheduled by ``tools/group_carryover_tool.py`` from the agent's worker
thread). The private session is never touched here — the DM only records the tool call/result.

Order is post-then-fork: the visible group message goes out first, so a failed send leaves the
group's session exactly as it was. The fork is a conversation boundary (the same funnel ``/new``
uses) for the requester's group lane, and the fresh session is seeded with one user→assistant pair,
so the group agent's first real turn builds a brand-new system prompt over a history that already
holds the synopsis — no cached prefix is rewritten and strict role alternation holds. Other live
per-user lanes in the same group (``group_sessions_per_user``) keep their history and get the
synopsis appended as a user-role mirror, the append-only pattern cron delivery already uses.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gateway.config import Platform
from gateway.session import SessionSource

logger = logging.getLogger(__name__)


@dataclass
class CarryoverRequest:
    """Everything the gateway needs, resolved and checked on the tool side."""
    platform: str
    chat_id: str
    chat_name: str
    chat_type: str
    user_id: str
    user_name: str
    synopsis: str
    profile: Optional[str] = None
    user_id_alt: Optional[str] = None
    thread_id: Optional[str] = None
    # The chat's live keyed sessions (``id`` / ``session_key``) as recorded in state.db.
    live_sessions: List[Dict[str, Any]] = field(default_factory=list)


def group_announcement(user_name: str, synopsis: str) -> str:
    """The visible message posted into the group (also the seeded assistant turn)."""
    who = user_name or "A group member"
    return (f"📎 {who} brought over some context from a private chat with me "
            f"(they reviewed it before it was shared):\n\n{synopsis}\n\n"
            "Pick it up here — I'll carry on from this with everyone.")


def seed_user_turn(user_name: str, synopsis: str) -> str:
    """The seeded user turn that frames the synopsis for the group agent."""
    who = user_name or "A group member"
    return (f"[Group carry-over] {who} brought this over from a private chat with you and approved "
            "sharing it with this group. Treat it as the starting context for this group "
            f"conversation; it supersedes anything older you may recall on the same topic.\n\n{synopsis}")


def _group_source(req: CarryoverRequest) -> SessionSource:
    return SessionSource(
        platform=Platform(req.platform), chat_id=str(req.chat_id), chat_name=req.chat_name or None,
        chat_type=req.chat_type or "group", user_id=req.user_id or None, user_name=req.user_name or None,
        user_id_alt=req.user_id_alt or None, thread_id=req.thread_id or None, profile=req.profile or None)


async def _send_announcement(adapter: Any, req: CarryoverRequest, text: str) -> Optional[str]:
    """Post the visible message; None on success, else the error text."""
    metadata = {"thread_id": req.thread_id} if req.thread_id else None
    try:
        result = await adapter.send(chat_id=str(req.chat_id), content=text, metadata=metadata)
    except Exception as exc:  # noqa: BLE001 - reported to the model, nothing forked
        return f"send failed: {exc}"
    if not getattr(result, "success", False):
        return f"send failed: {getattr(result, 'error', None) or 'unknown error'}"
    return None


async def _fork_group_lane(runner: Any, source: SessionSource, session_key: str) -> Any:
    """Conversation boundary for the requester's group lane (the ``/new`` funnel, minus the banner)."""
    invalidate = getattr(runner, "_invalidate_session_run_generation", None)
    if callable(invalidate):
        invalidate(session_key, reason="group_carryover")
    release = getattr(runner, "_release_running_agent_state", None)
    if callable(release):
        release(session_key)
    cleanup = getattr(runner, "_cleanup_old_agent_for_reset", None)
    if callable(cleanup):
        with contextlib.suppress(Exception):
            await cleanup(session_key)
    evict = getattr(runner, "_evict_cached_agent", None)
    if callable(evict):
        evict(session_key)
    clear = getattr(runner, "_clear_conversation_scope", None)
    if callable(clear):
        clear(session_key, reason="group_carryover")
    store = runner.async_session_store
    entry = await store.reset_session(session_key)
    if entry is None:  # the group never had a lane for this user: open one
        entry = await store.get_or_create_session(source, force_new=True)
    return entry


async def carry_into_group(runner: Any, adapter: Any, req: CarryoverRequest) -> Dict[str, Any]:
    """Post the approved synopsis to the group and fork the group's session seeded with it."""
    source = _group_source(req)
    session_key = runner._session_key_for_source(source)
    is_running = getattr(runner, "_is_session_running", None)
    if callable(is_running) and is_running(session_key):
        return {"error": "The group chat is mid-reply right now; nothing was posted. Try again in a moment."}

    announcement = group_announcement(req.user_name, req.synopsis)
    send_error = await _send_announcement(adapter, req, announcement)
    if send_error:
        return {"error": f"Could not post to the group ({send_error}); nothing was forked."}

    try:
        entry = await _fork_group_lane(runner, source, session_key)
        store = runner.async_session_store
        await store.append_to_transcript(entry.session_id, {"role": "user", "content": seed_user_turn(req.user_name, req.synopsis)})
        await store.append_to_transcript(entry.session_id, {"role": "assistant", "content": announcement})
    except Exception as exc:  # noqa: BLE001 - the post already went out; say so precisely
        logger.warning("Group carry-over: posted to %s:%s but seeding failed", req.platform, req.chat_id, exc_info=True)
        return {"success": False, "posted": True, "error": f"Posted to the group, but seeding its session failed: {exc}"}

    mirrored = 0
    other_lanes = [s["id"] for s in req.live_sessions
                   if s.get("session_key") != session_key and s.get("id") != entry.session_id]
    if other_lanes:
        from gateway.mirror import mirror_to_session
        for sid in other_lanes:
            if mirror_to_session(
                    req.platform, req.chat_id, seed_user_turn(req.user_name, req.synopsis),
                    source_label="group_carryover", role="user", session_id=sid):
                mirrored += 1
    logger.info("Group carry-over: %s:%s forked to %s (%d other lane(s) mirrored)",
                req.platform, req.chat_id, entry.session_id, mirrored)
    return {"success": True, "posted": True, "group_session_id": entry.session_id,
            "other_lanes_mirrored": mirrored}
