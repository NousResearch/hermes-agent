"""Session mirroring for cross-platform message delivery.

When a message is sent to a platform (send_message or cron delivery), append a
"delivery-mirror" record to the target session's transcript so the receiving-side
agent knows what was sent.  Standalone: works from CLI, cron and gateway contexts.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"
_SESSIONS_INDEX = _SESSIONS_DIR / "sessions.json"
_SESSIONS_INDEX_AT_IMPORT = _SESSIONS_INDEX


def _resolve_sessions_index() -> Path:
    """Active profile's ``sessions.json`` at call time: the patched ``_SESSIONS_INDEX`` when a test
    changed it, else live profile-scoped HERMES_HOME — under the multiplexed gateway one process
    serves every profile, so the import-time constant would resolve every profile's pre-migration
    session lookup against the launch profile's index."""
    return (_SESSIONS_INDEX if _SESSIONS_INDEX != _SESSIONS_INDEX_AT_IMPORT
            else get_hermes_home() / "sessions" / "sessions.json")


def _origin_user_id(entry: dict) -> str:
    return str((entry.get("origin") or {}).get("user_id") or "")


def mirror_to_session(
    platform: str, chat_id: str, message_text: str, source_label: str = "cli", thread_id: Optional[str] = None,
    user_id: Optional[str] = None, role: str = "assistant", session_id: Optional[str] = None,
) -> bool:
    """Append a delivery-mirror message to the target session's SQLite transcript.

    Pass ``session_id`` when the caller already holds the exact session (e.g. the
    cron in_channel seed) to skip the origin scan, which drops the mirror when a bare
    target's room can't be told apart (no General/thread-1 and no flat session).
    Text that is NOT the agent speaking (e.g. a cron brief) must pass
    ``role="user"``: ``mirror`` metadata is dropped at the SQLite boundary, so an
    assistant-role mirror replays as a real turn and yields assistant→assistant
    pairs that break strict-alternation providers, while a user-role mirror
    collapses safely via the consecutive-user merge.
    Returns True if mirrored, False if no matching session or error. Never raises.

    ``role`` defaults to ``"assistant"`` — correct for the interactive ``send_message`` mirror, where the
    mirrored text is the agent's own outgoing reply (a genuine assistant turn). See #2221.
    """
    try:
        if not session_id:
            session_id = _find_session_id(platform, str(chat_id), thread_id=thread_id, user_id=user_id)
        if not session_id:
            logger.warning(
                "Mirror: no session found for %s:%s thread=%s user=%s (explicit_id=none, origin-scan bailed)",
                platform, chat_id, thread_id, user_id,
            )
            return False
        _append_to_sqlite(session_id, {
            "role": role, "content": message_text, "timestamp": datetime.now().isoformat(),
            "mirror": True, "mirror_source": source_label,
        })
        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True
    except Exception as e:
        # WARNING, not debug: a silent mirror drop is the cron continuation-amnesia bug.
        logger.warning("Mirror failed for %s:%s thread=%s user=%s session=%s: %s", platform, chat_id, thread_id, user_id, session_id, e)
        return False


def _find_session_id(platform: str, chat_id: str, thread_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[str]:
    """Active session_id for a platform + chat_id pair.

    state.db is primary; sessions.json is the pre-migration fallback.  DM keys
    don't embed the chat_id ("agent:main:telegram:dm"), so match on the persisted
    origin.  With *user_id*, exact sender matches win; several same-chat candidates
    with no user match → None rather than contaminate another participant's session.

    A bare (thread-less) target names the chat itself, never "any room in it": probe the
    General room (thread "1"), the flat session, then a thread keyed on its own id (Discord
    thread channels store ``chat_id == thread_id``); none → None. An unfiltered scan returned
    the newest-started room of a threaded chat instead. Every probe runs against state.db
    before the sessions.json fallback, so a stale JSON entry never beats a live row.
    """
    probes = (str(thread_id),) if thread_id is not None else ("1", "", str(chat_id))
    try:
        from persistence_assembly import open_persistence, profile_scope
        with open_persistence(profile_scope()) as store:
            for probe in probes:
                session_id = store.routing.session_for_origin(platform=platform, chat_id=chat_id, thread_id=probe, user_id=user_id)
                if session_id:
                    return str(session_id)
    except Exception as e:
        logger.debug("Mirror state.db session lookup failed: %s", e)

    sessions_index = _resolve_sessions_index()
    if not sessions_index.exists():
        return None
    try:
        data = json.loads(sessions_index.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    for probe in probes:
        session_id = _find_in_sessions_index(data, platform, chat_id, probe, user_id)
        if session_id:
            return session_id
    return None


def _find_in_sessions_index(data: dict, platform: str, chat_id: str, thread_id: str,
                            user_id: Optional[str]) -> Optional[str]:
    """One exact-thread lookup in the pre-migration ``sessions.json`` index."""
    def _matches(entry: dict) -> bool:
        origin = entry.get("origin") or {}
        return ((origin.get("platform") or entry.get("platform", "")).lower() == platform.lower()
                and str(origin.get("chat_id", "")) == str(chat_id)
                and str(origin.get("thread_id") or "") == thread_id)

    # Keys starting with "_" (e.g. the gateway's "_README") are metadata sentinels.
    candidates = [e for k, e in data.items() if not str(k).startswith("_") and isinstance(e, dict) and _matches(e)]
    if not candidates:
        return None
    if user_id:
        exact_user_matches = [e for e in candidates if _origin_user_id(e) == str(user_id)]
        if exact_user_matches:
            candidates = exact_user_matches
        elif len(candidates) > 1:
            return None
    elif len(candidates) > 1 and len({u.strip() for u in map(_origin_user_id, candidates) if u.strip()}) > 1:
        return None
    return max(candidates, key=lambda entry: entry.get("updated_at", "")).get("session_id")


def _append_to_sqlite(session_id: str, message: dict) -> None:
    """Append a message to the SQLite session database.

    Raises on failure: ``mirror_to_session`` reports ``False`` (and warns) only when the
    exception reaches it — swallowing it here made every failed write look mirrored (#10130).
    """
    from persistence.conversation.messages import NewMessage
    from persistence_assembly import open_persistence, profile_scope

    with open_persistence(profile_scope()) as store:
        before = _stored_message_count(store, session_id)
        store.messages.append(session_id, NewMessage(role=message.get("role", "assistant"), content=message.get("content")))
        try:
            _rebaseline_cached_agent(session_id, before, _stored_message_count(store, session_id))
        except Exception as e:  # the row is written; a cache miss only costs one rebuild
            logger.debug("Mirror: cached agent message_count not re-baselined for %s: %s", session_id, e)


def _stored_message_count(store, session_id: str) -> Optional[int]:
    """The session's stored ``message_count`` (what the gateway's agent-cache coherence guard compares), or None."""
    try:
        row = store.sessions.get(session_id)
        return None if row is None else row.message_count
    except Exception:
        return None


def _rebaseline_cached_agent(session_id: str, before: Optional[int], after: Optional[int]) -> None:
    """Move an in-process cached agent's ``message_count`` snapshot past the row this mirror just wrote.

    The gateway rebuilds a chat's cached agent when the stored count differs from its snapshot ("possible
    cross-process write"), so an in-process mirror (cron delivery, send_message into another chat, a webhook
    delivery) used to cost a full agent rebuild and a cold prompt cache on the chat's next message. Same idea as
    ``GatewayRunner._refresh_agent_cache_message_count`` after a turn's own writes and after a completion note,
    but conservative: only a live entry snapshotted for this exact session whose count still equals the count
    just before the append moves; any other drift (a real cross-process write, a turn mid-way, a pending build)
    is left for the guard. No gateway runner in this process (CLI, cron-only, tests): nothing to do.
    """
    if before is None or after is None or after == before:
        return
    run = sys.modules.get("gateway.run")  # never import the gateway just to look
    runner = run._gateway_runner_ref() if run is not None else None
    lock, cache = getattr(runner, "_agent_cache_lock", None), getattr(runner, "_agent_cache", None)
    if runner is None or not lock or cache is None:
        return
    with lock:
        for key, cached in list(cache.items()):
            if (isinstance(cached, tuple) and len(cached) > 3 and cached[0] is not run._AGENT_PENDING_SENTINEL
                    and cached[3] == session_id and cached[2] == before):
                cache[key] = (cached[0], cached[1], after) + tuple(cached[3:])
