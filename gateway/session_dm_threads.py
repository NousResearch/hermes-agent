"""Telegram ``platforms.telegram.extra.dm_threads``: whether a private chat's threads key sessions.

With BotFather Threaded Mode on, Telegram stamps a ``message_thread_id`` on DM messages (a reply
into a thread, a client that opens a fresh thread per question), and ``build_session_key`` folds it
into the DM key, so one chat can run several sessions at once: each with its own busy guard, so a
re-sent question starts N parallel turns and N simultaneous final sends (#107133).

``per_thread`` (default) keeps that historical isolation. ``shared`` keys the whole DM as one
session; replies still go back to the thread each message came from. ``/topic`` multi-session mode
and operator-declared ``extra.dm_topics`` threads stay isolated under ``shared``: there a thread IS
a session by the user's or operator's own choice.

Every session-key builder passes its source through :func:`dm_thread_key_source` first, so the
adapter's batch/busy key, the store's routing key and the handoff key agree.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Mapping, Optional

from gateway.config import Platform

logger = logging.getLogger(__name__)

DM_THREAD_MODES = ("per_thread", "shared")
_warned_modes: set[str] = set()


def _declared_topic_thread(extra: Mapping[str, Any], chat_id: str, thread_id: str) -> bool:
    for entry in extra.get("dm_topics") or ():
        if not isinstance(entry, dict) or str(entry.get("chat_id")) != chat_id:
            continue
        for topic in entry.get("topics") or ():
            if isinstance(topic, dict) and str(topic.get("thread_id") or "") == thread_id:
                return True
    return False


def _topic_mode_enabled(db: Any, source: Any) -> bool:
    try:
        db = db() if callable(db) else db
    except Exception:
        logger.debug("dm_threads: no session DB; keeping thread isolation", exc_info=True)
        return True
    db = getattr(db, "_db", db)  # AsyncSessionDB facade -> the sync store
    check = getattr(db, "is_telegram_topic_mode_enabled", None)
    if not callable(check):
        return False
    try:
        return check(chat_id=str(source.chat_id), user_id=str(source.user_id),
                     profile_name=str(getattr(source, "profile", None) or "").strip() or "default") is True
    except Exception:
        # Unreadable topic state: keep the thread in the key (the historical, isolated shape).
        logger.debug("dm_threads: topic-mode read failed; keeping thread isolation", exc_info=True)
        return True


def dm_thread_key_source(source: Any, extra: Optional[Mapping[str, Any]], db: Any = None) -> Any:
    """*source* with its DM ``thread_id`` dropped when ``extra.dm_threads`` is ``shared`` and the
    thread is not a ``/topic`` lane or a declared ``dm_topics`` topic; else *source* itself. *db* is
    the SessionDB holding topic-mode state, or a zero-arg callable returning it (read only when a
    shared-mode DM thread needs the check)."""
    if (getattr(source, "platform", None) != Platform.TELEGRAM or getattr(source, "chat_type", None) != "dm"
            or not getattr(source, "thread_id", None) or getattr(source, "detached_route", None)):
        return source
    mode = str((extra or {}).get("dm_threads") or "per_thread").strip().lower()
    if mode != "shared":
        if mode not in DM_THREAD_MODES and mode not in _warned_modes:
            _warned_modes.add(mode)
            logger.warning("platforms.telegram.extra.dm_threads=%r is not one of %s; keeping per_thread",
                           mode, ", ".join(DM_THREAD_MODES))
        return source
    if _declared_topic_thread(extra or {}, str(source.chat_id), str(source.thread_id)):
        return source
    if _topic_mode_enabled(db, source):
        return source
    return dataclasses.replace(source, thread_id=None)


def telegram_extra_for(source: Any, config: Any) -> Mapping[str, Any]:
    """``platforms.telegram.extra`` that governs *source*: the receiving adapter's own (a multiplexed
    secondary bot has its own config), else the gateway config's."""
    ref = getattr(source, "_transport_adapter_ref", None)
    adapter = ref() if callable(ref) else None
    extra = getattr(getattr(adapter, "config", None), "extra", None)
    if isinstance(extra, Mapping):
        return extra
    platforms = getattr(config, "platforms", None) or {}
    pconfig = platforms.get(Platform.TELEGRAM) if isinstance(platforms, Mapping) else None
    extra = getattr(pconfig, "extra", None)
    return extra if isinstance(extra, Mapping) else {}
