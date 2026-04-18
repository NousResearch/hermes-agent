"""Subagent progress → Telegram forum-topic router (Phase A1).

Forwards ``subagent.*`` progress events (relayed from
:func:`tools.delegate_tool._build_child_progress_callback`) into a dedicated
Telegram forum topic per parent session, instead of flooding the main
conversation with child-agent chatter.

Responsibilities
----------------
* Lazy-create a forum topic on the first subagent event of a session.
* Persist ``session_id -> (chat_id, thread_id, ...)`` mappings in
  ``~/.hermes/state/subagent_topics.json`` (atomic writes, thread-safe).
* Forward each subsequent event as a linear-history message (no edits).
* Update ``last_message_ts`` on every successful forward so an external cron
  job can delete topics that have been idle for 24h.
* Soft-fail gracefully: any exception is swallowed with a warning so the
  parent agent never observes a progress-transport error.

Non-goals (Phase A1)
--------------------
* This module does **not** wire itself into :mod:`gateway.run` — that is a
  separate follow-up task.
* This module does **not** run a cron / deletion sweep.
* This module does **not** include unit tests (covered separately).

Design notes
------------
* ``route()`` is the synchronous entry point used by gateway's progress
  callback (which runs in the agent's synchronous worker thread).  It
  schedules ``_async_route`` on the gateway's asyncio loop via
  ``asyncio.run_coroutine_threadsafe`` and returns immediately.
* State I/O uses a module-level :class:`threading.Lock` plus
  ``os.replace`` for atomic rename, so concurrent callers from multiple
  threads cannot corrupt the JSON file.
* A runtime-only ``_forum_blocklist`` short-circuits chats that aren't
  forum supergroups or where the bot lacks ``can_manage_topics`` — after
  the first failed create we never retry for that chat in the current
  process.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Dict, Optional

from hermes_constants import get_hermes_home

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.session import SessionSource


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singletons / locks
# ---------------------------------------------------------------------------

_STATE_LOCK = threading.Lock()
_SINGLETON_LOCK = threading.Lock()
_SINGLETON: Optional["SubagentTopicRouter"] = None

_STATE_VERSION = 1
_DEFAULT_STATE: Dict[str, Any] = {"version": _STATE_VERSION, "topics": {}}

# Telegram hard-caps topic name length at 128, but we stay well under.
_TOPIC_NAME_MAX = 64
_TOPIC_SUMMARY_MAX = 20

# Control characters + markdown chars we strip out of topic names.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_MARKDOWN_STRIP = str.maketrans({c: None for c in "*`#_[]"})

# Simple per-session backpressure: if the 5 most recent events arrived
# within 3s we sleep 0.5s before sending the next one.
_RATE_WINDOW = 5
_RATE_THRESHOLD_SECONDS = 3.0
_RATE_BACKOFF_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------


def get_subagent_topic_router() -> "SubagentTopicRouter":
    """Return the process-wide :class:`SubagentTopicRouter` singleton.

    Thread-safe lazy initialization — the first caller constructs the
    instance; subsequent callers observe the cached one.
    """
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = SubagentTopicRouter()
    return _SINGLETON


# ---------------------------------------------------------------------------
# Core router class
# ---------------------------------------------------------------------------


class SubagentTopicRouter:
    """Routes ``subagent.*`` progress events to per-session Telegram topics."""

    def __init__(
        self,
        state_path: Optional[Path] = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        """Construct a router.

        Args:
            state_path: Override for the JSON state file.  Defaults to
                ``<hermes_home>/state/subagent_topics.json``.
            logger_: Optional logger override (defaults to this module's
                logger).
        """
        if state_path is None:
            state_path = get_hermes_home() / "state" / "subagent_topics.json"
        self.state_path: Path = Path(state_path)
        # Ensure parent directory exists so first save doesn't fail.
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover — disk full / permissions
            (logger_ or logger).warning(
                "Subagent topic router: failed to mkdir %s: %s",
                self.state_path.parent, e,
            )

        self.logger: logging.Logger = logger_ or logger

        # Runtime-only: chat_ids (str) known to reject forum-topic ops.
        self._forum_blocklist: set[str] = set()
        # Per-session recent-event timestamps for simple backpressure.
        self._rate_deques: Dict[str, Deque[float]] = {}
        # Per-session async lock to serialize lazy-create for the same session.
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._session_locks_guard = threading.Lock()

    # ---- sync entry point ------------------------------------------------

    def route(
        self,
        *,
        session_id: str,
        source: "SessionSource",
        event_type: str,
        tool_name: Optional[str] = None,
        preview: Optional[str] = None,
        goal: Optional[str] = None,
        adapter: "BasePlatformAdapter",
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Synchronously schedule a forwarding task on ``loop``.

        Safe to call from any thread.  Returns immediately; delivery
        happens asynchronously on the gateway's event loop.  All failures
        are swallowed — progress transport must never break the parent
        agent.
        """
        try:
            coro = self._async_route(
                session_id=session_id,
                source=source,
                event_type=event_type,
                tool_name=tool_name,
                preview=preview,
                goal=goal,
                adapter=adapter,
            )
            # Fire-and-forget: we intentionally do not await the Future.
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as e:  # pragma: no cover — defensive
            self.logger.warning(
                "Subagent topic route failed (schedule): %s", e, exc_info=True
            )

    # ---- async core ------------------------------------------------------

    async def _async_route(
        self,
        *,
        session_id: str,
        source: "SessionSource",
        event_type: str,
        tool_name: Optional[str],
        preview: Optional[str],
        goal: Optional[str],
        adapter: "BasePlatformAdapter",
    ) -> None:
        """Async body of :meth:`route` — see class docstring for flow."""
        try:
            # Import Platform lazily to avoid any import cycles at module load.
            from gateway.config import Platform

            if getattr(source, "platform", None) != Platform.TELEGRAM:
                return

            parent_chat_id = str(source.chat_id)
            if parent_chat_id in self._forum_blocklist:
                return

            # Serialize concurrent events for the same session so two
            # in-flight ``subagent.start`` callbacks don't both create a topic.
            lock = self._get_session_lock(session_id)
            async with lock:
                state = _load_state(self.state_path)
                topics = state.setdefault("topics", {})
                entry = topics.get(session_id)

                if entry is None:
                    # Lazy-create on first event.
                    thread_id = await self.lazy_create_topic(
                        session_id=session_id,
                        source=source,
                        goal=goal,
                        adapter=adapter,
                    )
                    if thread_id is None:
                        return  # creation failed; blocklisted inside helper
                    # lazy_create_topic persists the mapping via its own
                    # _load_state/_save_state cycle, so we must re-load from
                    # disk — the in-memory `topics` dict captured on L222
                    # does NOT see that update.
                    state = _load_state(self.state_path)
                    entry = state.get("topics", {}).get(session_id)
                    if entry is None:
                        # Persistence failed silently inside lazy_create_topic;
                        # bail out quietly.
                        return

            # Backpressure: throttle if the session is emitting fast.
            await self._apply_backpressure(session_id)

            msg = _format_event(event_type, tool_name, preview)
            if not msg:
                return

            try:
                await adapter.send(
                    chat_id=str(entry["chat_id"]),
                    content=msg,
                    metadata={"thread_id": int(entry["thread_id"])},
                )
            except Exception as e:
                self.logger.warning(
                    "Subagent topic send failed for session %s: %s",
                    session_id, e, exc_info=True,
                )
                return

            # Update last_message_ts on successful send.
            now = time.time()
            with _STATE_LOCK:
                state2 = _load_state(self.state_path)
                topic2 = state2.get("topics", {}).get(session_id)
                if topic2 is not None:
                    topic2["last_message_ts"] = now
                    _save_state(self.state_path, state2)
        except Exception as e:
            self.logger.warning(
                "Subagent topic route failed: %s", e, exc_info=True
            )

    # ---- topic creation --------------------------------------------------

    async def lazy_create_topic(
        self,
        session_id: str,
        source: "SessionSource",
        goal: Optional[str],
        adapter: "BasePlatformAdapter",
    ) -> Optional[str]:
        """Create a Telegram forum topic for ``session_id`` if missing.

        On success:
            * Persists the mapping to the state file.
            * Sends a pointer message to the parent chat/thread.
            * Sends an opening message inside the new topic.
            * Returns the new ``thread_id`` as a string.

        On any failure (no admin rights, chat not a forum, network error,
        etc.) this method logs a warning, adds the parent ``chat_id`` to
        ``_forum_blocklist`` so we stop retrying for the process lifetime,
        and returns ``None``.
        """
        parent_chat_id = str(source.chat_id)
        parent_thread_id = (
            str(source.thread_id) if getattr(source, "thread_id", None) else None
        )
        topic_name = _derive_topic_name(goal, session_id)

        try:
            from tools.telegram_topic_tool import (
                _classify_error,
                _load_telegram_token,
                _run_topic_op,
            )
        except Exception as e:
            self.logger.warning(
                "Subagent topic router: failed to import telegram_topic_tool: %s",
                e, exc_info=True,
            )
            self._forum_blocklist.add(parent_chat_id)
            return None

        token, token_err = _load_telegram_token()
        if token_err or not token:
            self.logger.warning(
                "Subagent topic router: no Telegram token available: %s",
                token_err,
            )
            self._forum_blocklist.add(parent_chat_id)
            return None

        try:
            result = await _run_topic_op(
                token,
                "create",
                chat_id=parent_chat_id,
                thread_id=None,
                name=topic_name,
            )
        except Exception as e:
            code = _classify_error(e)
            err_text = str(e).lower()
            is_not_forum = (
                "not a forum" in err_text
                or "chat_not_modified" in err_text
                or "forum_disabled" in err_text
            )
            if code == "no_rights" or is_not_forum:
                self.logger.warning(
                    "Subagent topic create rejected for chat %s (code=%s): %s — "
                    "blocklisting for session life",
                    parent_chat_id, code, e,
                )
            else:
                self.logger.warning(
                    "Subagent topic create failed for chat %s (code=%s): %s",
                    parent_chat_id, code, e, exc_info=True,
                )
            self._forum_blocklist.add(parent_chat_id)
            return None

        if not isinstance(result, dict) or not result.get("success"):
            self.logger.warning(
                "Subagent topic create returned non-success for chat %s: %r",
                parent_chat_id, result,
            )
            self._forum_blocklist.add(parent_chat_id)
            return None

        thread_id = str(result.get("thread_id") or "")
        if not thread_id:
            self.logger.warning(
                "Subagent topic create missing thread_id for chat %s: %r",
                parent_chat_id, result,
            )
            self._forum_blocklist.add(parent_chat_id)
            return None

        actual_name = str(result.get("name") or topic_name)
        now = time.time()

        # Persist mapping BEFORE sending the pointer/opening messages so
        # even a send failure doesn't make us re-create the topic.
        with _STATE_LOCK:
            state = _load_state(self.state_path)
            topics = state.setdefault("topics", {})
            topics[session_id] = {
                "chat_id": parent_chat_id,
                "thread_id": thread_id,
                "topic_name": actual_name,
                "created_ts": now,
                "last_message_ts": now,
                "parent_chat_id": parent_chat_id,
                "parent_thread_id": parent_thread_id,
            }
            _save_state(self.state_path, state)

        # Best-effort pointer message in parent chat/thread.
        # Telegram hashtags break on whitespace, so instead of "#SA foo bar"
        # we use a t.me deep link that navigates directly to the new topic.
        # For private supergroups (chat_id starts with "-100") the link format
        # is https://t.me/c/<chat_id_without_-100>/<thread_id>.
        try:
            topic_link = _build_topic_link(parent_chat_id, thread_id)
            if topic_link:
                pointer = (
                    f"🔀 Subagent 進度轉發至 → [{actual_name}]({topic_link})\n"
                    "（此 topic 在無新訊息 24h 後自動刪除）"
                )
            else:
                # Public/unknown chat formats — fall back to plain name.
                pointer = (
                    f"🔀 Subagent 進度轉發至 → {actual_name}\n"
                    "（此 topic 在無新訊息 24h 後自動刪除）"
                )
            pointer_meta: Dict[str, Any] = {}
            if parent_thread_id:
                try:
                    pointer_meta["thread_id"] = int(parent_thread_id)
                except (TypeError, ValueError):
                    pass
            await adapter.send(
                chat_id=parent_chat_id,
                content=pointer,
                metadata=pointer_meta or None,
            )
        except Exception as e:
            self.logger.warning(
                "Subagent topic pointer message failed for chat %s: %s",
                parent_chat_id, e,
            )

        # Best-effort opening message inside the new topic.
        try:
            goal_snippet = (goal or "").strip()
            if len(goal_snippet) > 300:
                goal_snippet = goal_snippet[:300].rstrip() + "…"
            opening = (
                f"🔀 Session {session_id[:8]}\n"
                f"Goal: {goal_snippet}\n"
                "— 無新訊息 24h 後此 topic 將被自動刪除"
            )
            await adapter.send(
                chat_id=parent_chat_id,
                content=opening,
                metadata={"thread_id": int(thread_id)},
            )
        except Exception as e:
            self.logger.warning(
                "Subagent topic opening message failed for chat %s thread %s: %s",
                parent_chat_id, thread_id, e,
            )

        return thread_id

    # ---- helpers ---------------------------------------------------------

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Return (and cache) the per-session asyncio lock."""
        with self._session_locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock

    async def _apply_backpressure(self, session_id: str) -> None:
        """Sleep briefly if the recent event rate exceeds the threshold."""
        dq = self._rate_deques.get(session_id)
        if dq is None:
            dq = collections.deque(maxlen=_RATE_WINDOW)
            self._rate_deques[session_id] = dq

        now = time.time()
        dq.append(now)
        if len(dq) >= _RATE_WINDOW:
            span = dq[-1] - dq[0]
            if span < _RATE_THRESHOLD_SECONDS:
                # Let CancelledError propagate cleanly for shutdown; asyncio.sleep
                # has no other exceptions worth swallowing here.
                await asyncio.sleep(_RATE_BACKOFF_SECONDS)


# ---------------------------------------------------------------------------
# Pure helpers: topic name + event formatting
# ---------------------------------------------------------------------------


def _clean_for_topic(text: str) -> str:
    """Strip control chars and common markdown characters."""
    text = _CONTROL_RE.sub(" ", text)
    text = text.translate(_MARKDOWN_STRIP)
    return text.strip()


def _build_topic_link(chat_id: str, thread_id: str) -> Optional[str]:
    """Build a ``t.me/c/...`` deep link to a forum topic, or None if the
    chat_id does not look like a private Telegram supergroup.

    Telegram private supergroups use chat_ids shaped ``-100XXXXXXXXXX``.
    The deep-link form is ``https://t.me/c/<chat_id_without_-100>/<thread_id>``.
    For public chats, DMs, or malformed ids we return ``None`` and the caller
    falls back to a plain-text pointer.
    """
    if not chat_id or not thread_id:
        return None
    cid = str(chat_id).strip()
    if not cid.startswith("-100"):
        return None
    stripped = cid[4:]  # drop the "-100" prefix
    if not stripped.isdigit():
        return None
    try:
        tid = int(str(thread_id).strip())
    except (TypeError, ValueError):
        return None
    if tid <= 0:
        return None
    return f"https://t.me/c/{stripped}/{tid}"


def _summarize(text: str, max_chars: int = _TOPIC_SUMMARY_MAX) -> str:
    """Return the first sentence/line of *text*, cleaned and truncated."""
    if not text:
        return ""
    # Take up to the first sentence-terminator or newline.
    for sep in ("\n", "。", "！", "？", "；", ". ", "! ", "? ", "; "):
        idx = text.find(sep)
        if idx != -1:
            text = text[:idx]
            break
    text = _clean_for_topic(text)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def _derive_topic_name(goal: Optional[str], session_id: str) -> str:
    """Derive a Telegram-safe forum-topic name.

    * If *goal* cleans up to fewer than 4 characters, fall back to
      ``"SA <sid8> · HH:MM"``.
    * Otherwise, ``"SA " + _summarize(goal, 20)``.
    * Always hard-capped to ``_TOPIC_NAME_MAX`` (64) chars.
    """
    cleaned = _clean_for_topic(goal or "")
    if len(cleaned) < 4:
        now_hm = datetime.now().strftime("%H:%M")
        name = f"SA {session_id[:8]} · {now_hm}"
    else:
        name = "SA " + _summarize(goal or "", max_chars=_TOPIC_SUMMARY_MAX)

    if len(name) > _TOPIC_NAME_MAX:
        name = name[: _TOPIC_NAME_MAX - 1].rstrip() + "…"
    return name or f"SA {session_id[:8]}"


def _format_event(
    event_type: str,
    tool_name: Optional[str],
    preview: Optional[str],
) -> str:
    """Render a subagent event as a single-line Telegram message."""
    preview = preview or ""
    if event_type == "subagent.start":
        body = preview or tool_name or ""
        return f"🔀 Started: {body}".rstrip()

    if event_type == "subagent.thinking":
        snippet = preview[:200]
        return f"💭 {snippet}".rstrip()

    if event_type == "subagent.tool":
        try:
            from agent.display import get_tool_emoji
            emoji = get_tool_emoji(tool_name or "", default="⚙️")
        except Exception:
            emoji = "⚙️"
        name = tool_name or ""
        line = f"{emoji} {name}".rstrip()
        if preview:
            snippet = preview[:60]
            line = f'{line}: "{snippet}"'
        return line

    if event_type == "subagent.progress":
        return f"⏳ {preview}".rstrip()

    if event_type == "subagent.complete":
        snippet = preview[:300] if preview else ""
        return f"✅ Done: {snippet}".rstrip()

    # Unknown event — still forward so debugging is easier.
    tail = preview or tool_name or ""
    return f"· {event_type}: {tail}".rstrip()


# ---------------------------------------------------------------------------
# State store: JSON on disk, threading.Lock + atomic replace
# ---------------------------------------------------------------------------


def _load_state(path: Path) -> Dict[str, Any]:
    """Load the router's JSON state, returning the default on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"version": _STATE_VERSION, "topics": {}}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Subagent topic state load failed (%s): %s", path, e)
        return {"version": _STATE_VERSION, "topics": {}}

    if not isinstance(data, dict):
        return {"version": _STATE_VERSION, "topics": {}}
    data.setdefault("version", _STATE_VERSION)
    topics = data.setdefault("topics", {})
    if not isinstance(topics, dict):
        data["topics"] = {}
    return data


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    """Write *state* atomically (tmp + ``os.replace``).

    Uses ``tempfile.mkstemp`` to avoid collisions between concurrent
    writers and guarantees the temp file is cleaned up on any failure
    (including ``KeyboardInterrupt``).  The final file is chmod'd to
    ``0o600`` before the rename so it never appears with wider perms.
    """
    import tempfile

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_str = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            try:
                os.chmod(tmp_path, 0o600)
            except (OSError, NotImplementedError):
                pass
            os.replace(tmp_path, path)
        except BaseException:
            # Ensure we don't leak tmp files on any failure (including
            # KeyboardInterrupt / SystemExit).
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise
    except Exception as e:
        logger.warning("Subagent topic state save failed (%s): %s", path, e)


__all__ = [
    "SubagentTopicRouter",
    "get_subagent_topic_router",
]
