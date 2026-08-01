"""User-authorization methods for ``TelegramAdapter``.

Extracted from ``plugins/platforms/telegram/adapter.py`` as part of the
god-file decomposition campaign, following the same mechanical mixin lift that
produced ``gateway/authz_mixin.py``. This mixin holds the Telegram
authorization cluster: whether the sender of a message or a callback query is
allowed to drive the agent, and which chats, topics and threads are in scope.

The boundary is deliberate. What lives here answers "is this permitted"; the
mention, guest-mode and free-response helpers that answer "should the bot
reply" stay on the adapter, because they are routing policy rather than a
trust decision.

Behavior-neutral: every method is lifted verbatim from ``TelegramAdapter``.
``self.*`` calls resolve unchanged via the MRO, and
``TelegramAuthorizationMixin`` precedes ``BasePlatformAdapter`` in the bases so
resolution order is what it was when these methods sat on the class.

Two details keep the lift observationally identical:

* ``logger`` is bound by explicit name rather than ``__name__``, so records
  emitted from these methods keep the logger name
  ``"plugins.platforms.telegram.adapter"``. ``getLogger`` returns the same
  singleton object the adapter module holds.
* ``Message`` is imported under the same ``ImportError`` guard the adapter
  uses, falling back to ``Any``. This module deliberately does not enable
  postponed annotation evaluation, matching the adapter, so the annotations on
  the lifted signatures are evaluated exactly as before.
"""

import logging
import os
from typing import Any, Optional

from gateway.authz_mixin import _coerce_allow_set
from gateway.config import Platform

try:
    from telegram import Message
except ImportError:  # pragma: no cover - mirrors the adapter's import guard
    Message = Any

# Bind the adapter's logger by name so log records lifted with these methods
# are emitted under exactly the name they were before.
logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramAuthorizationMixin:
    """Authorization cluster lifted verbatim from ``TelegramAdapter``."""

    def _is_callback_user_authorized(
        self,
        user_id: str,
        *,
        chat_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> bool:
        """Return whether a Telegram inline-button caller may perform gated actions."""
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return False

        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        auth_fn = getattr(runner, "_is_user_authorized", None)
        if callable(auth_fn):
            try:
                from gateway.session import SessionSource

                normalized_chat_type = str(chat_type or "dm").strip().lower() or "dm"
                if normalized_chat_type == "private":
                    normalized_chat_type = "dm"
                elif normalized_chat_type == "supergroup":
                    normalized_chat_type = "forum" if thread_id is not None else "group"

                source = SessionSource(
                    platform=Platform.TELEGRAM,
                    chat_id=str(chat_id or normalized_user_id),
                    chat_type=normalized_chat_type,
                    user_id=normalized_user_id,
                    user_name=str(user_name).strip() if user_name else None,
                    thread_id=str(thread_id) if thread_id is not None else None,
                )
                return bool(auth_fn(source))
            except Exception:
                logger.debug(
                    "[Telegram] Falling back to env-only callback auth for user %s",
                    normalized_user_id,
                    exc_info=True,
                )

        allowed_csv = os.getenv("TELEGRAM_ALLOWED_USERS", "").strip()
        if not allowed_csv:
            # Fail-closed: no allowlist means deny by default.
            # The runner auth path in _is_user_authorized() handles
            # GATEWAY_ALLOW_ALL_USERS; this fallback must not silently
            # allow everyone (fixes #24457).
            return os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}
        allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
        return "*" in allowed_ids or normalized_user_id in allowed_ids

    def _source_from_message_for_auth(self, message: Message):
        """Build the same Telegram source shape the gateway auth path expects.

        Resolves the identity to authorize from ``from_user`` for normal
        messages, falling back to ``sender_chat`` for channel posts (which
        carry no ``from_user``) so a removed/unauthorized channel cannot
        inject content via the broadcast path either.
        """
        from gateway.session import SessionSource

        user = getattr(message, "from_user", None)
        chat = getattr(message, "chat", None)
        user_id = str(getattr(user, "id", "")).strip() or None
        user_name = (
            str(getattr(user, "username", "") or getattr(user, "full_name", "") or "").strip()
            or None
        )
        # Channel posts have no from_user — authorize the sender chat instead.
        if not user_id:
            sender_chat = getattr(message, "sender_chat", None)
            if sender_chat is not None:
                user_id = str(getattr(sender_chat, "id", "")).strip() or None
                if not user_name:
                    user_name = (
                        str(getattr(sender_chat, "title", "") or "").strip() or None
                    )

        chat_id = str(getattr(chat, "id", "")).strip() or user_id
        chat_type = str(getattr(chat, "type", "dm")).strip().lower() or "dm"
        if chat_type == "private":
            chat_type = "dm"
        elif chat_type == "supergroup":
            thread_id_raw = getattr(message, "message_thread_id", None)
            is_topic_message = bool(getattr(message, "is_topic_message", False))
            is_forum_group = getattr(chat, "is_forum", False) is True
            chat_type = (
                "forum"
                if thread_id_raw is not None and (is_topic_message or is_forum_group)
                else "group"
            )

        thread_id = None
        thread_id_raw = getattr(message, "message_thread_id", None)
        if thread_id_raw is not None:
            is_topic_message = bool(getattr(message, "is_topic_message", False))
            is_forum_group = getattr(chat, "is_forum", False) is True
            if chat_type == "forum" and (is_topic_message or is_forum_group):
                thread_id = str(thread_id_raw)
            elif chat_type == "dm" and is_topic_message:
                thread_id = str(thread_id_raw)

        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id or "",
            chat_type=chat_type,
            user_id=user_id,
            user_name=user_name,
            thread_id=thread_id,
        )

    def _telegram_auth_env_configured(self) -> bool:
        """Return True when Telegram auth env vars make an early decision safe."""
        keys = (
            "TELEGRAM_ALLOWED_USERS",
            "TELEGRAM_GROUP_ALLOWED_USERS",
            "TELEGRAM_GROUP_ALLOWED_CHATS",
            "TELEGRAM_ALLOW_ALL_USERS",
            "GATEWAY_ALLOWED_USERS",
            "GATEWAY_ALLOW_ALL_USERS",
        )
        return any(os.getenv(key, "").strip() for key in keys)

    def _is_user_authorized_from_message(self, message: Message) -> bool:
        """Check if the sender of a Telegram message is authorized.

        Intake prefilter that runs BEFORE text batching, event construction,
        and unmentioned-group observation, so a removed/unauthorized user
        cannot inject prompt content into the agent path or the observed
        transcript (fixes #40863). It only rejects when it can make the same
        context-aware decision the runner would make. Unknown DMs with no
        allowlist still pass through so the normal pairing flow can run.
        """
        source = self._source_from_message_for_auth(message)
        user_id = source.user_id
        # No identity at all → genuine group service message (pin, delete,
        # new_chat_members, etc.). Defer to the cold path. Channel posts
        # without sender_chat already resolved to None above and fall here;
        # they carry no authorizable identity, so let the normal
        # _should_process_message gating handle them.
        if not user_id:
            return True

        # Adapter-level allow_from / group_allow_from: when set, they are the
        # sole authority.  Group chats use group_allow_from; DMs use allow_from.
        chat_type = source.chat_type or ""
        if chat_type in ("group", "forum", "channel"):
            adapter_allow_from = self.config.extra.get("group_allow_from")
        else:
            adapter_allow_from = self.config.extra.get("allow_from")
        if adapter_allow_from is not None:
            allowed = _coerce_allow_set(adapter_allow_from)
            return user_id in allowed or "*" in allowed

        # Test/custom injection only. The class method named
        # _is_callback_user_authorized is for inline button callbacks and must
        # not be treated as a user-id-only shortcut for real messages — only
        # honor an instance-level override (set in tests).
        callback_auth = self.__dict__.get("_is_callback_user_authorized")
        if callable(callback_auth):
            try:
                return bool(
                    callback_auth(
                        user_id,
                        chat_id=source.chat_id,
                        chat_type=source.chat_type,
                        thread_id=source.thread_id,
                        user_name=source.user_name,
                    )
                )
            except Exception:
                pass

        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        auth_fn = getattr(runner, "_is_user_authorized", None)
        if callable(auth_fn):
            # Only make an early decision via the runner when an allowlist
            # actually exists; otherwise unknown DMs must reach the pairing
            # flow rather than being default-denied here.
            if not self._telegram_auth_env_configured():
                return True
            try:
                return bool(auth_fn(source))
            except Exception:
                logger.debug(
                    "[Telegram] Falling back to env-only auth for user %s",
                    user_id,
                    exc_info=True,
                )

        allowed_csv = os.getenv("TELEGRAM_ALLOWED_USERS", "").strip()
        if not allowed_csv:
            return True
        allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
        return "*" in allowed_ids or user_id in allowed_ids

    def _telegram_allowed_chats(self) -> set[str]:
        """Return the whitelist of group/supergroup chat IDs the bot will respond in.

        When non-empty, group messages from chats NOT in this set are
        silently ignored unless ``guest_mode`` is enabled and the bot is
        explicitly @mentioned.  DMs are never filtered.
        Empty set means no restriction (fully backward compatible).
        """
        raw = self.config.extra.get("allowed_chats")
        if raw is None:
            raw = os.getenv("TELEGRAM_ALLOWED_CHATS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _telegram_group_allowed_chats(self) -> set[str]:
        """Return Telegram chats authorized at group scope."""
        raw = self.config.extra.get("group_allowed_chats")
        if raw is None:
            raw = os.getenv("TELEGRAM_GROUP_ALLOWED_CHATS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _telegram_observe_allowed_chats(self) -> set[str]:
        """Chats where observed group context may use a shared source.

        ``group_allowed_chats`` is the gateway authorization allowlist for
        user-less group sources.  ``allowed_chats`` remains an optional response
        gate; when set, observed context must satisfy both lists.
        """
        group_allowed = self._telegram_group_allowed_chats()
        if not group_allowed:
            return set()
        response_allowed = self._telegram_allowed_chats()
        if response_allowed:
            return group_allowed & response_allowed
        return group_allowed

    def _telegram_allowed_topics(self) -> set[str]:
        """Return the whitelist of Telegram forum topic IDs this bot handles.

        When non-empty, group/supergroup messages from other topics are
        silently ignored. DMs are never filtered by topic. Telegram may omit
        ``message_thread_id`` for the forum General topic, so ``None`` is
        treated as topic ``1`` for matching purposes.
        """
        raw = self.config.extra.get("allowed_topics")
        if raw is None:
            raw = os.getenv("TELEGRAM_ALLOWED_TOPICS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _telegram_ignored_threads(self) -> set[int]:
        raw = self.config.extra.get("ignored_threads")
        if raw is None:
            raw = os.getenv("TELEGRAM_IGNORED_THREADS", "")

        if isinstance(raw, list):
            values = raw
        else:
            values = str(raw).split(",")

        ignored: set[int] = set()
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            try:
                ignored.add(int(text))
            except (TypeError, ValueError):
                logger.warning("[%s] Ignoring invalid Telegram thread id: %r", self.name, value)
        return ignored
