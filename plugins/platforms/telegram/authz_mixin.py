"""User-authorization methods for ``TelegramAdapter``.

Extracted from ``plugins/platforms/telegram/adapter.py`` as part of the
god-file decomposition campaign, following the same mechanical mixin lift that
produced ``gateway/authz_mixin.py``. This mixin holds the Telegram
authorization decision: whether the sender of a message or a callback query is
allowed to drive the agent.

The boundary is deliberate. What lives here answers "is this permitted"; the
mention, guest-mode and free-response helpers that answer "should the bot
reply" stay on the adapter, because they are routing policy rather than a
trust decision. Helpers the adapter shares with methods that did not move
(``_normalize_chat_type``, ``_legacy_runner_auth_fn``,
``_env_allowlist_decision``) also stay on the adapter and resolve through the
MRO, so there is no second copy to drift.

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

import contextlib
import logging
from typing import Any, Optional

from gateway.authz_mixin import _coerce_allow_set
from gateway.config import Platform

try:
    from telegram import Message
except ImportError:  # pragma: no cover - mirrors the adapter's import guard
    Message = Any

# Bind the adapter's logger relative to whichever package this module was
# imported under, so log records lifted with these methods are emitted under
# exactly the name they were before. The name cannot be hard-coded: the plugin
# manager loads directory plugins as ``hermes_plugins.<slug>``, so the
# adapter's own ``getLogger(__name__)`` is ``hermes_plugins.<slug>.adapter``
# there and ``plugins.platforms.telegram.adapter`` under the canonical path.
logger = logging.getLogger(f"{__package__}.adapter")


def _scoped_gate_env(name: str, default: str = "") -> str:
    """Read a TELEGRAM_*/GATEWAY_* authorization gate env var per-profile.

    The reader itself stays in the adapter, which owns the rest of the
    profile-scoping helpers; delegating rather than copying keeps the two from
    drifting apart, which matters because these gates decide who may talk to
    the agent (issue #72348). The import is deferred to call time because the
    adapter imports this module while it is still executing.
    """
    from .adapter import _scoped_gate_env as _adapter_scoped_gate_env

    return _adapter_scoped_gate_env(name, default)


class TelegramAuthorizationMixin:
    """Authorization cluster lifted verbatim from ``TelegramAdapter``."""

    def _is_callback_user_authorized(
        self, user_id: str, *, chat_id: Optional[str] = None, chat_type: Optional[str] = None,
        thread_id: Optional[str] = None, user_name: Optional[str] = None) -> bool:
        """Return whether a Telegram inline-button caller may perform gated actions."""
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return False
        normalized_chat_type = self._normalize_chat_type(chat_type, is_forum=thread_id is not None)
        # Preferred: the auth callback GatewayRunner injects (set_authorization_check) → full
        # _is_user_authorized chain; also works for a multiplexed adapter whose _message_handler is a
        # profile closure. getattr tolerates partially-constructed adapters (object.__new__ in tests).
        if getattr(self, "_authorization_check", None) is not None:
            injected = self._is_sender_authorized(
                normalized_user_id, chat_type=normalized_chat_type, chat_id=str(chat_id or normalized_user_id),
                thread_id=str(thread_id) if thread_id is not None else None)
            if injected is not None:
                return injected
        auth_fn = self._legacy_runner_auth_fn()
        if auth_fn is not None:
            try:
                from gateway.session import SessionSource
                source = SessionSource(
                    platform=Platform.TELEGRAM, chat_id=str(chat_id or normalized_user_id), chat_type=normalized_chat_type,
                    user_id=normalized_user_id, user_name=str(user_name).strip() if user_name else None,
                    thread_id=str(thread_id) if thread_id is not None else None)
                return bool(auth_fn(source))
            except Exception:
                logger.debug(
                    "[Telegram] Falling back to env-only callback auth for user %s", normalized_user_id, exc_info=True)
        decision = self._env_allowlist_decision(normalized_user_id)
        if decision is None:
            # Fail-closed: no allowlist means deny unless GATEWAY_ALLOW_ALL_USERS is set.
            # The runner auth path in _is_user_authorized() handles GATEWAY_ALLOW_ALL_USERS; this fallback
            # must not silently allow everyone (fixes #24457).
            return _scoped_gate_env("GATEWAY_ALLOW_ALL_USERS").lower() in {"true", "1", "yes"}
        return decision

    def _source_from_message_for_auth(self, message: Message):
        """Build the SessionSource the gateway auth path expects; identity comes from ``from_user``,
        falling back to ``sender_chat`` for channel posts so an unauthorized channel can't inject."""
        from gateway.session import SessionSource
        user = getattr(message, "from_user", None)
        chat = getattr(message, "chat", None)
        user_id = str(getattr(user, "id", "")).strip() or None
        # Carry is_bot so the runner's ``*_ALLOW_BOTS`` branch is reachable, as in build_source.
        is_bot = bool(getattr(user, "is_bot", False)) if user is not None else False
        user_name = str(getattr(user, "username", "") or getattr(user, "full_name", "") or "").strip() or None
        if not user_id:  # channel post — authorize the sender chat instead
            sender_chat = getattr(message, "sender_chat", None)
            if sender_chat is not None:
                user_id = str(getattr(sender_chat, "id", "")).strip() or None
                if not user_name:
                    user_name = str(getattr(sender_chat, "title", "") or "").strip() or None
        chat_id = str(getattr(chat, "id", "")).strip() or user_id
        thread_id_raw = getattr(message, "message_thread_id", None)
        is_topic_message = bool(getattr(message, "is_topic_message", False))
        is_forum_group = getattr(chat, "is_forum", False) is True
        chat_type = self._normalize_chat_type(
            getattr(chat, "type", "dm"), is_forum=thread_id_raw is not None and (is_topic_message or is_forum_group))
        thread_id = None
        if thread_id_raw is not None and (
            (chat_type == "forum" and (is_topic_message or is_forum_group)) or (chat_type == "dm" and is_topic_message)):
            thread_id = str(thread_id_raw)
        return SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id or "", chat_type=chat_type, user_id=user_id,
            user_name=user_name, thread_id=thread_id, is_bot=is_bot)

    def _telegram_auth_env_configured(self) -> bool:
        """Return True when Telegram auth env vars make an early decision safe."""
        keys = (
            "TELEGRAM_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_CHATS",
            "TELEGRAM_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS")
        return any(_scoped_gate_env(key).strip() for key in keys)

    def _is_user_authorized_from_message(self, message: Message) -> bool:
        """Intake auth prefilter, run BEFORE batching/event construction/group observation.

        Only rejects when it can make the same context-aware decision the runner would; unknown DMs pass through when
        there is no allowlist or pairing is the unauthorized-DM behavior."""
        source = self._source_from_message_for_auth(message)
        user_id = source.user_id
        # No identity → service message or channel post without sender_chat; defer to message gating.
        if not user_id:
            return True
        authorized: Optional[bool] = None
        # Adapter-level allow_from (DMs) / group_allow_from (groups) are the sole authority if set.
        adapter_allow_from = self.config.extra.get(
            "group_allow_from" if (source.chat_type or "") in ("group", "forum", "channel") else "allow_from")
        if adapter_allow_from is not None:
            allowed = _coerce_allow_set(adapter_allow_from)
            authorized = user_id in allowed or "*" in allowed
        # Instance-level override only (tests): the class method _is_callback_user_authorized is for
        # inline buttons and must not become a user-id-only shortcut for real messages.
        if authorized is None:
            callback_auth = self.__dict__.get("_is_callback_user_authorized")
            if callable(callback_auth):
                with contextlib.suppress(Exception):
                    authorized = bool(callback_auth(
                        user_id, chat_id=source.chat_id, chat_type=source.chat_type, thread_id=source.thread_id,
                        user_name=source.user_name))
        if authorized is None:
            # Runner's full auth chain; prefer the set_authorization_check callback (survives multiplex
            # handler wrapping, unlike bound-handler __self__).
            auth_fn = self._legacy_runner_auth_fn()
            has_callback = getattr(self, "_authorization_check", None) is not None
            if has_callback or auth_fn is not None:
                # No allowlist → unknown DMs must reach pairing, not be default-denied here.
                if not self._telegram_auth_env_configured():
                    return True
                decision = self._is_sender_authorized(
                    user_id, chat_type=source.chat_type, chat_id=source.chat_id, is_bot=source.is_bot,
                    thread_id=source.thread_id) if has_callback else None
                if decision is not None:
                    authorized = decision
                elif auth_fn is not None:
                    try:
                        authorized = bool(auth_fn(source))
                    except Exception:
                        logger.debug("[Telegram] Falling back to env-only auth for user %s", user_id, exc_info=True)
        if authorized is None:
            authorized = self._env_allowlist_decision(user_id)
            if authorized is None:
                return True
        if authorized:
            return True
        # Unauthorized DM the gateway would pair: forward so pairing can run.
        return self._should_pass_unauthorized_dm_for_pairing(source)
