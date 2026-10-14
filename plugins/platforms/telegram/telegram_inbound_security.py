"""Telegram ingress security: forwarded provenance and denied-address notifications."""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Any

from gateway.slash_access import _coerce_id_list

logger = logging.getLogger(__name__)

_MAX_PREVIEW_CHARS = 500
_MAX_SENDER_NAME_CHARS = 64
_MAX_SEEN_SENDERS = 2048
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def forwarded_evidence_provenance(message: Any) -> list[dict[str, str]]:
    """Provider-authenticated provenance for modern and legacy Telegram forwards."""
    origin = getattr(message, "forward_origin", None)
    legacy_user = getattr(message, "forward_from", None)
    legacy_chat = getattr(message, "forward_from_chat", None)
    hidden_name = (
        getattr(origin, "sender_user_name", None)
        or getattr(origin, "author_signature", None)
        or getattr(message, "forward_sender_name", None)
    )
    if not any((origin, legacy_user, legacy_chat, hidden_name, getattr(message, "forward_date", None))):
        return []

    record = {"platform": "telegram", "kind": "forwarded_message"}
    origin_object = next(
        (candidate for candidate in (
            getattr(origin, "sender_user", None),
            getattr(origin, "sender_chat", None),
            getattr(origin, "chat", None),
            legacy_user,
            legacy_chat,
        ) if candidate is not None),
        None,
    )
    if origin_object is not None:
        origin_id = getattr(origin_object, "id", None)
        origin_name = (
            getattr(origin_object, "full_name", None)
            or getattr(origin_object, "title", None)
            or getattr(origin_object, "username", None)
            or hidden_name
        )
        if origin_id is not None:
            record["origin_id"] = str(origin_id)
        if origin_name:
            record["origin_name"] = str(origin_name)
    elif hidden_name:
        record["origin_name"] = str(hidden_name)
    return [record]


def _safe_notice_text(value: str, *, max_chars: int) -> str:
    """Bound provider/user labels and remove mention, link, and formatting surfaces."""
    from gateway.session import neutralize_untrusted_inline_text

    safe = neutralize_untrusted_inline_text(value or "", max_chars=max_chars)
    safe = _URL_RE.sub("[link removed]", safe)
    safe = " ".join(safe.splitlines())
    return "".join(ch for ch in safe if ch not in "@<>[]()`*_~#").strip()


def group_access_request(source: Any, preview: str) -> str:
    """Owner-facing copy for an explicitly addressed, denied Telegram group sender."""
    user_name = _safe_notice_text(source.user_name or "Unknown", max_chars=_MAX_SENDER_NAME_CHARS)
    user_id = _safe_notice_text(str(source.user_id or "unknown"), max_chars=128)
    chat_name = _safe_notice_text(source.chat_name or "Unknown chat", max_chars=128)
    chat_id = _safe_notice_text(str(source.chat_id or "unknown"), max_chars=128)
    thread_id = _safe_notice_text(str(source.thread_id or "general"), max_chars=128)
    safe_preview = _safe_notice_text(preview, max_chars=_MAX_PREVIEW_CHARS) or "(no text caption)"
    return (
        "🔐 Telegram access request\n\n"
        f"User: {user_name}\n"
        f"User ID: {user_id}\n"
        f"Chat: {chat_name}\n"
        f"Chat ID: {chat_id}\n"
        f"Topic/thread: {thread_id}\n\n"
        f"Message preview (untrusted): {safe_preview}\n\n"
        "To approve group/topic comments from this person, add their User ID to "
        "platforms.telegram.group_allow_from."
    )


class UnauthorizedGroupAddressNotifier:
    """Notify explicitly configured DM admins once per profile/chat/sender."""

    def __init__(self, max_seen: int = _MAX_SEEN_SENDERS) -> None:
        self._seen: OrderedDict[tuple[str, str, str, str], None] = OrderedDict()
        self._max_seen = max(1, int(max_seen))

    def _first_time(self, adapter: Any, source: Any) -> bool:
        profile = str(source.profile or getattr(adapter, "_owner_profile", None) or "default")
        key = (profile, str(source.chat_id), str(source.thread_id or ""), str(source.user_id))
        if key in self._seen:
            self._seen.move_to_end(key)
            return False
        self._seen[key] = None
        while len(self._seen) > self._max_seen:
            self._seen.popitem(last=False)
        return True

    async def notify(self, adapter: Any, source: Any, preview: str) -> None:
        """Send through the receiving adapter only to explicit DM admins; otherwise stay silent."""
        extra = getattr(getattr(adapter, "config", None), "extra", None)
        extra = extra if isinstance(extra, dict) else {}
        owner_ids = sorted(
            owner for owner in _coerce_id_list(extra.get("allow_admin_from")) if owner != "*"
        )
        if not owner_ids:
            logger.warning(
                "Telegram group access requested by %s but no explicit allow_admin_from owner is configured",
                source.user_id,
            )
            return
        if not self._first_time(adapter, source):
            return
        notice = group_access_request(source, preview)
        for owner_id in owner_ids:
            try:
                await adapter.send(owner_id, notice)
            except Exception:
                logger.warning(
                    "Failed to send Telegram group access request to owner %s", owner_id, exc_info=True,
                )


async def authorize_message_or_report(
    adapter: Any, message: Any, *, level: int = logging.WARNING, what: str = "unauthorized user",
) -> bool:
    """Run the injected admission check before trigger/state logic and report addressed denials."""
    if adapter._is_user_authorized_from_message(message):
        return True
    adapter._log_blocked_user(message, level=level, what=what)
    auth_source = adapter._source_from_message_for_auth(message)
    if auth_source.chat_type not in {"group", "forum"} or not auth_source.user_id:
        return False
    try:
        addressed = (
            adapter._is_reply_to_bot(message)
            or adapter._message_mentions_bot(message)
            or adapter._message_matches_mention_patterns(message)
        )
    except Exception:
        logger.debug("[Telegram] Failed to evaluate denied group address", exc_info=True)
        return False
    if not addressed:
        return False
    source = adapter.build_source(
        chat_id=auth_source.chat_id,
        chat_name=getattr(getattr(message, "chat", None), "title", None),
        chat_type=auth_source.chat_type,
        user_id=auth_source.user_id,
        user_name=auth_source.user_name,
        thread_id=auth_source.thread_id,
        is_bot=auth_source.is_bot,
        message_id=str(getattr(message, "message_id", "") or "") or None,
    )
    notifier = getattr(adapter, "_unauthorized_group_address_notifier", None)
    if notifier is None:
        notifier = adapter._unauthorized_group_address_notifier = UnauthorizedGroupAddressNotifier()
    preview = str(getattr(message, "text", None) or getattr(message, "caption", None) or "")[:_MAX_PREVIEW_CHARS]
    await notifier.notify(adapter, source, preview)
    return False
