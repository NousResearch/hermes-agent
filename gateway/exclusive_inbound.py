"""Fail-closed exclusive admission for one configured inbound chat."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Optional

from gateway.platforms.base import BasePlatformAdapter, MessageEvent
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def _claim(adapter: BasePlatformAdapter) -> Optional[dict[str, Any]]:
    """Return the one exact-chat claim configured for *adapter*, or validate its absence."""
    raw = (getattr(getattr(adapter, "config", None), "extra", None) or {}).get("exclusive_inbound")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("platform extra.exclusive_inbound must be a mapping")
    chat_id, handler = raw.get("chat_id"), raw.get("handler")
    if not isinstance(chat_id, str) or not chat_id.strip():
        raise ValueError("exclusive_inbound.chat_id must be a non-empty string")
    if not isinstance(handler, str) or not handler.strip():
        raise ValueError("exclusive_inbound.handler must be a non-empty string")
    allowed_senders = raw.get("allowed_senders")
    if allowed_senders is not None and (
        not isinstance(allowed_senders, list)
        or not allowed_senders
        or any(not isinstance(sender, str) or not sender.strip() for sender in allowed_senders)
    ):
        raise ValueError("exclusive_inbound.allowed_senders must be a non-empty list of strings")
    return {
        "chat_id": chat_id.strip(),
        "handler": handler.strip(),
        "allowed_senders": ({sender.strip().lower() for sender in allowed_senders}
                            if allowed_senders is not None else None),
    }


def configure_exclusive_inbound(runner: Any, adapter: BasePlatformAdapter, *, profile_name: str | None = None) -> None:
    """Install the profile-scoped exact-chat boundary before adapter ingress can queue a turn."""
    claim = _claim(adapter)
    if claim is None:
        adapter.set_exclusive_inbound_handler(None)
        return

    authorization_home = Path(get_hermes_home())
    if profile_name is not None:
        from hermes_cli.profiles import get_profile_dir
        authorization_home = Path(get_profile_dir(profile_name))

    async def admit(event: MessageEvent) -> bool:
        source = event.source
        if str(getattr(source, "chat_id", "")) != claim["chat_id"]:
            return False

        # A match is terminal even if authorization, discovery, or the claimant fails.
        source._authorization_profile_home = authorization_home
        if profile_name and not getattr(source, "profile", None):
            source.profile = profile_name
        try:
            from gateway.run import _async_profile_runtime_scope
            from hermes_cli.plugins import get_plugin_manager

            async with _async_profile_runtime_scope(authorization_home):
                sender_allowed = claim["allowed_senders"] is None or (
                    str(getattr(source, "user_id", "")).strip().lower() in claim["allowed_senders"]
                )
                if not sender_allowed or not runner._is_user_authorized_for_source(source):
                    logger.warning("Exclusive inbound sender refused on %s/%s", adapter.platform.value, source.chat_id)
                    return True
                callbacks = get_plugin_manager().iter_exclusive_inbound_handlers(claim["handler"])
                if len(callbacks) != 1:
                    logger.error("Exclusive inbound handler %s has %d registrations", claim["handler"], len(callbacks))
                    return True
                accepted = callbacks[0](event)
                if not inspect.isawaitable(accepted):
                    raise TypeError("exclusive inbound handler must return an awaitable")
                if await accepted is not True:
                    raise RuntimeError("exclusive inbound handler did not durably accept message")
        except Exception:
            logger.exception("Exclusive inbound handler %s failed", claim["handler"])
        return True

    adapter.set_exclusive_inbound_handler(admit)
