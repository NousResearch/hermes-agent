"""Routing for operator-facing gateway notices.

These notices describe gateway state rather than the conversation.  Operators may keep the
historical chat delivery, send them to configured platform admins, or retain them only in logs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from gateway.slash_access import policy_from_extra

logger = logging.getLogger("gateway.run")

_NOTICE_MODES = frozenset({"chat", "admin_dm", "log"})


def resolve_operator_notice_mode(user_config: Any, kind: str) -> str:
    """Return the configured route for ``kind``; absent or invalid values preserve chat delivery."""
    config = user_config if isinstance(user_config, dict) else {}
    gateway = config.get("gateway") if isinstance(config.get("gateway"), dict) else {}
    notices = gateway.get("notices") if isinstance(gateway.get("notices"), dict) else {}
    raw = notices.get(kind)
    mode = str(raw or "").strip().lower()
    return mode if mode in _NOTICE_MODES else "chat"


def _adapter_admin_ids(adapter: Any) -> tuple[str, ...]:
    """Read DM-admin identities from the routed adapter's own profile config."""
    adapter_config = getattr(adapter, "config", None)
    extra = getattr(adapter_config, "extra", None)
    if not isinstance(extra, dict):
        extra = {}
    return tuple(sorted(policy_from_extra(extra, "dm").admin_user_ids))


def _direct_metadata(source: Any) -> dict[str, Any]:
    """Keep transport/profile routing metadata while deliberately dropping chat/thread routing."""
    metadata: dict[str, Any] = {"_interim_send": True}
    platform = str(getattr(getattr(source, "platform", None), "value", "") or "").lower()
    scope_id = str(getattr(source, "scope_id", None) or "").strip()
    if platform == "slack" and scope_id:
        metadata.update({"slack_team_id": scope_id, "scope_id": scope_id})
    profile = str(getattr(source, "profile", None) or "").strip()
    if profile:
        metadata["hermes_profile"] = profile
    return metadata


async def deliver_operator_notice(
    *, adapter: Any, source: Any, kind: str, content: str, user_config: Any,
    chat_metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Deliver one notice according to policy and return the resolved mode.

    ``admin_dm`` is fail-closed: missing admins and failed direct sends are logged and never
    retried against the originating chat.  Adapters may implement ``send_direct_notice`` when a
    user identity cannot be passed directly to their ordinary ``send`` method (Discord does).
    """
    mode = resolve_operator_notice_mode(user_config, kind)
    if mode == "chat":
        if adapter is not None:
            await adapter.send(source.chat_id, content, metadata=chat_metadata)
        return mode

    if mode == "log":
        logger.warning("Operator notice [%s]: %s", kind, content)
        return mode

    admin_ids = _adapter_admin_ids(adapter)
    if not admin_ids:
        logger.warning(
            "Operator notice [%s] configured for admin_dm but no allow_admin_from identities "
            "are configured; notice retained in logs: %s", kind, content,
        )
        return mode

    direct_sender = getattr(type(adapter), "send_direct_notice", None)
    direct_metadata = _direct_metadata(source)
    for admin_id in admin_ids:
        try:
            if callable(direct_sender):
                result = await direct_sender(adapter, admin_id, content, metadata=direct_metadata)
            else:
                result = await adapter.send(admin_id, content, metadata=direct_metadata)
            if not getattr(result, "success", False):
                logger.warning(
                    "Operator notice [%s] DM to admin %s failed; notice retained in logs: %s",
                    kind, admin_id, content,
                )
        except Exception:
            logger.warning(
                "Operator notice [%s] DM to admin %s failed; notice retained in logs: %s",
                kind, admin_id, content, exc_info=True,
            )
    return mode
