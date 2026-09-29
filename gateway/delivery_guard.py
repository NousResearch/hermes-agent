"""Fail-closed policy for server-marked outbound handoff turns."""

from typing import Any, Mapping


HANDOFF_MAX_CHARS = 2_000


class HandoffDeliveryBlocked(RuntimeError):
    """The final handoff payload is not safe to deliver."""


def guard_pre_delivery(*, platform: Any, content: str, target: Mapping[str, Any],
                       session_id: str | None, turn_id: str | None,
                       metadata: Mapping[str, Any] | None) -> None:
    """Block invalid server-marked Telegram handoffs before adapter send/chunking."""
    if str(getattr(platform, "value", platform)).lower() != "telegram" or not (metadata or {}).get("handoff"):
        return
    if not session_id or not turn_id:
        raise HandoffDeliveryBlocked("HANDOFF_DELIVERY_BLOCKED: missing server handoff marker")
    if len(content) > HANDOFF_MAX_CHARS:
        raise HandoffDeliveryBlocked(f"HANDOFF_DELIVERY_BLOCKED: {len(content)} > {HANDOFF_MAX_CHARS}")
    if not (content.startswith("```text\n") and content.endswith("\n```") and content.count("```") == 2):
        raise HandoffDeliveryBlocked("HANDOFF_DELIVERY_BLOCKED: invalid fenced text block")
