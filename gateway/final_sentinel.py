"""Gateway final-message sentinel helpers.

Telegram/Discord may show a standalone ``COMPLETE`` marker, but that marker
must mean true-idle rather than merely "the main response was delivered".
The platform/delivery gate lives here with a small lifecycle snapshot so call
sites can suppress the marker whenever follow-up work is still pending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


FINAL_MESSAGE_SENTINEL = "COMPLETE"
SUPPORTED_FINAL_SENTINEL_PLATFORMS = {"telegram", "discord"}


@dataclass(frozen=True)
class FinalSentinelLifecycleSnapshot:
    """Minimal true-idle inputs for sending the final gateway sentinel.

    Every field represents known unfinished work.  Unknown work should be
    modeled by setting the closest field to ``True``; if true idle cannot be
    proven, callers must suppress the sentinel.
    """

    active_session: bool = False
    active_session_task: bool = False
    pending_message: bool = False
    drain_task_spawned: bool = False
    post_delivery_callback_pending: bool = False
    approval_pending: bool = False
    running_agent_active: bool = False
    background_review_pending: bool = False
    final_report_pending: bool = False
    gateway_active_run: bool = False

    def blocking_reasons(self) -> Tuple[str, ...]:
        reasons: list[str] = []
        for name, value in (
            ("active_session", self.active_session),
            ("active_session_task", self.active_session_task),
            ("pending_message", self.pending_message),
            ("drain_task_spawned", self.drain_task_spawned),
            ("post_delivery_callback_pending", self.post_delivery_callback_pending),
            ("approval_pending", self.approval_pending),
            ("running_agent_active", self.running_agent_active),
            ("background_review_pending", self.background_review_pending),
            ("final_report_pending", self.final_report_pending),
            ("gateway_active_run", self.gateway_active_run),
        ):
            if value:
                reasons.append(name)
        return tuple(reasons)

    def is_true_idle(self) -> bool:
        """Return True only when no known lifecycle work remains."""
        return not self.blocking_reasons()


def platform_name(platform: Any) -> str:
    """Normalize a Platform enum / raw platform value to lowercase text."""
    value = getattr(platform, "value", platform)
    return str(value or "").lower()


def is_final_sentinel_platform(platform: Any) -> bool:
    """Return True when this platform should receive final sentinels."""
    return platform_name(platform) in SUPPORTED_FINAL_SENTINEL_PLATFORMS


def strip_trailing_final_sentinel(text: str) -> str:
    """Remove a model-produced trailing standalone sentinel if present.

    The durable marker is sent by the gateway as a separate message. If the
    model includes a final standalone COMPLETE line, strip it from the body so
    the user does not see duplicates or a marker buried in the main response.
    """
    if not isinstance(text, str) or not text:
        return text

    lines = text.rstrip().splitlines()
    if not lines:
        return ""
    if lines[-1].strip() != FINAL_MESSAGE_SENTINEL:
        return text
    return "\n".join(lines[:-1]).rstrip()


def should_send_final_sentinel(
    *,
    platform: Any,
    message_type: Any = None,
    response_delivered: bool = False,
    is_ephemeral_response: bool = False,
    failed: bool = False,
    suppressed_or_stale: bool = False,
    lifecycle: FinalSentinelLifecycleSnapshot | None = None,
) -> bool:
    """Return True when a gateway turn may emit standalone COMPLETE.

    The response must have been visibly delivered *and* lifecycle state must be
    true-idle.  A missing lifecycle snapshot is treated as not idle because the
    incident class here is false-positive COMPLETE when idle cannot be proven.
    """
    if not is_final_sentinel_platform(platform):
        return False
    if not response_delivered:
        return False
    if failed or suppressed_or_stale or is_ephemeral_response:
        return False

    message_type_value = getattr(message_type, "value", message_type)
    if str(message_type_value or "").lower() == "command":
        return False

    if lifecycle is None:
        return False
    return lifecycle.is_true_idle()
