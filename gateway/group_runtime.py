"""Platform-neutral group chat runtime helpers.

This module holds group-chat decision logic that should be shared by QQ,
Weixin, and any future group-capable messaging platforms.  Platform adapters
are responsible for translating raw payloads into these normalized inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class GroupTriggerState:
    """Normalized trigger inputs for one inbound group message."""

    require_explicit_trigger: bool = True
    slash_command: bool = False
    mentioned_bot: bool = False
    replied_to_bot: bool = False
    shared_followup: bool = False
    user_followup: bool = False
    recent_session_followup: bool = False
    name_trigger: bool = False


@dataclass(frozen=True)
class GroupBatchItem:
    """Normalized batch inputs for project-collaboration style group handling."""

    speaker_id: str
    text: str = ""
    direct_trigger_reason: Optional[str] = None
    is_admin: bool = False
    has_nontext_media: bool = False


@dataclass(frozen=True)
class GroupDispatchThresholds:
    """Thresholds for score-based project-group dispatch decisions."""

    min_messages: int = 4
    min_speakers: int = 3
    min_chars: int = 160


def resolve_group_trigger_reason(state: GroupTriggerState) -> Optional[str]:
    """Resolve the first matching group trigger reason in priority order."""
    if not state.require_explicit_trigger:
        return "require_mention_disabled"
    if state.slash_command:
        return "slash_command"
    if state.mentioned_bot:
        return "bot_mention"
    if state.replied_to_bot:
        return "reply_to_bot"
    if state.shared_followup:
        return "group_followup_window"
    if state.user_followup:
        return "user_followup_window"
    if state.recent_session_followup:
        return "recent_session_followup"
    if state.name_trigger:
        return "name_trigger"
    return None


_EXPLICIT_GROUP_TRIGGER_REASONS = {"bot_mention", "reply_to_bot", "name_trigger"}


def build_group_message_metadata(
    *,
    trigger_reason: str | None = None,
    explicit_reason: str | None = None,
    group_policy_mode: str | None = None,
    allow_model_dispatch: bool | None = None,
) -> dict[str, Any] | None:
    """Build normalized cross-platform metadata for inbound group messages."""
    normalized_trigger_reason = str(trigger_reason or "").strip()
    normalized_explicit_reason = str(explicit_reason or "").strip()
    if not normalized_explicit_reason and normalized_trigger_reason in _EXPLICIT_GROUP_TRIGGER_REASONS:
        normalized_explicit_reason = normalized_trigger_reason

    explicit_addressed = bool(normalized_explicit_reason)
    metadata: dict[str, Any] = {}
    if normalized_trigger_reason:
        metadata["group_trigger_reason"] = normalized_trigger_reason
    if normalized_explicit_reason:
        metadata["explicit_group_trigger_reason"] = normalized_explicit_reason
        metadata["address_reason"] = normalized_explicit_reason
    if normalized_trigger_reason or normalized_explicit_reason:
        metadata["explicit_group_trigger"] = explicit_addressed
        metadata["explicit_addressed"] = explicit_addressed
        metadata["requires_reply"] = explicit_addressed
    if group_policy_mode is not None:
        metadata["group_policy_mode"] = str(group_policy_mode or "").strip().lower()
    if allow_model_dispatch is not None:
        metadata["allow_model_dispatch"] = bool(allow_model_dispatch)
    return metadata or None


def text_looks_like_request(text: str) -> bool:
    """Best-effort heuristic for request-like group messages."""
    body = str(text or "").strip().lower()
    if not body:
        return False
    if any(token in body for token in ("?", "？")):
        return True
    request_markers = (
        "吗",
        "么",
        "咋",
        "怎么",
        "如何",
        "为啥",
        "为什么",
        "能不能",
        "可不可以",
        "要不要",
        "看看",
        "看下",
        "看一下",
        "帮我",
        "帮忙",
        "查下",
        "查一下",
        "分析",
        "安排",
        "处理",
        "修",
        "改",
        "做一下",
        "做到哪",
        "进度",
        "还在",
        "在吗",
        "在?",
        "在？",
        "什么情况",
        "啥情况",
        "有没有",
        "发一下",
        "给我",
    )
    return any(marker in body for marker in request_markers)


def decide_project_group_dispatch(
    batch: Iterable[GroupBatchItem],
    *,
    thresholds: GroupDispatchThresholds | None = None,
) -> tuple[bool, str]:
    """Return whether a batched group discussion should hit the main model."""
    items = list(batch)
    if not items:
        return False, "empty"

    for item in items:
        if item.direct_trigger_reason:
            return True, f"direct_trigger:{item.direct_trigger_reason}"
        if item.is_admin:
            return True, "admin_user"
        if item.has_nontext_media:
            return True, "media"

    thresholds = thresholds or GroupDispatchThresholds()
    score = 0
    reasons: list[str] = []

    unique_speakers = {item.speaker_id.strip() for item in items if item.speaker_id.strip()}
    total_chars = sum(len(str(item.text or "").strip()) for item in items)

    if len(items) >= thresholds.min_messages:
        score += 1
        reasons.append("message_volume")

    if len(unique_speakers) >= thresholds.min_speakers:
        score += 1
        reasons.append("multi_speaker")

    for item in items:
        if text_looks_like_request(item.text):
            score += 2
            reasons.append("explicit_request")
            break

    if total_chars >= thresholds.min_chars:
        score += 1
        reasons.append("text_volume")

    reason = ",".join(reasons) or f"score={score}"
    return score >= 2, reason
