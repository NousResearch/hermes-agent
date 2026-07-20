"""Shared runtime-policy helpers for group-capable gateway adapters."""

from __future__ import annotations

import logging
from typing import Any, Callable


logger = logging.getLogger(__name__)


def qq_group_message_allowed(
    group_id: str,
    *,
    allow_all_groups: bool,
    allowed_groups: set[str],
    has_policy: bool,
    overlay_active: bool,
) -> bool:
    """Return whether a QQ group should enter runtime processing."""
    normalized_group_id = str(group_id or "").strip()
    if not normalized_group_id:
        return False
    if overlay_active or has_policy:
        return True
    if not allow_all_groups and allowed_groups and normalized_group_id not in allowed_groups:
        return False
    if not allow_all_groups and not allowed_groups:
        return False
    return True


def qq_policy_has_runtime_override(policy: dict[str, Any]) -> bool:
    """Return True when explicit policy state should win over runtime overlay."""
    mode = str(policy.get("mode") or "").strip().lower()
    return bool(
        mode not in {"", "default"}
        or bool(policy.get("archive_enabled"))
        or bool(policy.get("daily_report_enabled"))
        or str(policy.get("daily_report_target") or "").strip()
        or str(policy.get("manual_report_target") or "").strip()
        or not bool(policy.get("purge_raw_after_rollup", True))
    )


def merge_qq_overlay_policy(policy: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge QQ intel collect-only overlay into a base group policy."""
    merged = dict(policy or {})
    workers = list((overlay or {}).get("workers") or [])
    daily_report_targets: list[str] = []
    manual_report_targets: list[str] = []
    notify_targets: list[str] = []
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        if bool(worker.get("daily_report_enabled")):
            target = str(worker.get("daily_report_target") or "").strip()
            if target and target not in daily_report_targets:
                daily_report_targets.append(target)
        manual_target = str(worker.get("manual_report_target") or "").strip()
        if manual_target and manual_target not in manual_report_targets:
            manual_report_targets.append(manual_target)
        notify_target = str(worker.get("notify_target") or "").strip()
        if notify_target and notify_target not in notify_targets:
            notify_targets.append(notify_target)

    merged["mode"] = "collect_only"
    merged["archive_enabled"] = bool((overlay or {}).get("archive_enabled"))
    merged["daily_report_enabled"] = bool((overlay or {}).get("daily_report_enabled"))
    merged["daily_report_target"] = daily_report_targets[0] if len(daily_report_targets) == 1 else None
    merged["manual_report_target"] = manual_report_targets[0] if len(manual_report_targets) == 1 else None
    merged["notify_target"] = notify_targets[0] if len(notify_targets) == 1 else None
    merged["daily_report_targets"] = daily_report_targets
    merged["manual_report_targets"] = manual_report_targets
    merged["notify_targets"] = notify_targets
    return merged


def resolve_qq_effective_group_policy(
    group_id: str,
    *,
    policy_loader: Callable[[str], dict[str, Any]],
    default_policy_loader: Callable[[str], dict[str, Any]],
    overlay_loader: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Load QQ group policy, folding in intel overlay when appropriate."""
    normalized_group_id = str(group_id or "").strip()
    if not normalized_group_id:
        return default_policy_loader("")
    try:
        policy = policy_loader(normalized_group_id)
    except Exception:
        logger.exception("Failed to load QQ group policy for %s", normalized_group_id)
        policy = default_policy_loader(normalized_group_id)

    overlay = overlay_loader(normalized_group_id)
    if bool((overlay or {}).get("active")) and not qq_policy_has_runtime_override(policy):
        return merge_qq_overlay_policy(policy, overlay)
    return policy


def weixin_group_message_allowed(
    chat_id: str,
    *,
    has_policy: bool,
    group_policy_mode: str,
    group_allow_from: set[str],
) -> bool:
    """Return whether a Weixin group should enter runtime processing."""
    normalized_chat_id = str(chat_id or "").strip()
    if not normalized_chat_id:
        return False
    if has_policy:
        return True
    if group_policy_mode == "disabled":
        return False
    if group_policy_mode == "allowlist":
        return normalized_chat_id in group_allow_from
    return True


def resolve_weixin_effective_group_policy(
    chat_id: str,
    *,
    policy_loader: Callable[[str], dict[str, Any]],
    default_policy_loader: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Load Weixin group policy with safe fallback."""
    normalized_chat_id = str(chat_id or "").strip()
    if not normalized_chat_id:
        return default_policy_loader("")
    try:
        return policy_loader(normalized_chat_id)
    except Exception:
        logger.exception("Failed to load Weixin group policy for %s", normalized_chat_id)
        return default_policy_loader(normalized_chat_id)
