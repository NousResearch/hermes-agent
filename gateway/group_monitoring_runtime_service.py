"""Shared runtime monitoring summaries for collect-only group targets."""

from __future__ import annotations

import logging
from typing import Any, Callable

from gateway.group_runtime_platform_specs import (
    GroupMonitoringRuntimePlatformSpec,
    build_group_monitoring_runtime_platform_specs,
)


logger = logging.getLogger(__name__)


def _sorted_group_entries(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        groups,
        key=lambda item: (
            str(item.get("platform_label") or "").strip(),
            str(item.get("group_name") or "").strip(),
            str(item.get("group_id") or item.get("chat_id") or "").strip(),
        ),
    )


def _normalize_platform_group_summary(
    summary: dict[str, Any] | None,
    *,
    platform: str,
) -> dict[str, Any]:
    payload = summary if isinstance(summary, dict) else {}
    groups: list[dict[str, Any]] = []
    for group in payload.get("groups") or []:
        if not isinstance(group, dict):
            continue
        normalized = dict(group)
        platform_name = str(normalized.get("platform") or platform).strip() or platform
        normalized["platform"] = platform_name
        groups.append(normalized)
    return {
        "platform": platform,
        "active_worker_count": int(payload.get("active_worker_count") or 0),
        "groups": groups,
    }


def build_group_monitoring_summary(
    *,
    platform_specs: list[GroupMonitoringRuntimePlatformSpec] | None = None,
    list_qq_group_policies_fn: Callable[[], list[dict[str, Any]]] | None = None,
    list_qq_intel_workers_fn: Callable[..., list[dict[str, Any]]] | None = None,
    list_weixin_group_policies_fn: Callable[[], list[dict[str, Any]]] | None = None,
    describe_weixin_group_reporting_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if platform_specs is None:
        platform_specs = build_group_monitoring_runtime_platform_specs(
            list_qq_group_policies_fn=list_qq_group_policies_fn,
            list_qq_intel_workers_fn=list_qq_intel_workers_fn,
            list_weixin_group_policies_fn=list_weixin_group_policies_fn,
            describe_weixin_group_reporting_fn=describe_weixin_group_reporting_fn,
        )

    groups: list[dict[str, Any]] = []
    platform_counts: dict[str, int] = {}
    platform_active_worker_counts: dict[str, int] = {}
    total_active_worker_count = 0

    for spec in platform_specs or []:
        try:
            platform_summary = _normalize_platform_group_summary(
                spec.load_summary(),
                platform=spec.platform,
            )
        except Exception as exc:
            logger.debug("Failed to load %s collect-only runtime summary: %s", spec.platform, exc)
            platform_summary = _normalize_platform_group_summary({}, platform=spec.platform)
        platform_groups = platform_summary["groups"]
        active_worker_count = platform_summary["active_worker_count"]
        platform_counts[spec.platform] = len(platform_groups)
        platform_active_worker_counts[spec.platform] = active_worker_count
        total_active_worker_count += active_worker_count
        groups.extend(platform_groups)

    groups = _sorted_group_entries(groups)
    return {
        "active_collect_only_groups": len(groups),
        "active_worker_count": total_active_worker_count,
        "platform_counts": platform_counts,
        "platform_active_worker_counts": platform_active_worker_counts,
        "groups": groups[:8],
    }
