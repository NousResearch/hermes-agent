"""Shared runtime archive summaries across group platforms."""

from __future__ import annotations

import logging
from typing import Any, Callable

from gateway.group_runtime_platform_specs import (
    GroupArchiveRuntimePlatformSpec,
    build_group_archive_runtime_platform_specs,
)


logger = logging.getLogger(__name__)


def _normalize_platform_archive_stats(stats: dict[str, Any] | None, *, platform: str) -> dict[str, Any]:
    payload = stats if isinstance(stats, dict) else {}
    raw_scope_count = payload.get("raw_scope_count")
    if raw_scope_count is None:
        raw_scope_count = payload.get("raw_group_count")
    due_scope_count = payload.get("due_scope_count")
    if due_scope_count is None:
        due_scope_count = payload.get("due_rollup_count")
    return {
        "platform": platform,
        "raw_message_count": int(payload.get("raw_message_count") or 0),
        "raw_scope_count": int(raw_scope_count or 0),
        "due_rollup_count": int(payload.get("due_rollup_count") or 0),
        "due_scope_count": int(due_scope_count or 0),
        "report_count": int(payload.get("report_count") or 0),
        "oldest_raw_date": payload.get("oldest_raw_date"),
        "newest_raw_date": payload.get("newest_raw_date"),
    }


def _choose_oldest(*values: Any) -> str | None:
    normalized = [str(value).strip() for value in values if str(value or "").strip()]
    return min(normalized) if normalized else None


def _choose_newest(*values: Any) -> str | None:
    normalized = [str(value).strip() for value in values if str(value or "").strip()]
    return max(normalized) if normalized else None


def build_group_archive_runtime_summary(
    *,
    platform_specs: list[GroupArchiveRuntimePlatformSpec] | None = None,
    load_qq_archive_stats_fn: Callable[[], dict[str, Any]] | None = None,
    load_weixin_archive_stats_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if platform_specs is None:
        platform_specs = build_group_archive_runtime_platform_specs(
            load_qq_archive_stats_fn=load_qq_archive_stats_fn,
            load_weixin_archive_stats_fn=load_weixin_archive_stats_fn,
        )

    platform_stats: dict[str, dict[str, Any]] = {}
    for spec in platform_specs or []:
        try:
            stats = _normalize_platform_archive_stats(spec.load_runtime_stats(), platform=spec.platform)
        except Exception as exc:
            logger.debug("Failed to load %s archive runtime stats: %s", spec.platform, exc)
            stats = _normalize_platform_archive_stats({}, platform=spec.platform)
        platform_stats[spec.platform] = stats

    all_stats = list(platform_stats.values())
    return {
        "raw_message_count": sum(item["raw_message_count"] for item in all_stats),
        "raw_scope_count": sum(item["raw_scope_count"] for item in all_stats),
        "due_rollup_count": sum(item["due_rollup_count"] for item in all_stats),
        "due_scope_count": sum(item["due_scope_count"] for item in all_stats),
        "report_count": sum(item["report_count"] for item in all_stats),
        "oldest_raw_date": _choose_oldest(*(item["oldest_raw_date"] for item in all_stats)),
        "newest_raw_date": _choose_newest(*(item["newest_raw_date"] for item in all_stats)),
        "platforms": platform_stats,
    }
