"""Shared runtime archive summaries across group platforms."""

from __future__ import annotations

import logging
from typing import Any, Callable

from gateway.qq_group_archive import QqGroupArchiveStore
from gateway.weixin_group_archive import WeixinGroupArchiveStore


logger = logging.getLogger(__name__)

QQ_PLATFORM = "qq_napcat"
WEIXIN_PLATFORM = "weixin"


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
    load_qq_archive_stats_fn: Callable[[], dict[str, Any]] | None = None,
    load_weixin_archive_stats_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    load_qq_archive_stats_fn = load_qq_archive_stats_fn or QqGroupArchiveStore().get_runtime_stats
    load_weixin_archive_stats_fn = load_weixin_archive_stats_fn or WeixinGroupArchiveStore().get_runtime_stats

    try:
        qq_stats = _normalize_platform_archive_stats(load_qq_archive_stats_fn(), platform=QQ_PLATFORM)
    except Exception as exc:
        logger.debug("Failed to load QQ archive runtime stats: %s", exc)
        qq_stats = _normalize_platform_archive_stats({}, platform=QQ_PLATFORM)

    try:
        weixin_stats = _normalize_platform_archive_stats(load_weixin_archive_stats_fn(), platform=WEIXIN_PLATFORM)
    except Exception as exc:
        logger.debug("Failed to load Weixin archive runtime stats: %s", exc)
        weixin_stats = _normalize_platform_archive_stats({}, platform=WEIXIN_PLATFORM)

    return {
        "raw_message_count": qq_stats["raw_message_count"] + weixin_stats["raw_message_count"],
        "raw_scope_count": qq_stats["raw_scope_count"] + weixin_stats["raw_scope_count"],
        "due_rollup_count": qq_stats["due_rollup_count"] + weixin_stats["due_rollup_count"],
        "due_scope_count": qq_stats["due_scope_count"] + weixin_stats["due_scope_count"],
        "report_count": qq_stats["report_count"] + weixin_stats["report_count"],
        "oldest_raw_date": _choose_oldest(qq_stats["oldest_raw_date"], weixin_stats["oldest_raw_date"]),
        "newest_raw_date": _choose_newest(qq_stats["newest_raw_date"], weixin_stats["newest_raw_date"]),
        "platforms": {
            QQ_PLATFORM: qq_stats,
            WEIXIN_PLATFORM: weixin_stats,
        },
    }


def build_legacy_qq_archive_summary(group_archive: dict[str, Any] | None) -> dict[str, Any]:
    summary = group_archive if isinstance(group_archive, dict) else {}
    platform_stats = summary.get("platforms") if isinstance(summary.get("platforms"), dict) else {}
    qq_stats = platform_stats.get(QQ_PLATFORM) if isinstance(platform_stats, dict) else {}
    qq_stats = qq_stats if isinstance(qq_stats, dict) else {}
    return {
        "raw_message_count": int(qq_stats.get("raw_message_count") or 0),
        "raw_group_count": int(qq_stats.get("raw_scope_count") or 0),
        "due_rollup_count": int(qq_stats.get("due_rollup_count") or 0),
        "report_count": int(qq_stats.get("report_count") or 0),
        "oldest_raw_date": qq_stats.get("oldest_raw_date"),
        "newest_raw_date": qq_stats.get("newest_raw_date"),
    }
