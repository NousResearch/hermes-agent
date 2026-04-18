"""Shared runtime monitoring summaries for collect-only group targets."""

from __future__ import annotations

import logging
from typing import Any, Callable

from gateway.group_control_plane import build_group_runtime_snapshot
from gateway.qq_group_policies import list_group_policies as list_qq_group_policies
from gateway.qq_intel_assignments import list_intel_workers
from gateway.weixin_group_archive import WeixinGroupArchiveStore
from gateway.weixin_group_policies import list_group_policies as list_weixin_group_policies


logger = logging.getLogger(__name__)

QQ_PLATFORM = "qq_napcat"
WEIXIN_PLATFORM = "weixin"


def _normalize_target_list(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _append_unique(values: list[str], candidate: Any) -> None:
    text = str(candidate or "").strip()
    if text and text not in values:
        values.append(text)


def _sorted_group_entries(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        groups,
        key=lambda item: (
            str(item.get("platform_label") or "").strip(),
            str(item.get("group_name") or "").strip(),
            str(item.get("group_id") or item.get("chat_id") or "").strip(),
        ),
    )


def build_group_monitoring_summary(
    *,
    list_qq_group_policies_fn: Callable[[], list[dict[str, Any]]] | None = None,
    list_qq_intel_workers_fn: Callable[..., list[dict[str, Any]]] | None = None,
    list_weixin_group_policies_fn: Callable[[], list[dict[str, Any]]] | None = None,
    describe_weixin_group_reporting_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    list_qq_group_policies_fn = list_qq_group_policies_fn or list_qq_group_policies
    list_qq_intel_workers_fn = list_qq_intel_workers_fn or list_intel_workers
    list_weixin_group_policies_fn = list_weixin_group_policies_fn or list_weixin_group_policies
    describe_weixin_group_reporting = describe_weixin_group_reporting_fn
    if describe_weixin_group_reporting is None:
        describe_weixin_group_reporting = WeixinGroupArchiveStore().describe_group_reporting

    qq_groups_by_id: dict[str, dict[str, Any]] = {}
    try:
        qq_policies = list_qq_group_policies_fn()
    except Exception as exc:
        logger.debug("Failed to load QQ collect-only policy snapshot: %s", exc)
        qq_policies = []

    for policy in qq_policies:
        if not isinstance(policy, dict):
            continue
        if str(policy.get("mode") or "").strip().lower() != "collect_only":
            continue
        group_id = str(policy.get("group_id") or "").strip()
        if not group_id:
            continue
        qq_groups_by_id[group_id] = {
            "group_id": group_id,
            "group_name": str(policy.get("group_name") or group_id).strip() or group_id,
            "archive_enabled": bool(policy.get("archive_enabled")),
            "daily_report_enabled": bool(policy.get("daily_report_enabled")),
            "daily_targets": _normalize_target_list([policy.get("daily_report_target")]),
            "manual_targets": _normalize_target_list([policy.get("manual_report_target")]),
            "worker_names": [],
        }

    try:
        qq_workers = list_qq_intel_workers_fn(status="active_collecting")
    except Exception as exc:
        logger.debug("Failed to load QQ collect-only worker snapshot: %s", exc)
        qq_workers = []

    qq_active_worker_count = 0
    for worker in qq_workers:
        if not isinstance(worker, dict):
            continue
        if str(worker.get("status") or "").strip().lower() != "active_collecting":
            continue
        qq_active_worker_count += 1
        group_id = str(worker.get("target_group_id") or "").strip()
        group_ref = str(worker.get("target_group_ref") or "").strip()
        if not group_id and group_ref.startswith("group:"):
            group_id = group_ref.split(":", 1)[1]
        group_name = str(worker.get("target_group_name") or group_id).strip() or group_id
        if not group_id and not group_name:
            continue
        key = group_id or group_name
        entry = qq_groups_by_id.setdefault(
            key,
            {
                "group_id": group_id,
                "group_name": group_name,
                "archive_enabled": True,
                "daily_report_enabled": False,
                "daily_targets": [],
                "manual_targets": [],
                "worker_names": [],
            },
        )
        entry["group_id"] = entry.get("group_id") or group_id
        entry["group_name"] = str(entry.get("group_name") or group_name).strip() or (group_name or group_id)
        entry["archive_enabled"] = bool(entry.get("archive_enabled")) or True
        if bool(worker.get("daily_report_enabled")):
            entry["daily_report_enabled"] = True
        _append_unique(entry["worker_names"], worker.get("worker_name"))
        _append_unique(entry["daily_targets"], worker.get("daily_report_target"))
        _append_unique(entry["manual_targets"], worker.get("manual_report_target"))

    qq_groups: list[dict[str, Any]] = []
    for entry in qq_groups_by_id.values():
        snapshot = build_group_runtime_snapshot(
            platform_label="QQ 群",
            target_label=str(entry.get("group_id") or entry.get("group_name") or "").strip(),
            effective_mode="collect_only",
            archive_enabled=bool(entry.get("archive_enabled")),
            daily_report_enabled=bool(entry.get("daily_report_enabled")),
            daily_targets=entry.get("daily_targets"),
            manual_targets=entry.get("manual_targets"),
            worker_names=entry.get("worker_names"),
        )
        qq_groups.append(
            {
                "platform": QQ_PLATFORM,
                "platform_label": snapshot.platform_label,
                "group_id": str(entry.get("group_id") or "").strip(),
                "chat_id": str(entry.get("group_id") or "").strip(),
                "group_name": str(entry.get("group_name") or entry.get("group_id") or "").strip(),
                **snapshot.to_status_details(),
            }
        )

    weixin_groups: list[dict[str, Any]] = []
    try:
        weixin_policies = list_weixin_group_policies_fn()
    except Exception as exc:
        logger.debug("Failed to load Weixin collect-only policy snapshot: %s", exc)
        weixin_policies = []

    for policy in weixin_policies:
        if not isinstance(policy, dict):
            continue
        if str(policy.get("mode") or "").strip().lower() != "collect_only":
            continue
        chat_id = str(policy.get("chat_id") or policy.get("group_id") or "").strip()
        if not chat_id:
            continue
        try:
            reporting = describe_weixin_group_reporting(chat_id=chat_id) or {}
        except Exception as exc:
            logger.debug("Failed to load Weixin reporting state for %s: %s", chat_id, exc)
            reporting = {}
        effective_targets = reporting.get("effective_targets") or {}
        snapshot = build_group_runtime_snapshot(
            platform_label="微信群",
            target_label=chat_id,
            effective_mode="collect_only",
            archive_enabled=bool(policy.get("archive_enabled")),
            daily_report_enabled=bool(policy.get("daily_report_enabled")),
            daily_targets=effective_targets.get("daily_report_targets"),
            manual_targets=effective_targets.get("manual_report_targets"),
            worker_names=[],
        )
        weixin_groups.append(
            {
                "platform": WEIXIN_PLATFORM,
                "platform_label": snapshot.platform_label,
                "group_id": chat_id,
                "chat_id": chat_id,
                "group_name": str(policy.get("group_name") or chat_id).strip() or chat_id,
                **snapshot.to_status_details(),
            }
        )

    groups = _sorted_group_entries([*qq_groups, *weixin_groups])
    platform_counts = {
        QQ_PLATFORM: len(qq_groups),
        WEIXIN_PLATFORM: len(weixin_groups),
    }
    platform_active_worker_counts = {
        QQ_PLATFORM: qq_active_worker_count,
        WEIXIN_PLATFORM: 0,
    }
    return {
        "active_collect_only_groups": len(groups),
        "active_worker_count": qq_active_worker_count,
        "platform_counts": platform_counts,
        "platform_active_worker_counts": platform_active_worker_counts,
        "groups": groups[:8],
    }


def build_legacy_qq_monitoring_summary(group_monitoring: dict[str, Any] | None) -> dict[str, Any]:
    summary = group_monitoring if isinstance(group_monitoring, dict) else {}
    groups: list[dict[str, Any]] = []
    for entry in summary.get("groups") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("platform") or "").strip() != QQ_PLATFORM:
            continue
        group_id = str(entry.get("group_id") or entry.get("chat_id") or "").strip()
        if not group_id:
            continue
        groups.append(
            {
                "group_id": group_id,
                "group_name": str(entry.get("group_name") or group_id).strip() or group_id,
                "mode": str(entry.get("effective_mode") or "collect_only").strip() or "collect_only",
                "worker_names": _normalize_target_list(entry.get("worker_names")),
                "daily_report_enabled": bool(entry.get("daily_report_enabled")),
            }
        )
    platform_workers = summary.get("platform_active_worker_counts") or {}
    return {
        "active_collect_only_groups": len(groups),
        "active_worker_count": int(platform_workers.get(QQ_PLATFORM) or 0),
        "groups": groups[:8],
    }
