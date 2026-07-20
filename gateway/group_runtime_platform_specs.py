"""Platform specs for shared group runtime summaries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from gateway.group_control_plane import build_group_runtime_snapshot
from gateway.qq_group_archive import QqGroupArchiveStore
from gateway.qq_group_policies import list_group_policies as list_qq_group_policies
from gateway.qq_intel_assignments import list_intel_workers
from gateway.weixin_group_archive import WeixinGroupArchiveStore
from gateway.weixin_group_policies import list_group_policies as list_weixin_group_policies


logger = logging.getLogger(__name__)

QQ_PLATFORM = "qq_napcat"
WEIXIN_PLATFORM = "weixin"


@dataclass(frozen=True)
class GroupArchiveRuntimePlatformSpec:
    platform: str
    load_runtime_stats: Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class GroupMonitoringRuntimePlatformSpec:
    platform: str
    load_summary: Callable[[], dict[str, Any]]


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


def _load_qq_group_monitoring_summary(
    *,
    list_qq_group_policies_fn: Callable[[], list[dict[str, Any]]],
    list_qq_intel_workers_fn: Callable[..., list[dict[str, Any]]],
) -> dict[str, Any]:
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
            "chat_id": group_id,
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
                "chat_id": group_id,
                "group_name": group_name,
                "archive_enabled": True,
                "daily_report_enabled": False,
                "daily_targets": [],
                "manual_targets": [],
                "worker_names": [],
            },
        )
        entry["group_id"] = entry.get("group_id") or group_id
        entry["chat_id"] = entry.get("chat_id") or group_id
        entry["group_name"] = str(entry.get("group_name") or group_name).strip() or (group_name or group_id)
        entry["archive_enabled"] = bool(entry.get("archive_enabled")) or True
        if bool(worker.get("daily_report_enabled")):
            entry["daily_report_enabled"] = True
        _append_unique(entry["worker_names"], worker.get("worker_name"))
        _append_unique(entry["daily_targets"], worker.get("daily_report_target"))
        _append_unique(entry["manual_targets"], worker.get("manual_report_target"))

    groups: list[dict[str, Any]] = []
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
        groups.append(
            {
                "platform": QQ_PLATFORM,
                "platform_label": snapshot.platform_label,
                "group_id": str(entry.get("group_id") or "").strip(),
                "chat_id": str(entry.get("chat_id") or entry.get("group_id") or "").strip(),
                "group_name": str(entry.get("group_name") or entry.get("group_id") or "").strip(),
                **snapshot.to_status_details(),
            }
        )

    return {
        "active_worker_count": qq_active_worker_count,
        "groups": groups,
    }


def _load_weixin_group_monitoring_summary(
    *,
    list_weixin_group_policies_fn: Callable[[], list[dict[str, Any]]],
    describe_weixin_group_reporting_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    try:
        weixin_policies = list_weixin_group_policies_fn()
    except Exception as exc:
        logger.debug("Failed to load Weixin collect-only policy snapshot: %s", exc)
        weixin_policies = []

    groups: list[dict[str, Any]] = []
    for policy in weixin_policies:
        if not isinstance(policy, dict):
            continue
        if str(policy.get("mode") or "").strip().lower() != "collect_only":
            continue
        chat_id = str(policy.get("chat_id") or policy.get("group_id") or "").strip()
        if not chat_id:
            continue
        try:
            reporting = describe_weixin_group_reporting_fn(chat_id=chat_id) or {}
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
        groups.append(
            {
                "platform": WEIXIN_PLATFORM,
                "platform_label": snapshot.platform_label,
                "group_id": chat_id,
                "chat_id": chat_id,
                "group_name": str(policy.get("group_name") or chat_id).strip() or chat_id,
                **snapshot.to_status_details(),
            }
        )

    return {
        "active_worker_count": 0,
        "groups": groups,
    }


def build_group_monitoring_runtime_platform_specs(
    *,
    list_qq_group_policies_fn: Callable[[], list[dict[str, Any]]] | None = None,
    list_qq_intel_workers_fn: Callable[..., list[dict[str, Any]]] | None = None,
    list_weixin_group_policies_fn: Callable[[], list[dict[str, Any]]] | None = None,
    describe_weixin_group_reporting_fn: Callable[..., dict[str, Any]] | None = None,
) -> list[GroupMonitoringRuntimePlatformSpec]:
    list_qq_group_policies_fn = list_qq_group_policies_fn or list_qq_group_policies
    list_qq_intel_workers_fn = list_qq_intel_workers_fn or list_intel_workers
    list_weixin_group_policies_fn = list_weixin_group_policies_fn or list_weixin_group_policies
    describe_weixin_group_reporting_fn = (
        describe_weixin_group_reporting_fn
        or WeixinGroupArchiveStore().describe_group_reporting
    )

    return [
        GroupMonitoringRuntimePlatformSpec(
            platform=QQ_PLATFORM,
            load_summary=lambda: _load_qq_group_monitoring_summary(
                list_qq_group_policies_fn=list_qq_group_policies_fn,
                list_qq_intel_workers_fn=list_qq_intel_workers_fn,
            ),
        ),
        GroupMonitoringRuntimePlatformSpec(
            platform=WEIXIN_PLATFORM,
            load_summary=lambda: _load_weixin_group_monitoring_summary(
                list_weixin_group_policies_fn=list_weixin_group_policies_fn,
                describe_weixin_group_reporting_fn=describe_weixin_group_reporting_fn,
            ),
        ),
    ]


def build_group_archive_runtime_platform_specs(
    *,
    load_qq_archive_stats_fn: Callable[[], dict[str, Any]] | None = None,
    load_weixin_archive_stats_fn: Callable[[], dict[str, Any]] | None = None,
) -> list[GroupArchiveRuntimePlatformSpec]:
    load_qq_archive_stats_fn = load_qq_archive_stats_fn or QqGroupArchiveStore().get_runtime_stats
    load_weixin_archive_stats_fn = load_weixin_archive_stats_fn or WeixinGroupArchiveStore().get_runtime_stats

    return [
        GroupArchiveRuntimePlatformSpec(
            platform=QQ_PLATFORM,
            load_runtime_stats=load_qq_archive_stats_fn,
        ),
        GroupArchiveRuntimePlatformSpec(
            platform=WEIXIN_PLATFORM,
            load_runtime_stats=load_weixin_archive_stats_fn,
        ),
    ]
