from __future__ import annotations

from typing import Any, Callable

from gateway.group_control_plane import build_group_runtime_snapshot


def unique_report_targets(values: list[Any]) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        targets.append(text)
    return targets


def worker_report_targets(
    workers: list[dict[str, Any]],
    key: str,
    *,
    require_daily_enabled: bool = False,
) -> list[str]:
    values: list[Any] = []
    for item in workers:
        if not isinstance(item, dict):
            continue
        if require_daily_enabled and not bool(item.get("daily_report_enabled")):
            continue
        values.append(item.get(key))
    return unique_report_targets(values)


def build_qq_group_runtime_status_details(
    target: str,
    *,
    get_group_policy_fn: Callable[[str], dict[str, Any]],
    get_group_monitoring_overlay_fn: Callable[[str], dict[str, Any] | None],
) -> dict[str, Any]:
    group_id = str(target).replace("group:", "").strip()
    policy = get_group_policy_fn(group_id)
    overlay = get_group_monitoring_overlay_fn(group_id)
    workers = list((overlay or {}).get("workers") or [])
    worker_names = [
        worker_name
        for worker_name in (
            str(item.get("worker_name") or "").strip()
            for item in workers
            if isinstance(item, dict)
        )
        if worker_name
    ]
    policy_mode = str(policy.get("mode") or "default").strip() or "default"
    overlay_mode = str((overlay or {}).get("mode") or "").strip()
    effective_mode = overlay_mode if policy_mode == "default" and overlay_mode else policy_mode
    effective_archive_enabled = bool(policy.get("archive_enabled") or (overlay or {}).get("archive_enabled"))
    effective_daily_enabled = bool(
        policy.get("daily_report_enabled") or (overlay or {}).get("daily_report_enabled")
    )
    daily_targets = unique_report_targets(
        [policy.get("daily_report_target")]
        + worker_report_targets(
            workers,
            "daily_report_target",
            require_daily_enabled=True,
        )
    )
    manual_targets = unique_report_targets(
        [policy.get("manual_report_target")]
        + worker_report_targets(
            workers,
            "manual_report_target",
        )
    )
    snapshot = build_group_runtime_snapshot(
        platform_label="QQ 群",
        target_label=group_id,
        effective_mode=effective_mode,
        archive_enabled=effective_archive_enabled,
        daily_report_enabled=effective_daily_enabled,
        daily_targets=daily_targets,
        manual_targets=manual_targets,
        worker_names=worker_names,
    )
    return snapshot.to_status_details()


def build_weixin_group_runtime_status_details(
    target: str,
    *,
    get_group_policy_fn: Callable[[str], dict[str, Any]],
    describe_group_reporting_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    policy = get_group_policy_fn(target)
    reporting = describe_group_reporting_fn(chat_id=target)
    effective_mode = str(policy.get("mode") or "default").strip() or "default"
    effective_archive_enabled = bool(policy.get("archive_enabled"))
    effective_daily_enabled = bool(policy.get("daily_report_enabled"))
    daily_targets = unique_report_targets(
        list((reporting.get("effective_targets") or {}).get("daily_report_targets") or [])
    )
    manual_targets = unique_report_targets(
        list((reporting.get("effective_targets") or {}).get("manual_report_targets") or [])
    )
    snapshot = build_group_runtime_snapshot(
        platform_label="微信群",
        target_label=target,
        effective_mode=effective_mode,
        archive_enabled=effective_archive_enabled,
        daily_report_enabled=effective_daily_enabled,
        daily_targets=daily_targets,
        manual_targets=manual_targets,
        worker_names=[],
    )
    return snapshot.to_status_details()
