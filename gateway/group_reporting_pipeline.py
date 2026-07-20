"""Shared helpers for group-report retry collection and delivery execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class GroupReportDeliveryPlan:
    platform: str
    kind: str
    entity_id: str
    report_date: str
    delivery_key: str
    target: str
    job: dict[str, Any]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    on_success: Callable[[], None] | None = None


def collect_reports_for_delivery_retry(
    reports: Iterable[dict[str, Any]] | None,
    *,
    entity_key: str,
    list_reports: Callable[[int], list[dict[str, Any]]],
    lookback_days: int,
    now,
    limit: int = 512,
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for report in reports or []:
        entity_id = str(report.get(entity_key) or "").strip()
        report_date = str(report.get("report_date") or "").strip()
        if entity_id and report_date:
            merged[(entity_id, report_date)] = report

    cutoff_date = (now.date() - timedelta(days=lookback_days)).isoformat()
    for report in list_reports(limit):
        entity_id = str(report.get(entity_key) or "").strip()
        report_date = str(report.get("report_date") or "").strip()
        if not entity_id or not report_date or report_date < cutoff_date:
            continue
        merged.setdefault((entity_id, report_date), report)
    return list(merged.values())


def build_policy_report_delivery_plans(
    reports: Iterable[dict[str, Any]] | None,
    *,
    platform: str,
    entity_key: str,
    get_policy: Callable[[str], dict[str, Any]],
    has_successful_delivery: Callable[[str, str, str], bool],
    format_report: Callable[..., str],
    build_job: Callable[[str, str, str], dict[str, Any]],
) -> list[GroupReportDeliveryPlan]:
    plans: list[GroupReportDeliveryPlan] = []
    for report in reports or []:
        entity_id = str(report.get(entity_key) or "").strip()
        report_date = str(report.get("report_date") or "").strip()
        if not entity_id or not report_date:
            continue

        policy = get_policy(entity_id)
        target = str(policy.get("daily_report_target") or "").strip()
        if not target:
            continue

        delivery_key = f"policy:{target}"
        if has_successful_delivery(entity_id, report_date, delivery_key):
            continue

        plans.append(
            GroupReportDeliveryPlan(
                platform=platform,
                kind="policy_daily",
                entity_id=entity_id,
                report_date=report_date,
                delivery_key=delivery_key,
                target=target,
                job=build_job(entity_id, report_date, target),
                content=format_report(report, group_name=policy.get("group_name")),
            )
        )
    return plans


def execute_report_delivery_plans(
    plans: Iterable[GroupReportDeliveryPlan] | None,
    *,
    deliver_result: Callable[..., str | None],
    record_delivery: Callable[[str, str, str, str, str | None], dict[str, Any]],
    adapters=None,
    loop=None,
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for plan in plans or []:
        error = deliver_result(plan.job, plan.content, adapters=adapters, loop=loop)
        state = record_delivery(
            plan.entity_id,
            plan.report_date,
            plan.delivery_key,
            plan.target,
            error,
        )
        if error is None and callable(plan.on_success):
            plan.on_success()
        outcomes.append(
            {
                "platform": plan.platform,
                "kind": plan.kind,
                "entity_id": plan.entity_id,
                "report_date": plan.report_date,
                "delivery_key": plan.delivery_key,
                "target": plan.target,
                "error": error,
                "state": state,
                **dict(plan.metadata or {}),
            }
        )
    return outcomes
