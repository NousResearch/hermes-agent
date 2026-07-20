"""Weixin-specific facade over the platform-neutral group archive store."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from gateway.group_archive_store import GroupArchiveStore, coerce_archive_timestamp, group_archive_store_path
from gateway.group_policy_store import normalize_group_scope_key, split_group_scope_key
from gateway.weixin_group_policies import get_group_policy, summarize_group_policy_state


WEIXIN_GROUP_PLATFORM = "weixin"


def _archive_db_path() -> Path:
    return group_archive_store_path()


def _weixin_scope_key(chat_id: str) -> str:
    normalized_chat_id = str(chat_id or "").strip()
    if not normalized_chat_id:
        raise ValueError("chat_id is required")
    return normalize_group_scope_key(WEIXIN_GROUP_PLATFORM, normalized_chat_id)


def _scope_message_to_weixin_message(message: dict[str, Any]) -> dict[str, Any]:
    result = dict(message)
    _, chat_id = split_group_scope_key(
        str(result.get("scope_key") or _weixin_scope_key(str(result.get("chat_id") or "")))
    )
    result["chat_id"] = chat_id
    result["group_id"] = chat_id
    return result


def _rewrite_weixin_summary_text(summary_text: str, *, chat_id: str, report_date: str) -> str:
    lines = [line for line in str(summary_text or "").splitlines() if line.strip()]
    first = f"{report_date} 微信群 {chat_id} 日报"
    if not lines:
        return first
    return "\n".join([first] + lines[1:])


def _scope_report_to_weixin_report(report: dict[str, Any]) -> dict[str, Any]:
    result = dict(report)
    _, chat_id = split_group_scope_key(
        str(result.get("scope_key") or _weixin_scope_key(str(result.get("chat_id") or "")))
    )
    result["chat_id"] = chat_id
    result["group_id"] = chat_id
    result["summary_text"] = _rewrite_weixin_summary_text(
        result.get("summary_text") or "",
        chat_id=chat_id,
        report_date=str(result.get("report_date") or ""),
    )
    return result


def _scope_delivery_to_weixin_delivery(delivery: dict[str, Any] | None) -> dict[str, Any] | None:
    if delivery is None:
        return None
    result = dict(delivery)
    _, chat_id = split_group_scope_key(
        str(result.get("scope_key") or _weixin_scope_key(str(result.get("chat_id") or "")))
    )
    result["chat_id"] = chat_id
    result["group_id"] = chat_id
    return result


class WeixinGroupArchiveStore:
    def __init__(self, db_path: Path | None = None):
        self._store = GroupArchiveStore(db_path=db_path)
        self.db_path = self._store.db_path

    def archive_inbound_message(
        self,
        *,
        chat_id: str,
        message_id: str,
        observed_at: datetime | str | int | float,
        user_id: str | None,
        user_name: str | None,
        text: str,
        media_types: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_media_types = [str(item).strip() for item in (media_types or []) if str(item).strip()]
        row = self._store.archive_message(
            scope_key=_weixin_scope_key(chat_id),
            message_id=str(message_id or "").strip(),
            observed_at=observed_at,
            user_id=user_id,
            user_name=user_name,
            text=str(text or "").strip(),
            has_media=bool(normalized_media_types),
            media_types=normalized_media_types,
            segment_types=normalized_media_types,
        )
        return _scope_message_to_weixin_message(row)

    def list_recent_messages(
        self,
        *,
        chat_id: str | None = None,
        report_date: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        scope_key = _weixin_scope_key(chat_id) if chat_id else None
        return [
            _scope_message_to_weixin_message(item)
            for item in self._store.list_recent_messages(scope_key=scope_key, report_date=report_date, limit=limit)
        ]

    def search_messages(
        self,
        *,
        query: str,
        chat_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        scope_key = _weixin_scope_key(chat_id) if chat_id else None
        return [
            _scope_message_to_weixin_message(item)
            for item in self._store.search_messages(query=query, scope_key=scope_key, limit=limit)
        ]

    def build_snapshot_report(self, *, chat_id: str, report_date: str) -> dict[str, Any]:
        report = self._store.build_snapshot_report(scope_key=_weixin_scope_key(chat_id), report_date=report_date)
        return _scope_report_to_weixin_report(report)

    def rollup_daily(self, *, chat_id: str, report_date: str) -> dict[str, Any]:
        normalized_chat_id = str(chat_id or "").strip()
        if not normalized_chat_id:
            raise ValueError("chat_id is required")
        policy = get_group_policy(normalized_chat_id)
        result = self._store.rollup_daily(
            scope_key=_weixin_scope_key(normalized_chat_id),
            report_date=report_date,
            purge_raw_after_rollup=bool(policy.get("purge_raw_after_rollup", True)),
        )
        if result.get("report"):
            result = dict(result)
            result["report"] = _scope_report_to_weixin_report(result["report"])
        return result

    def rollup_due_days(self, *, now: datetime | None = None) -> dict[str, Any]:
        current_time = coerce_archive_timestamp(now)
        today = current_time.date().isoformat()
        candidates = self._store.list_due_scope_dates(before_date=today)
        reports: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        purged_total = 0
        for candidate in candidates:
            scope_key = str(candidate.get("scope_key") or "")
            platform, chat_id = split_group_scope_key(scope_key)
            if platform != WEIXIN_GROUP_PLATFORM:
                continue
            policy = get_group_policy(chat_id)
            if not bool(policy.get("daily_report_enabled")):
                continue
            result = self.rollup_daily(chat_id=chat_id, report_date=str(candidate.get("report_date") or ""))
            if result.get("success"):
                reports.append(result["report"])
                purged_total += int(result.get("purged_raw_messages") or 0)
            else:
                failures.append(
                    {
                        "chat_id": chat_id,
                        "report_date": candidate.get("report_date"),
                        "error": result.get("error") or "Unknown rollup failure",
                    }
                )
        return {
            "success": not failures,
            "rolled_up_count": len(reports),
            "purged_raw_messages": purged_total,
            "reports": reports,
            "failures": failures,
        }

    def list_reports(self, *, chat_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        scope_key = _weixin_scope_key(chat_id) if chat_id else None
        return [
            _scope_report_to_weixin_report(item)
            for item in self._store.list_reports(scope_key=scope_key, limit=limit)
            if item.get("platform") == WEIXIN_GROUP_PLATFORM
        ]

    def get_report(self, *, chat_id: str, report_date: str) -> dict[str, Any] | None:
        report = self._store.get_report(scope_key=_weixin_scope_key(chat_id), report_date=report_date)
        return _scope_report_to_weixin_report(report) if report is not None else None

    def get_report_delivery(
        self,
        *,
        chat_id: str,
        report_date: str,
        delivery_key: str,
    ) -> dict[str, Any] | None:
        return _scope_delivery_to_weixin_delivery(
            self._store.get_report_delivery(
                scope_key=_weixin_scope_key(chat_id),
                report_date=report_date,
                delivery_key=delivery_key,
            )
        )

    def has_successful_report_delivery(
        self,
        *,
        chat_id: str,
        report_date: str,
        delivery_key: str,
    ) -> bool:
        return self._store.has_successful_report_delivery(
            scope_key=_weixin_scope_key(chat_id),
            report_date=report_date,
            delivery_key=delivery_key,
        )

    def record_report_delivery(
        self,
        *,
        chat_id: str,
        report_date: str,
        delivery_key: str,
        target: str,
        error: str | None = None,
        attempted_at: datetime | None = None,
    ) -> dict[str, Any]:
        return _scope_delivery_to_weixin_delivery(
            self._store.record_report_delivery(
                scope_key=_weixin_scope_key(chat_id),
                report_date=report_date,
                delivery_key=delivery_key,
                target=target,
                error=error,
                attempted_at=attempted_at,
            )
        ) or {}

    def describe_group_reporting(self, *, chat_id: str) -> dict[str, Any]:
        policy = get_group_policy(chat_id)
        summary = summarize_group_policy_state(policy)
        daily_targets = [target for target in [policy.get("daily_report_target")] if str(target or "").strip()]
        manual_targets = [target for target in [policy.get("manual_report_target")] if str(target or "").strip()]
        return {
            "chat_id": str(chat_id or "").strip(),
            "scope_key": _weixin_scope_key(chat_id),
            "policy": summary,
            "effective_targets": {
                "daily_report_targets": daily_targets,
                "manual_report_targets": manual_targets,
            },
            "report_control": {
                "archive_enabled": bool(policy.get("archive_enabled")),
                "daily_report_enabled": bool(policy.get("daily_report_enabled")),
                "purge_raw_after_rollup": bool(policy.get("purge_raw_after_rollup", True)),
                "reply_behavior": summary.get("reply_behavior"),
            },
        }

    def get_runtime_stats(self) -> dict[str, Any]:
        return self._store.get_storage_stats(before_date=coerce_archive_timestamp(None).date().isoformat())


def run_due_weixin_group_rollups(*, now: datetime | None = None) -> dict[str, Any]:
    store = WeixinGroupArchiveStore()
    return store.rollup_due_days(now=now)


def format_group_report_for_delivery(
    report: dict[str, Any],
    *,
    group_name: str | None = None,
) -> str:
    label = str(group_name or "").strip() or str(report.get("chat_id") or "").strip() or "未知群"
    report_date = str(report.get("report_date") or "").strip() or "未知日期"
    snapshot = bool(report.get("snapshot"))
    title = f"微信群快照｜{label}｜{report_date}" if snapshot else f"微信群日报｜{label}｜{report_date}"
    summary_text = str(report.get("summary_text") or "").strip()
    if not summary_text:
        summary_text = f"{report_date} 微信群 {label} 报告"
    return f"{title}\n{summary_text}"
