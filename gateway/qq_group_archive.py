"""QQ-specific facade over the platform-neutral group archive store."""

from __future__ import annotations

from datetime import datetime
import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from gateway.group_archive_store import (
    ARCHIVE_DB_FILENAME,
    GroupArchiveStore,
    coerce_archive_timestamp,
    group_archive_store_path,
)
from gateway.group_policy_store import normalize_group_scope_key, split_group_scope_key
from gateway.qq_intel_assignments import (
    get_group_monitoring_overlay,
    list_active_daily_report_workers_for_group,
    summarize_intel_worker_assignment,
)
from gateway.qq_group_policies import get_group_policy, summarize_group_policy_state


QQ_GROUP_PLATFORM = "qq_napcat"


def _archive_db_path() -> Path:
    return group_archive_store_path()


def _qq_scope_key(group_id: str) -> str:
    normalized_group_id = str(group_id or "").strip()
    if not normalized_group_id:
        raise ValueError("group_id is required")
    return normalize_group_scope_key(QQ_GROUP_PLATFORM, normalized_group_id)


def _normalize_text(payload: dict[str, Any]) -> str:
    raw_text = str(payload.get("raw_message") or "").strip()
    if raw_text and "[CQ:" not in raw_text:
        return raw_text

    segments = payload.get("message")
    if not isinstance(segments, list):
        return raw_text

    parts: list[str] = []
    placeholders: list[str] = []
    for segment in segments:
        seg_type = str((segment or {}).get("type") or "").lower()
        data = (segment or {}).get("data") or {}
        if seg_type == "text":
            text = str(data.get("text") or "")
            if text:
                parts.append(text)
        elif seg_type == "at":
            qq = str(data.get("qq") or "").strip()
            if qq:
                parts.append(f"@{qq}")
        elif seg_type == "image":
            placeholders.append("[图片]")
        elif seg_type == "record":
            placeholders.append("[语音]")
        elif seg_type == "video":
            placeholders.append("[视频]")
        elif seg_type == "file":
            placeholders.append("[文件]")

    normalized = "".join(parts).strip()
    if normalized:
        return normalized
    return " ".join(placeholders).strip()


def _synthetic_message_id(payload: dict[str, Any], observed_at: datetime, text: str) -> str:
    digest = hashlib.sha1(
        "|".join(
            [
                str(payload.get("group_id") or ""),
                str(payload.get("user_id") or ""),
                observed_at.isoformat(),
                text,
            ]
        ).encode("utf-8", errors="ignore")
    ).hexdigest()
    return f"synthetic:{digest[:20]}"


def _segment_types(payload: dict[str, Any]) -> list[str]:
    segments = payload.get("message")
    if not isinstance(segments, list):
        return []
    result: list[str] = []
    for segment in segments:
        seg_type = str((segment or {}).get("type") or "").strip().lower()
        if seg_type:
            result.append(seg_type)
    return result


def _collect_unique_targets(values: list[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _scope_message_to_qq_message(message: dict[str, Any]) -> dict[str, Any]:
    result = dict(message)
    _, group_id = split_group_scope_key(str(result.get("scope_key") or _qq_scope_key(str(result.get("chat_id") or ""))))
    result["group_id"] = group_id
    return result


def _rewrite_qq_summary_text(summary_text: str, *, group_id: str, report_date: str) -> str:
    lines = [line for line in str(summary_text or "").splitlines() if line.strip()]
    first = f"{report_date} QQ 群 {group_id} 日报"
    if not lines:
        return first
    return "\n".join([first] + lines[1:])


def _scope_report_to_qq_report(report: dict[str, Any]) -> dict[str, Any]:
    result = dict(report)
    _, group_id = split_group_scope_key(str(result.get("scope_key") or _qq_scope_key(str(result.get("chat_id") or ""))))
    result["group_id"] = group_id
    result["summary_text"] = _rewrite_qq_summary_text(
        result.get("summary_text") or "",
        group_id=group_id,
        report_date=str(result.get("report_date") or ""),
    )
    return result


def _scope_delivery_to_qq_delivery(delivery: dict[str, Any] | None) -> dict[str, Any] | None:
    if delivery is None:
        return None
    result = dict(delivery)
    _, group_id = split_group_scope_key(str(result.get("scope_key") or _qq_scope_key(str(result.get("chat_id") or ""))))
    result["group_id"] = group_id
    return result


class QqGroupArchiveStore:
    def __init__(self, db_path: Path | None = None):
        self._store = GroupArchiveStore(db_path=db_path)
        self.db_path = self._store.db_path

    def archive_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        group_id = str(payload.get("group_id") or "").strip()
        if not group_id:
            raise ValueError("QQ group archive requires group_id")

        observed_at = coerce_archive_timestamp(payload.get("time"))
        text = _normalize_text(payload)
        message_id = str(
            payload.get("message_id")
            or payload.get("raw_message_id")
            or payload.get("real_id")
            or ""
        ).strip() or _synthetic_message_id(payload, observed_at, text)

        sender = payload.get("sender") or {}
        user_id = str(payload.get("user_id") or "").strip() or None
        user_name = (
            str(sender.get("card") or "").strip()
            or str(sender.get("nickname") or "").strip()
            or user_id
        )
        segment_types = _segment_types(payload)
        media_types = [item for item in segment_types if item in {"image", "record", "video", "file"}]
        row = self._store.archive_message(
            scope_key=_qq_scope_key(group_id),
            message_id=message_id,
            observed_at=observed_at,
            user_id=user_id,
            user_name=user_name,
            text=text,
            has_media=bool(media_types),
            media_types=media_types,
            segment_types=segment_types,
        )
        return _scope_message_to_qq_message(row)

    def list_recent_messages(
        self,
        *,
        group_id: str | None = None,
        report_date: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        scope_key = _qq_scope_key(group_id) if group_id else None
        return [
            _scope_message_to_qq_message(item)
            for item in self._store.list_recent_messages(scope_key=scope_key, report_date=report_date, limit=limit)
        ]

    def search_messages(
        self,
        *,
        query: str,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        scope_key = _qq_scope_key(group_id) if group_id else None
        return [
            _scope_message_to_qq_message(item)
            for item in self._store.search_messages(query=query, scope_key=scope_key, limit=limit)
        ]

    def build_snapshot_report(self, *, group_id: str, report_date: str) -> dict[str, Any]:
        report = self._store.build_snapshot_report(scope_key=_qq_scope_key(group_id), report_date=report_date)
        return _scope_report_to_qq_report(report)

    def rollup_daily(self, *, group_id: str, report_date: str) -> dict[str, Any]:
        normalized_group_id = str(group_id or "").strip()
        if not normalized_group_id:
            raise ValueError("group_id is required")
        policy = get_group_policy(normalized_group_id)
        result = self._store.rollup_daily(
            scope_key=_qq_scope_key(normalized_group_id),
            report_date=report_date,
            purge_raw_after_rollup=bool(policy.get("purge_raw_after_rollup", True)),
        )
        if result.get("report"):
            result = dict(result)
            result["report"] = _scope_report_to_qq_report(result["report"])
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
            platform, group_id = split_group_scope_key(scope_key)
            if platform != QQ_GROUP_PLATFORM:
                continue
            policy = get_group_policy(group_id)
            has_worker_reports = bool(list_active_daily_report_workers_for_group(group_id))
            if not bool(policy.get("daily_report_enabled")) and not has_worker_reports:
                continue
            result = self.rollup_daily(group_id=group_id, report_date=str(candidate.get("report_date") or ""))
            if result.get("success"):
                reports.append(result["report"])
                purged_total += int(result.get("purged_raw_messages") or 0)
            else:
                failures.append(
                    {
                        "group_id": group_id,
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

    def list_reports(self, *, group_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        scope_key = _qq_scope_key(group_id) if group_id else None
        return [
            _scope_report_to_qq_report(item)
            for item in self._store.list_reports(scope_key=scope_key, limit=limit)
            if item.get("platform") == QQ_GROUP_PLATFORM
        ]

    def get_report(self, *, group_id: str, report_date: str) -> dict[str, Any] | None:
        report = self._store.get_report(scope_key=_qq_scope_key(group_id), report_date=report_date)
        if report is None:
            return None
        return _scope_report_to_qq_report(report)

    def get_runtime_stats(self, *, now: datetime | None = None) -> dict[str, Any]:
        current_time = coerce_archive_timestamp(now)
        today = current_time.date().isoformat()

        raw_message_count = 0
        raw_group_ids: set[str] = set()
        report_count = 0
        oldest_raw_date: str | None = None
        newest_raw_date: str | None = None
        self._store._ensure_schema()
        with sqlite3.connect(str(self.db_path), timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            raw_rows = conn.execute("SELECT group_id, local_date FROM raw_messages").fetchall()
            report_rows = conn.execute("SELECT group_id FROM daily_reports").fetchall()

        for row in raw_rows:
            try:
                platform, group_id = split_group_scope_key(
                    normalize_group_scope_key(str(row["group_id"]))
                    if ":" in str(row["group_id"])
                    else _qq_scope_key(str(row["group_id"]))
                )
            except ValueError:
                continue
            if platform != QQ_GROUP_PLATFORM:
                continue
            raw_message_count += 1
            raw_group_ids.add(group_id)
            local_date = str(row["local_date"] or "").strip() or None
            if local_date:
                oldest_raw_date = local_date if oldest_raw_date is None else min(oldest_raw_date, local_date)
                newest_raw_date = local_date if newest_raw_date is None else max(newest_raw_date, local_date)

        for row in report_rows:
            try:
                platform, _group_id = split_group_scope_key(
                    normalize_group_scope_key(str(row["group_id"]))
                    if ":" in str(row["group_id"])
                    else _qq_scope_key(str(row["group_id"]))
                )
            except ValueError:
                continue
            if platform == QQ_GROUP_PLATFORM:
                report_count += 1

        due_rollup_count = 0
        due_groups: set[str] = set()
        for item in self._store.list_due_scope_dates(before_date=today):
            platform, group_id = split_group_scope_key(str(item["scope_key"]))
            if platform != QQ_GROUP_PLATFORM:
                continue
            policy = get_group_policy(group_id)
            has_worker_reports = bool(list_active_daily_report_workers_for_group(group_id))
            if not bool(policy.get("daily_report_enabled")) and not has_worker_reports:
                continue
            due_rollup_count += 1
            due_groups.add(group_id)

        return {
            "raw_message_count": raw_message_count,
            "raw_group_count": len(raw_group_ids),
            "due_rollup_count": due_rollup_count,
            "due_group_count": len(due_groups),
            "report_count": report_count,
            "oldest_raw_date": oldest_raw_date,
            "newest_raw_date": newest_raw_date,
        }

    def _group_archive_state(self, *, group_id: str) -> dict[str, Any]:
        aliases = [str(group_id).strip(), _qq_scope_key(group_id)]
        self._store._ensure_schema()
        with sqlite3.connect(str(self.db_path), timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            raw_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS raw_message_count,
                    MAX(local_date) AS newest_raw_date
                FROM raw_messages
                WHERE group_id IN (?, ?)
                """,
                (aliases[0], aliases[1]),
            ).fetchone()
            report_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS report_count,
                    MAX(report_date) AS newest_report_date
                FROM daily_reports
                WHERE group_id IN (?, ?)
                """,
                (aliases[0], aliases[1]),
            ).fetchone()
        return {
            "raw_message_count": int((raw_row["raw_message_count"] if raw_row else 0) or 0),
            "report_count": int((report_row["report_count"] if report_row else 0) or 0),
            "newest_raw_date": (raw_row["newest_raw_date"] if raw_row else None),
            "newest_report_date": (report_row["newest_report_date"] if report_row else None),
        }

    def describe_group_reporting(self, *, group_id: str) -> dict[str, Any]:
        normalized_group_id = str(group_id or "").strip()
        if not normalized_group_id:
            raise ValueError("group_id is required")

        policy = get_group_policy(normalized_group_id)
        policy_summary = summarize_group_policy_state(policy)
        overlay = get_group_monitoring_overlay(normalized_group_id)
        workers = list(overlay.get("workers") or [])
        worker_assignments = [
            summarize_intel_worker_assignment(worker)
            for worker in workers
        ]
        group_name = str(policy.get("group_name") or "").strip()
        if not group_name:
            group_name = str((workers[0].get("target_group_name") if workers else "") or "").strip()
        effective_mode = str(policy.get("mode") or "").strip() or "default"
        if bool(overlay.get("active")) and effective_mode == "default":
            effective_mode = "collect_only"

        worker_daily_targets = _collect_unique_targets(
            [worker.get("daily_report_target") for worker in workers if bool(worker.get("daily_report_enabled"))]
        )
        worker_manual_targets = _collect_unique_targets(
            [worker.get("manual_report_target") for worker in workers]
        )
        worker_notify_targets = _collect_unique_targets(
            [worker.get("notify_target") for worker in workers]
        )
        effective_daily_targets = _collect_unique_targets(
            ([policy.get("daily_report_target")] if bool(policy.get("daily_report_enabled")) else [])
            + worker_daily_targets
        )
        effective_manual_targets = _collect_unique_targets(
            [policy.get("manual_report_target")] + worker_manual_targets
        )
        worker_names = _collect_unique_targets([worker.get("worker_name") for worker in workers])
        collect_only = effective_mode == "collect_only"

        archive_state = self._group_archive_state(group_id=normalized_group_id)
        report_control = {
            "daily_report_enabled": bool(policy.get("daily_report_enabled")) or bool(overlay.get("daily_report_enabled")),
            "manual_report_available": bool(effective_manual_targets),
            "daily_report_targets": effective_daily_targets,
            "manual_report_targets": effective_manual_targets,
            "notify_targets": worker_notify_targets,
            "purge_raw_after_rollup": bool(policy.get("purge_raw_after_rollup", True)),
        }
        summary_parts = [
            f"mode={effective_mode}",
            "reply=no_reply" if collect_only else f"reply={policy_summary['reply_behavior']}",
        ]
        if effective_daily_targets:
            summary_parts.append(f"daily={','.join(effective_daily_targets)}")
        if effective_manual_targets:
            summary_parts.append(f"manual={','.join(effective_manual_targets)}")
        if worker_names:
            summary_parts.append(f"workers={','.join(worker_names)}")

        return {
            "group_id": normalized_group_id,
            "group_name": group_name or normalized_group_id,
            "mode": effective_mode,
            "collect_only": collect_only,
            "replies_disabled": effective_mode in {"collect_only", "disabled"},
            "reply_behavior": "no_reply" if collect_only else policy_summary["reply_behavior"],
            "monitoring_intent": "collect_only_monitoring" if collect_only else "default",
            "worker_names": worker_names,
            "worker_assignments": worker_assignments,
            "active_worker_count": len(worker_assignments),
            "purge_raw_after_rollup": bool(policy.get("purge_raw_after_rollup", True)),
            "policy_summary": policy_summary,
            "policy_targets": {
                "daily_report_target": str(policy.get("daily_report_target") or "").strip() or None,
                "manual_report_target": str(policy.get("manual_report_target") or "").strip() or None,
            },
            "worker_targets": {
                "daily_report_targets": worker_daily_targets,
                "manual_report_targets": worker_manual_targets,
                "notify_targets": worker_notify_targets,
            },
            "effective_targets": {
                "daily_report_targets": effective_daily_targets,
                "manual_report_targets": effective_manual_targets,
            },
            "delivery_targets": {
                "daily_report_targets": effective_daily_targets,
                "manual_report_targets": effective_manual_targets,
                "notify_targets": worker_notify_targets,
            },
            "report_control": report_control,
            "archive_state": archive_state,
            "overlay": {
                "active": bool(overlay.get("active")),
                "mode": str(overlay.get("mode") or "default"),
                "daily_report_enabled": bool(overlay.get("daily_report_enabled")),
                "monitoring_intent": str(overlay.get("monitoring_intent") or "default"),
                "worker_names": worker_names,
            },
            "summary": " | ".join(summary_parts),
        }

    def get_report_delivery(
        self,
        *,
        group_id: str,
        report_date: str,
        delivery_key: str,
    ) -> dict[str, Any] | None:
        record = self._store.get_report_delivery(
            scope_key=_qq_scope_key(group_id),
            report_date=report_date,
            delivery_key=delivery_key,
        )
        return _scope_delivery_to_qq_delivery(record)

    def has_successful_report_delivery(
        self,
        *,
        group_id: str,
        report_date: str,
        delivery_key: str,
    ) -> bool:
        return self._store.has_successful_report_delivery(
            scope_key=_qq_scope_key(group_id),
            report_date=report_date,
            delivery_key=delivery_key,
        )

    def record_report_delivery(
        self,
        *,
        group_id: str,
        report_date: str,
        delivery_key: str,
        target: str,
        error: str | None = None,
        attempted_at: datetime | None = None,
    ) -> dict[str, Any]:
        record = self._store.record_report_delivery(
            scope_key=_qq_scope_key(group_id),
            report_date=report_date,
            delivery_key=delivery_key,
            target=target,
            error=error,
            attempted_at=attempted_at,
        )
        qq_record = _scope_delivery_to_qq_delivery(record)
        if qq_record is None:
            raise RuntimeError("failed to persist QQ report delivery state")
        return qq_record


def run_due_qq_group_rollups(*, now: datetime | None = None) -> dict[str, Any]:
    """Run due QQ group daily rollups outside the transport adapter lifecycle."""
    store = QqGroupArchiveStore()
    return store.rollup_due_days(now=now)


def format_group_report_for_delivery(
    report: dict[str, Any],
    *,
    group_name: str | None = None,
) -> str:
    label = str(group_name or "").strip() or str(report.get("group_id") or "").strip() or "未知群"
    report_date = str(report.get("report_date") or "").strip() or "未知日期"
    snapshot = bool(report.get("snapshot"))
    title = f"QQ 群快照｜{label}｜{report_date}" if snapshot else f"QQ 群日报｜{label}｜{report_date}"
    summary_text = str(report.get("summary_text") or "").strip()
    if not summary_text:
        summary_text = f"{report_date} QQ 群 {label} 报告"
    return f"{title}\n{summary_text}"
