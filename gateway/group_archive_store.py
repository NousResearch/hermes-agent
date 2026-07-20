"""Platform-neutral group archive store and daily rollup logic."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator

from gateway.group_policy_store import (
    LEGACY_DEFAULT_PLATFORM,
    normalize_group_scope_key,
    split_group_scope_key,
)
from hermes_constants import get_hermes_home
from hermes_time import get_timezone, now as hermes_now


ARCHIVE_DB_FILENAME = "qq_group_archive.db"
_URL_RE = re.compile(r"https?://\S+")


def group_archive_store_path() -> Path:
    return get_hermes_home() / ARCHIVE_DB_FILENAME


def coerce_archive_timestamp(value: Any) -> datetime:
    tz = get_timezone()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(tz) if tz else value.astimezone()
        return value.replace(tzinfo=tz) if tz else value.astimezone()

    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is not None:
                    return parsed.astimezone(tz) if tz else parsed.astimezone()
                return parsed.replace(tzinfo=tz) if tz else parsed.astimezone()

    if value in (None, ""):
        return hermes_now()

    try:
        ts = float(value)
    except (TypeError, ValueError):
        return hermes_now()

    if tz is not None:
        return datetime.fromtimestamp(ts, tz)
    return datetime.fromtimestamp(ts).astimezone()


def _normalize_scope_key(raw_scope: Any, *, legacy_default_platform: str) -> str | None:
    scope_text = str(raw_scope or "").strip()
    if not scope_text:
        return None
    try:
        if ":" in scope_text:
            return normalize_group_scope_key(scope_text)
        return normalize_group_scope_key(legacy_default_platform, scope_text)
    except ValueError:
        return None


def _scope_aliases(scope_key: str, *, legacy_default_platform: str) -> list[str]:
    normalized_scope_key = normalize_group_scope_key(scope_key)
    platform, chat_id = split_group_scope_key(normalized_scope_key)
    aliases = [normalized_scope_key]
    if platform == legacy_default_platform:
        aliases.append(chat_id)
    unique: list[str] = []
    seen: set[str] = set()
    for item in aliases:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _question_like(text: str) -> bool:
    body = str(text or "").strip().lower()
    if not body:
        return False
    if "?" in body or "？" in body:
        return True
    markers = ("吗", "么", "怎么", "如何", "为什么", "咋", "看下", "看看", "安排", "处理", "在吗")
    return any(marker in body for marker in markers)


def _placeholder_in_clause(values: list[str]) -> str:
    return ", ".join("?" for _ in values)


class GroupArchiveStore:
    def __init__(
        self,
        db_path: Path | None = None,
        *,
        legacy_default_platform: str = LEGACY_DEFAULT_PLATFORM,
    ):
        self.db_path = Path(db_path or group_archive_store_path())
        self.legacy_default_platform = legacy_default_platform
        self._init_lock = threading.Lock()
        self._initialized = False

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._ensure_schema()
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=30)
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS raw_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        group_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        local_date TEXT NOT NULL,
                        observed_at TEXT NOT NULL,
                        user_id TEXT,
                        user_name TEXT,
                        text TEXT,
                        has_media INTEGER NOT NULL DEFAULT 0,
                        media_types_json TEXT NOT NULL DEFAULT '[]',
                        segment_types_json TEXT NOT NULL DEFAULT '[]',
                        UNIQUE(group_id, message_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_qq_raw_messages_group_date
                    ON raw_messages(group_id, local_date, observed_at);

                    CREATE TABLE IF NOT EXISTS daily_reports (
                        group_id TEXT NOT NULL,
                        report_date TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        total_messages INTEGER NOT NULL,
                        unique_speakers INTEGER NOT NULL,
                        first_message_at TEXT,
                        last_message_at TEXT,
                        summary_text TEXT NOT NULL,
                        summary_json TEXT NOT NULL,
                        PRIMARY KEY(group_id, report_date)
                    );

                    CREATE TABLE IF NOT EXISTS report_deliveries (
                        group_id TEXT NOT NULL,
                        report_date TEXT NOT NULL,
                        delivery_key TEXT NOT NULL,
                        target TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        delivered_at TEXT,
                        last_error TEXT,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY(group_id, report_date, delivery_key)
                    );

                    CREATE INDEX IF NOT EXISTS idx_qq_report_deliveries_lookup
                    ON report_deliveries(group_id, report_date, delivery_key);
                    """
                )
                conn.commit()
            finally:
                conn.close()
            self._initialized = True

    def archive_message(
        self,
        *,
        scope_key: str,
        message_id: str,
        observed_at: datetime | str | int | float,
        user_id: str | None,
        user_name: str | None,
        text: str,
        has_media: bool = False,
        media_types: list[str] | None = None,
        segment_types: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_scope_key = normalize_group_scope_key(scope_key)
        platform, chat_id = split_group_scope_key(normalized_scope_key)
        normalized_observed_at = coerce_archive_timestamp(observed_at)
        row = {
            "scope_key": normalized_scope_key,
            "platform": platform,
            "chat_id": chat_id,
            "message_id": str(message_id or "").strip(),
            "local_date": normalized_observed_at.date().isoformat(),
            "observed_at": normalized_observed_at.isoformat(),
            "user_id": str(user_id or "").strip() or None,
            "user_name": str(user_name or "").strip() or (str(user_id or "").strip() or None),
            "text": str(text or "").strip(),
            "has_media": bool(has_media),
            "media_types": list(media_types or []),
            "segment_types": list(segment_types or []),
        }
        if not row["message_id"]:
            raise ValueError("message_id is required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO raw_messages (
                    group_id, message_id, local_date, observed_at,
                    user_id, user_name, text, has_media,
                    media_types_json, segment_types_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_scope_key,
                    row["message_id"],
                    row["local_date"],
                    row["observed_at"],
                    row["user_id"],
                    row["user_name"],
                    row["text"],
                    1 if row["has_media"] else 0,
                    json.dumps(row["media_types"], ensure_ascii=False),
                    json.dumps(row["segment_types"], ensure_ascii=False),
                ),
            )
            conn.commit()
        return row

    def _row_to_raw_message(self, row: sqlite3.Row) -> dict[str, Any]:
        scope_key = _normalize_scope_key(row["group_id"], legacy_default_platform=self.legacy_default_platform)
        if scope_key is None:
            raise ValueError("Invalid stored scope key")
        platform, chat_id = split_group_scope_key(scope_key)
        return {
            "scope_key": scope_key,
            "platform": platform,
            "chat_id": chat_id,
            "message_id": row["message_id"],
            "local_date": row["local_date"],
            "observed_at": row["observed_at"],
            "user_id": row["user_id"],
            "user_name": row["user_name"],
            "text": row["text"],
            "has_media": bool(row["has_media"]),
            "media_types": json.loads(row["media_types_json"] or "[]"),
            "segment_types": json.loads(row["segment_types_json"] or "[]"),
        }

    def list_recent_messages(
        self,
        *,
        scope_key: str | None = None,
        report_date: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if scope_key:
            aliases = _scope_aliases(scope_key, legacy_default_platform=self.legacy_default_platform)
            clauses.append(f"group_id IN ({_placeholder_in_clause(aliases)})")
            params.extend(aliases)
        if report_date:
            clauses.append("local_date = ?")
            params.append(str(report_date))

        sql = "SELECT * FROM raw_messages"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY observed_at DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_raw_message(row) for row in rows]

    def search_messages(
        self,
        *,
        query: str,
        scope_key: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        text = str(query or "").strip()
        if not text:
            raise ValueError("query is required")

        clauses = ["text LIKE ?"]
        params: list[Any] = [f"%{text}%"]
        if scope_key:
            aliases = _scope_aliases(scope_key, legacy_default_platform=self.legacy_default_platform)
            clauses.append(f"group_id IN ({_placeholder_in_clause(aliases)})")
            params.extend(aliases)
        params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(
                (
                    "SELECT * FROM raw_messages "
                    f"WHERE {' AND '.join(clauses)} "
                    "ORDER BY observed_at DESC LIMIT ?"
                ),
                params,
            ).fetchall()
        return [self._row_to_raw_message(row) for row in rows]

    def _build_report_from_rows(
        self,
        *,
        scope_key: str,
        report_date: str,
        rows: list[sqlite3.Row],
        snapshot: bool = False,
    ) -> dict[str, Any]:
        normalized_scope_key = normalize_group_scope_key(scope_key)
        platform, chat_id = split_group_scope_key(normalized_scope_key)
        first_message_at = rows[0]["observed_at"] if rows else None
        last_message_at = rows[-1]["observed_at"] if rows else None
        speaker_counter: Counter[tuple[str, str]] = Counter()
        links: list[str] = []
        seen_links: set[str] = set()
        question_highlights: list[dict[str, Any]] = []
        notable_messages: list[dict[str, Any]] = []
        media_message_count = 0

        for row in rows:
            user_id = str(row["user_id"] or "").strip()
            user_name = str(row["user_name"] or "").strip() or user_id or "unknown"
            speaker_counter[(user_id, user_name)] += 1
            text = str(row["text"] or "").strip()
            has_media = bool(row["has_media"])

            if has_media:
                media_message_count += 1

            for link in _URL_RE.findall(text):
                if link in seen_links:
                    continue
                seen_links.add(link)
                links.append(link)

            if _question_like(text) and len(question_highlights) < 8:
                question_highlights.append(
                    {
                        "user_name": user_name,
                        "text": text,
                        "observed_at": row["observed_at"],
                    }
                )

            if (
                (has_media or text.startswith("/") or "@" in text or _URL_RE.search(text) or _question_like(text))
                and len(notable_messages) < 12
            ):
                notable_messages.append(
                    {
                        "user_name": user_name,
                        "text": text,
                        "observed_at": row["observed_at"],
                        "has_media": has_media,
                    }
                )

        top_speakers = [
            {
                "user_id": user_id,
                "user_name": user_name,
                "message_count": count,
            }
            for (user_id, user_name), count in speaker_counter.most_common(8)
        ]

        summary = {
            "total_messages": len(rows),
            "unique_speakers": len(speaker_counter),
            "first_message_at": first_message_at,
            "last_message_at": last_message_at,
            "top_speakers": top_speakers,
            "media_message_count": media_message_count,
            "links": links[:12],
            "question_highlights": question_highlights,
            "notable_messages": notable_messages,
        }

        speaker_line = "、".join(
            f"{item['user_name']}({item['message_count']})" for item in top_speakers[:5]
        ) or "无"
        question_line = "；".join(
            f"{item['user_name']}：{item['text']}" for item in question_highlights[:5]
        ) or "无"
        links_line = "、".join(links[:5]) or "无"
        summary_text = "\n".join(
            [
                f"{report_date} 群会话 {chat_id} 日报",
                f"消息 {summary['total_messages']} 条，活跃成员 {summary['unique_speakers']} 人。",
                f"活跃时间：{first_message_at or '未知'} -> {last_message_at or '未知'}",
                f"高频发言：{speaker_line}",
                f"问题/请求：{question_line}",
                f"链接：{links_line}",
                f"媒体消息：{media_message_count} 条",
            ]
        )
        return {
            "scope_key": normalized_scope_key,
            "platform": platform,
            "chat_id": chat_id,
            "report_date": report_date,
            "created_at": hermes_now().isoformat(),
            "summary_text": summary_text,
            "summary": summary,
            "snapshot": bool(snapshot),
        }

    def build_snapshot_report(self, *, scope_key: str, report_date: str) -> dict[str, Any]:
        normalized_scope_key = normalize_group_scope_key(scope_key)
        normalized_report_date = str(report_date or "").strip()
        if not normalized_report_date:
            raise ValueError("report_date is required")
        aliases = _scope_aliases(normalized_scope_key, legacy_default_platform=self.legacy_default_platform)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM raw_messages
                WHERE group_id IN ({_placeholder_in_clause(aliases)}) AND local_date = ?
                ORDER BY observed_at ASC
                """,
                (*aliases, normalized_report_date),
            ).fetchall()
            if rows:
                return self._build_report_from_rows(
                    scope_key=normalized_scope_key,
                    report_date=normalized_report_date,
                    rows=list(rows),
                    snapshot=True,
                )

            report = self.get_report(scope_key=normalized_scope_key, report_date=normalized_report_date)
            if report is not None:
                report = dict(report)
                report["snapshot"] = False
                return report

        raise ValueError("No group report data found for that scope/date.")

    def rollup_daily(
        self,
        *,
        scope_key: str,
        report_date: str,
        purge_raw_after_rollup: bool = True,
    ) -> dict[str, Any]:
        normalized_scope_key = normalize_group_scope_key(scope_key)
        normalized_report_date = str(report_date or "").strip()
        if not normalized_report_date:
            raise ValueError("report_date is required")
        aliases = _scope_aliases(normalized_scope_key, legacy_default_platform=self.legacy_default_platform)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM raw_messages
                WHERE group_id IN ({_placeholder_in_clause(aliases)}) AND local_date = ?
                ORDER BY observed_at ASC
                """,
                (*aliases, normalized_report_date),
            ).fetchall()

            if not rows:
                existing = self.get_report(scope_key=normalized_scope_key, report_date=normalized_report_date)
                if existing is not None:
                    return {
                        "success": True,
                        "report": existing,
                        "purged_raw_messages": 0,
                        "existing": True,
                    }
                return {
                    "success": False,
                    "error": "No archived raw messages found for that group/date.",
                }

            report = self._build_report_from_rows(
                scope_key=normalized_scope_key,
                report_date=normalized_report_date,
                rows=list(rows),
            )
            conn.execute(
                f"DELETE FROM daily_reports WHERE group_id IN ({_placeholder_in_clause(aliases)}) AND report_date = ?",
                (*aliases, normalized_report_date),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_reports (
                    group_id, report_date, created_at, total_messages,
                    unique_speakers, first_message_at, last_message_at,
                    summary_text, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_scope_key,
                    normalized_report_date,
                    report["created_at"],
                    report["summary"]["total_messages"],
                    report["summary"]["unique_speakers"],
                    report["summary"]["first_message_at"],
                    report["summary"]["last_message_at"],
                    report["summary_text"],
                    json.dumps(report["summary"], ensure_ascii=False),
                ),
            )
            purged_raw_messages = 0
            if purge_raw_after_rollup:
                purged_raw_messages = conn.execute(
                    f"DELETE FROM raw_messages WHERE group_id IN ({_placeholder_in_clause(aliases)}) AND local_date = ?",
                    (*aliases, normalized_report_date),
                ).rowcount
            conn.commit()

        return {
            "success": True,
            "report": report,
            "purged_raw_messages": max(0, int(purged_raw_messages or 0)),
        }

    def _row_to_report(self, row: sqlite3.Row) -> dict[str, Any]:
        scope_key = _normalize_scope_key(row["group_id"], legacy_default_platform=self.legacy_default_platform)
        if scope_key is None:
            raise ValueError("Invalid stored report scope key")
        platform, chat_id = split_group_scope_key(scope_key)
        return {
            "scope_key": scope_key,
            "platform": platform,
            "chat_id": chat_id,
            "report_date": row["report_date"],
            "created_at": row["created_at"],
            "summary_text": row["summary_text"],
            "summary": json.loads(row["summary_json"] or "{}"),
        }

    def list_due_scope_dates(self, *, before_date: str) -> list[dict[str, str]]:
        normalized_before_date = str(before_date or "").strip()
        if not normalized_before_date:
            raise ValueError("before_date is required")
        with self._connect() as conn:
            candidates = conn.execute(
                """
                SELECT DISTINCT group_id, local_date
                FROM raw_messages
                WHERE local_date < ?
                ORDER BY local_date ASC, group_id ASC
                """,
                (normalized_before_date,),
            ).fetchall()
        due: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in candidates:
            scope_key = _normalize_scope_key(row["group_id"], legacy_default_platform=self.legacy_default_platform)
            if not scope_key:
                continue
            key = (scope_key, str(row["local_date"]))
            if key in seen:
                continue
            seen.add(key)
            due.append({"scope_key": scope_key, "report_date": str(row["local_date"])})
        return due

    def list_reports(self, *, scope_key: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        params: list[Any] = []
        sql = "SELECT * FROM daily_reports"
        if scope_key:
            aliases = _scope_aliases(scope_key, legacy_default_platform=self.legacy_default_platform)
            sql += f" WHERE group_id IN ({_placeholder_in_clause(aliases)})"
            params.extend(aliases)
        sql += " ORDER BY report_date DESC, group_id ASC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        reports: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            report = self._row_to_report(row)
            dedupe_key = (report["scope_key"], report["report_date"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            reports.append(report)
            if len(reports) >= max(1, int(limit)):
                break
        return reports

    def get_report(self, *, scope_key: str, report_date: str) -> dict[str, Any] | None:
        normalized_scope_key = normalize_group_scope_key(scope_key)
        normalized_report_date = str(report_date or "").strip()
        if not normalized_report_date:
            raise ValueError("report_date is required")
        aliases = _scope_aliases(normalized_scope_key, legacy_default_platform=self.legacy_default_platform)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM daily_reports
                WHERE group_id IN ({_placeholder_in_clause(aliases)}) AND report_date = ?
                ORDER BY CASE WHEN group_id = ? THEN 0 ELSE 1 END, group_id ASC
                """,
                (*aliases, normalized_report_date, normalized_scope_key),
            ).fetchall()
        if not rows:
            return None
        return self._row_to_report(rows[0])

    def get_storage_stats(self, *, before_date: str | None = None) -> dict[str, Any]:
        with self._connect() as conn:
            raw_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS raw_message_count,
                    MIN(local_date) AS oldest_raw_date,
                    MAX(local_date) AS newest_raw_date
                FROM raw_messages
                """
            ).fetchone()
            report_row = conn.execute(
                "SELECT COUNT(*) AS report_count FROM daily_reports"
            ).fetchone()
            group_rows = conn.execute("SELECT DISTINCT group_id FROM raw_messages").fetchall()

        raw_scope_keys: set[str] = set()
        for row in group_rows:
            scope_key = _normalize_scope_key(row["group_id"], legacy_default_platform=self.legacy_default_platform)
            if scope_key:
                raw_scope_keys.add(scope_key)
        due_scope_dates = self.list_due_scope_dates(before_date=before_date) if before_date else []
        return {
            "raw_message_count": int((raw_row["raw_message_count"] if raw_row else 0) or 0),
            "raw_scope_count": len(raw_scope_keys),
            "due_rollup_count": len(due_scope_dates),
            "due_scope_count": len({item["scope_key"] for item in due_scope_dates}),
            "report_count": int((report_row["report_count"] if report_row else 0) or 0),
            "oldest_raw_date": (raw_row["oldest_raw_date"] if raw_row else None),
            "newest_raw_date": (raw_row["newest_raw_date"] if raw_row else None),
        }

    def get_report_delivery(
        self,
        *,
        scope_key: str,
        report_date: str,
        delivery_key: str,
    ) -> dict[str, Any] | None:
        normalized_scope_key = normalize_group_scope_key(scope_key)
        normalized_report_date = str(report_date or "").strip()
        normalized_delivery_key = str(delivery_key or "").strip()
        if not normalized_report_date or not normalized_delivery_key:
            raise ValueError("report_date and delivery_key are required")
        aliases = _scope_aliases(normalized_scope_key, legacy_default_platform=self.legacy_default_platform)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM report_deliveries
                WHERE group_id IN ({_placeholder_in_clause(aliases)}) AND report_date = ? AND delivery_key = ?
                ORDER BY CASE WHEN group_id = ? THEN 0 ELSE 1 END, group_id ASC
                """,
                (*aliases, normalized_report_date, normalized_delivery_key, normalized_scope_key),
            ).fetchall()
        if not rows:
            return None
        row = rows[0]
        platform, chat_id = split_group_scope_key(normalized_scope_key)
        return {
            "scope_key": normalized_scope_key,
            "platform": platform,
            "chat_id": chat_id,
            "report_date": row["report_date"],
            "delivery_key": row["delivery_key"],
            "target": row["target"],
            "updated_at": row["updated_at"],
            "delivered_at": row["delivered_at"],
            "last_error": row["last_error"],
            "attempt_count": int(row["attempt_count"] or 0),
        }

    def has_successful_report_delivery(
        self,
        *,
        scope_key: str,
        report_date: str,
        delivery_key: str,
    ) -> bool:
        record = self.get_report_delivery(
            scope_key=scope_key,
            report_date=report_date,
            delivery_key=delivery_key,
        )
        return bool(record and str(record.get("delivered_at") or "").strip())

    def record_report_delivery(
        self,
        *,
        scope_key: str,
        report_date: str,
        delivery_key: str,
        target: str,
        error: str | None = None,
        attempted_at: datetime | None = None,
    ) -> dict[str, Any]:
        normalized_scope_key = normalize_group_scope_key(scope_key)
        normalized_report_date = str(report_date or "").strip()
        normalized_delivery_key = str(delivery_key or "").strip()
        normalized_target = str(target or "").strip()
        if not normalized_report_date or not normalized_delivery_key:
            raise ValueError("report_date and delivery_key are required")
        if not normalized_target:
            raise ValueError("target is required")

        aliases = _scope_aliases(normalized_scope_key, legacy_default_platform=self.legacy_default_platform)
        existing = self.get_report_delivery(
            scope_key=normalized_scope_key,
            report_date=normalized_report_date,
            delivery_key=normalized_delivery_key,
        )
        attempt_count = int(existing.get("attempt_count") or 0) + 1 if existing else 1
        attempt_time = coerce_archive_timestamp(attempted_at)
        delivered_at = attempt_time.isoformat() if not str(error or "").strip() else None
        last_error = str(error or "").strip() or None

        with self._connect() as conn:
            conn.execute(
                f"""
                DELETE FROM report_deliveries
                WHERE group_id IN ({_placeholder_in_clause(aliases)}) AND report_date = ? AND delivery_key = ?
                """,
                (*aliases, normalized_report_date, normalized_delivery_key),
            )
            conn.execute(
                """
                INSERT INTO report_deliveries (
                    group_id, report_date, delivery_key, target,
                    updated_at, delivered_at, last_error, attempt_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_scope_key,
                    normalized_report_date,
                    normalized_delivery_key,
                    normalized_target,
                    attempt_time.isoformat(),
                    delivered_at,
                    last_error,
                    attempt_count,
                ),
            )
            conn.commit()

        record = self.get_report_delivery(
            scope_key=normalized_scope_key,
            report_date=normalized_report_date,
            delivery_key=normalized_delivery_key,
        )
        if record is None:
            raise RuntimeError("failed to persist group report delivery state")
        return record
