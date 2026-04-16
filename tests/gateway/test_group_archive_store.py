"""Tests for platform-neutral group archive storage."""

from datetime import datetime
import sqlite3

from gateway.group_archive_store import GroupArchiveStore, coerce_archive_timestamp

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]


def test_group_archive_store_round_trips_scope_key(tmp_path):
    store = GroupArchiveStore(db_path=tmp_path / "group_archive.db")
    observed_at = datetime(2026, 4, 16, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))

    row = store.archive_message(
        scope_key="weixin:group@chatroom",
        message_id="m-001",
        observed_at=observed_at,
        user_id="u-1",
        user_name="Alice",
        text="今天安排啥？",
        segment_types=["text"],
        media_types=[],
    )

    assert row["scope_key"] == "weixin:group@chatroom"
    assert row["platform"] == "weixin"
    assert row["chat_id"] == "group@chatroom"
    recent = store.list_recent_messages(scope_key="weixin:group@chatroom", limit=5)
    assert len(recent) == 1
    assert recent[0]["message_id"] == "m-001"
    assert recent[0]["text"] == "今天安排啥？"


def test_coerce_archive_timestamp_parses_iso_string():
    observed_at = coerce_archive_timestamp("2026-04-16T10:30:00+08:00")

    assert observed_at.isoformat().startswith("2026-04-16T10:30:00+08:00")


def test_group_archive_store_rollup_daily_persists_report_and_purges_raw(tmp_path):
    store = GroupArchiveStore(db_path=tmp_path / "group_archive.db")
    observed_at = datetime(2026, 4, 16, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))

    store.archive_message(
        scope_key="qq_napcat:987654321",
        message_id="m-101",
        observed_at=observed_at,
        user_id="u-1",
        user_name="Alice",
        text="看下这个链接 https://example.com/spec",
        segment_types=["text"],
        media_types=[],
    )

    result = store.rollup_daily(
        scope_key="qq_napcat:987654321",
        report_date="2026-04-16",
        purge_raw_after_rollup=True,
    )

    assert result["success"] is True
    assert result["report"]["scope_key"] == "qq_napcat:987654321"
    assert result["report"]["summary"]["total_messages"] == 1
    assert result["purged_raw_messages"] == 1
    assert store.list_recent_messages(scope_key="qq_napcat:987654321", limit=5) == []
    report = store.get_report(scope_key="qq_napcat:987654321", report_date="2026-04-16")
    assert report is not None
    assert report["summary"]["links"] == ["https://example.com/spec"]


def test_group_archive_store_reads_legacy_qq_bare_group_rows(tmp_path):
    db_path = tmp_path / "group_archive.db"
    store = GroupArchiveStore(db_path=db_path)
    store._ensure_schema()

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO raw_messages (
                group_id, message_id, local_date, observed_at, user_id, user_name, text,
                has_media, media_types_json, segment_types_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "987654321",
                "legacy-1",
                "2026-04-16",
                "2026-04-16T08:00:00+08:00",
                "u-2",
                "Bob",
                "旧数据",
                0,
                "[]",
                '["text"]',
            ),
        )
        conn.commit()

    recent = store.list_recent_messages(scope_key="qq_napcat:987654321", limit=5)

    assert len(recent) == 1
    assert recent[0]["scope_key"] == "qq_napcat:987654321"
    assert recent[0]["platform"] == "qq_napcat"
    assert recent[0]["chat_id"] == "987654321"
    assert recent[0]["message_id"] == "legacy-1"


def test_group_archive_store_reads_legacy_qq_bare_report_rows(tmp_path):
    db_path = tmp_path / "group_archive.db"
    store = GroupArchiveStore(db_path=db_path)
    store._ensure_schema()

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO daily_reports (
                group_id, report_date, created_at, total_messages, unique_speakers,
                first_message_at, last_message_at, summary_text, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "987654321",
                "2026-04-16",
                "2026-04-16T12:00:00+08:00",
                2,
                1,
                "2026-04-16T08:00:00+08:00",
                "2026-04-16T09:00:00+08:00",
                "legacy report",
                '{"total_messages":2,"unique_speakers":1}',
            ),
        )
        conn.commit()

    report = store.get_report(scope_key="qq_napcat:987654321", report_date="2026-04-16")

    assert report is not None
    assert report["scope_key"] == "qq_napcat:987654321"
    assert report["platform"] == "qq_napcat"
    assert report["chat_id"] == "987654321"
    assert report["summary"]["total_messages"] == 2


def test_group_archive_store_reads_and_updates_legacy_qq_bare_delivery_rows(tmp_path):
    db_path = tmp_path / "group_archive.db"
    store = GroupArchiveStore(db_path=db_path)
    store._ensure_schema()

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO report_deliveries (
                group_id, report_date, delivery_key, target, updated_at, delivered_at, last_error, attempt_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "987654321",
                "2026-04-16",
                "manual:qq_napcat:dm:179033731",
                "qq_napcat:dm:179033731",
                "2026-04-16T12:00:00+08:00",
                None,
                "temporary failure",
                2,
            ),
        )
        conn.commit()

    existing = store.get_report_delivery(
        scope_key="qq_napcat:987654321",
        report_date="2026-04-16",
        delivery_key="manual:qq_napcat:dm:179033731",
    )
    assert existing is not None
    assert existing["attempt_count"] == 2

    updated = store.record_report_delivery(
        scope_key="qq_napcat:987654321",
        report_date="2026-04-16",
        delivery_key="manual:qq_napcat:dm:179033731",
        target="qq_napcat:dm:179033731",
        error=None,
    )

    assert updated["scope_key"] == "qq_napcat:987654321"
    assert updated["attempt_count"] == 3
    assert updated["delivered_at"] is not None
