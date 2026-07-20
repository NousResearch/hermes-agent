"""Tests for QQ group raw archive and daily rollup."""

from datetime import datetime

import pytest

from gateway.qq_group_archive import QqGroupArchiveStore
from gateway.qq_intel_assignments import hire_intel_worker
from gateway.qq_group_policies import QqGroupPolicyStore, set_group_policy

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]


def _group_payload(*, message_id, user_id, text, when, group_id=987654321, nickname="Alice", card=None):
    return {
        "post_type": "message",
        "message_type": "group",
        "message_id": message_id,
        "user_id": user_id,
        "group_id": group_id,
        "time": int(when.timestamp()),
        "raw_message": text,
        "message": [{"type": "text", "data": {"text": text}}],
        "sender": {"nickname": nickname, "card": card or nickname},
    }


@pytest.fixture
def shanghai_timezone(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "Asia/Shanghai")
    from hermes_time import reset_cache

    reset_cache()
    yield ZoneInfo("Asia/Shanghai")
    reset_cache()


def test_rollup_daily_persists_report_and_purges_raw_messages(shanghai_timezone):
    store = QqGroupArchiveStore()
    report_day = "2026-04-12"

    store.archive_payload(
        _group_payload(
            message_id=301,
            user_id=456789,
            text="今天安排啥？",
            nickname="Alice",
            card="AliceCard",
            when=datetime(2026, 4, 12, 9, 5, tzinfo=shanghai_timezone),
        )
    )
    store.archive_payload(
        _group_payload(
            message_id=302,
            user_id=111222,
            text="看下这个链接 https://example.com/spec",
            nickname="Bob",
            card="BobCard",
            when=datetime(2026, 4, 12, 10, 45, tzinfo=shanghai_timezone),
        )
    )

    result = store.rollup_daily(group_id="987654321", report_date=report_day)

    assert result["success"] is True
    assert result["report"]["group_id"] == "987654321"
    assert result["report"]["report_date"] == report_day
    assert result["report"]["summary"]["total_messages"] == 2
    assert result["report"]["summary"]["unique_speakers"] == 2
    assert result["purged_raw_messages"] == 2
    assert store.list_recent_messages(group_id="987654321", limit=10) == []

    stored = store.get_report(group_id="987654321", report_date=report_day)
    assert stored is not None
    assert stored["summary"]["top_speakers"][0]["user_name"] == "AliceCard"
    assert stored["summary"]["links"] == ["https://example.com/spec"]


def test_rollup_due_days_only_processes_completed_local_days(shanghai_timezone):
    store = QqGroupArchiveStore()
    set_group_policy("987654321", mode="collect_only", daily_report_enabled=True, updated_by="test")

    store.archive_payload(
        _group_payload(
            message_id=401,
            user_id=456789,
            text="昨天的记录",
            nickname="Alice",
            when=datetime(2026, 4, 12, 23, 55, tzinfo=shanghai_timezone),
        )
    )
    store.archive_payload(
        _group_payload(
            message_id=402,
            user_id=456789,
            text="今天还在继续",
            nickname="Alice",
            when=datetime(2026, 4, 13, 8, 5, tzinfo=shanghai_timezone),
        )
    )

    result = store.rollup_due_days(now=datetime(2026, 4, 13, 9, 0, tzinfo=shanghai_timezone))

    assert result["success"] is True
    assert result["rolled_up_count"] == 1
    assert result["purged_raw_messages"] == 1
    assert store.get_report(group_id="987654321", report_date="2026-04-12") is not None

    remaining = store.list_recent_messages(group_id="987654321", limit=10)
    assert len(remaining) == 1
    assert remaining[0]["message_id"] == "402"


def test_rollup_due_days_respects_daily_report_enabled_policy(shanghai_timezone):
    store = QqGroupArchiveStore()

    set_group_policy("987654321", mode="collect_only", daily_report_enabled=False, updated_by="test")
    set_group_policy("123456789", mode="collect_only", daily_report_enabled=True, updated_by="test")

    store.archive_payload(
        _group_payload(
            message_id=410,
            user_id=456789,
            text="这组先不生成日报",
            nickname="Alice",
            when=datetime(2026, 4, 12, 8, 0, tzinfo=shanghai_timezone),
        )
    )
    store.archive_payload(
        _group_payload(
            message_id=411,
            user_id=111222,
            text="这组可以正常日报",
            group_id=123456789,
            nickname="Bob",
            when=datetime(2026, 4, 12, 9, 0, tzinfo=shanghai_timezone),
        )
    )

    result = store.rollup_due_days(now=datetime(2026, 4, 13, 9, 0, tzinfo=shanghai_timezone))

    assert result["success"] is True
    assert result["rolled_up_count"] == 1
    assert result["purged_raw_messages"] == 1
    assert store.get_report(group_id="123456789", report_date="2026-04-12") is not None
    assert store.get_report(group_id="987654321", report_date="2026-04-12") is None

    remaining_disabled = store.list_recent_messages(group_id="987654321", limit=10)
    assert len(remaining_disabled) == 1
    assert remaining_disabled[0]["message_id"] == "410"


def test_rollup_daily_respects_purge_toggle(shanghai_timezone):
    store = QqGroupArchiveStore()
    set_group_policy(
        "987654321",
        mode="collect_only",
        daily_report_enabled=True,
        purge_raw_after_rollup=False,
        updated_by="test",
    )

    store.archive_payload(
        _group_payload(
            message_id=420,
            user_id=456789,
            text="这天先不删原始记录",
            nickname="Alice",
            when=datetime(2026, 4, 12, 11, 0, tzinfo=shanghai_timezone),
        )
    )

    result = store.rollup_daily(group_id="987654321", report_date="2026-04-12")

    assert result["success"] is True
    assert result["purged_raw_messages"] == 0
    remaining = store.list_recent_messages(group_id="987654321", limit=10)
    assert len(remaining) == 1
    assert remaining[0]["message_id"] == "420"


def test_rollup_due_days_runs_for_active_intel_worker_without_group_policy(shanghai_timezone):
    store = QqGroupArchiveStore()
    hire_intel_worker(
        worker_name="钢镚",
        target_group_ref="group:987654321",
        objective="去刺探情报",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:179033731",
        manual_report_target="qq_napcat:dm:179033731",
        notify_target="qq_napcat:dm:179033731",
        updated_by="test",
        joined_groups=[{"group_id": "987654321", "group_name": "目标群"}],
    )

    store.archive_payload(
        _group_payload(
            message_id=421,
            user_id=456789,
            text="昨天有一条线索",
            nickname="Alice",
            when=datetime(2026, 4, 12, 20, 0, tzinfo=shanghai_timezone),
        )
    )

    result = store.rollup_due_days(now=datetime(2026, 4, 13, 9, 0, tzinfo=shanghai_timezone))

    assert result["success"] is True
    assert result["rolled_up_count"] == 1
    assert store.get_report(group_id="987654321", report_date="2026-04-12") is not None


def test_report_delivery_state_tracks_failures_and_successes(shanghai_timezone):
    store = QqGroupArchiveStore()
    set_group_policy("987654321", mode="collect_only", daily_report_enabled=True, updated_by="test")
    store.archive_payload(
        _group_payload(
            message_id=430,
            user_id=456789,
            text="这条要进日报",
            nickname="Alice",
            when=datetime(2026, 4, 12, 12, 0, tzinfo=shanghai_timezone),
        )
    )
    store.rollup_daily(group_id="987654321", report_date="2026-04-12")

    failed = store.record_report_delivery(
        group_id="987654321",
        report_date="2026-04-12",
        delivery_key="policy:qq_napcat:dm:179033731",
        target="qq_napcat:dm:179033731",
        error="temporary failure",
    )

    assert failed["attempt_count"] == 1
    assert failed["delivered_at"] is None
    assert failed["last_error"] == "temporary failure"
    assert store.has_successful_report_delivery(
        group_id="987654321",
        report_date="2026-04-12",
        delivery_key="policy:qq_napcat:dm:179033731",
    ) is False

    succeeded = store.record_report_delivery(
        group_id="987654321",
        report_date="2026-04-12",
        delivery_key="policy:qq_napcat:dm:179033731",
        target="qq_napcat:dm:179033731",
        error=None,
    )

    assert succeeded["attempt_count"] == 2
    assert succeeded["delivered_at"] is not None
    assert succeeded["last_error"] is None
    assert store.has_successful_report_delivery(
        group_id="987654321",
        report_date="2026-04-12",
        delivery_key="policy:qq_napcat:dm:179033731",
    ) is True


def test_describe_group_reporting_exposes_consistent_delivery_targets():
    store = QqGroupArchiveStore()
    set_group_policy(
        "987654321",
        mode="collect_only",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:179033731",
        manual_report_target="qq_napcat:group:987654321",
        group_name="研发群",
        updated_by="test",
    )
    hire_intel_worker(
        worker_name="钢镚",
        target_group_ref="group:987654321",
        objective="去刺探情报",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:200000001",
        manual_report_target="qq_napcat:dm:200000001",
        notify_target="qq_napcat:dm:300000001",
        updated_by="test",
        joined_groups=[{"group_id": "987654321", "group_name": "研发群"}],
    )

    reporting = store.describe_group_reporting(group_id="987654321")

    assert reporting["group_name"] == "研发群"
    assert reporting["mode"] == "collect_only"
    assert reporting["monitoring_intent"] == "collect_only_monitoring"
    assert reporting["worker_names"] == ["钢镚"]
    assert reporting["purge_raw_after_rollup"] is True
    assert reporting["collect_only"] is True
    assert reporting["replies_disabled"] is True
    assert reporting["reply_behavior"] == "no_reply"
    assert reporting["active_worker_count"] == 1
    assert reporting["delivery_targets"] == {
        "daily_report_targets": [
            "qq_napcat:dm:179033731",
            "qq_napcat:dm:200000001",
        ],
        "manual_report_targets": [
            "qq_napcat:group:987654321",
            "qq_napcat:dm:200000001",
        ],
        "notify_targets": ["qq_napcat:dm:300000001"],
    }
    assert reporting["report_control"] == {
        "daily_report_enabled": True,
        "manual_report_available": True,
        "daily_report_targets": [
            "qq_napcat:dm:179033731",
            "qq_napcat:dm:200000001",
        ],
        "manual_report_targets": [
            "qq_napcat:group:987654321",
            "qq_napcat:dm:200000001",
        ],
        "notify_targets": ["qq_napcat:dm:300000001"],
        "purge_raw_after_rollup": True,
    }
    assert reporting["worker_assignments"][0]["worker_name"] == "钢镚"
    assert reporting["policy_summary"]["reply_behavior"] == "no_reply"
    assert "mode=collect_only" in reporting["summary"]


def test_policy_store_reload_reads_external_file_update():
    store = QqGroupPolicyStore()

    initial = store.get_policy("987654321")
    assert initial["mode"] == "default"

    store.path.write_text(
        """
{"version":1,"updated_at":"2026-04-13T00:00:00+08:00","groups":{"987654321":{"mode":"collect_only","archive_enabled":true,"daily_report_enabled":true,"group_name":"研发群","notes":"外部更新","updated_at":"2026-04-13T00:00:00+08:00","updated_by":"external"}}}
        """.strip(),
        encoding="utf-8",
    )

    reloaded = store.get_policy("987654321")
    assert reloaded["mode"] == "collect_only"
    assert reloaded["archive_enabled"] is True
    assert reloaded["daily_report_enabled"] is True
    assert reloaded["group_name"] == "研发群"


def test_runtime_stats_include_raw_due_and_report_counts(shanghai_timezone):
    store = QqGroupArchiveStore()
    set_group_policy("987654321", mode="collect_only", daily_report_enabled=True, updated_by="test")

    store.archive_payload(
        _group_payload(
            message_id=501,
            user_id=456789,
            text="昨天留档",
            nickname="Alice",
            when=datetime(2026, 4, 12, 18, 0, tzinfo=shanghai_timezone),
        )
    )
    store.archive_payload(
        _group_payload(
            message_id=502,
            user_id=111222,
            text="今天继续采集",
            group_id=123456789,
            nickname="Bob",
            when=datetime(2026, 4, 13, 9, 0, tzinfo=shanghai_timezone),
        )
    )
    store.rollup_daily(group_id="123456789", report_date="2026-04-13")

    stats = store.get_runtime_stats(now=datetime(2026, 4, 13, 12, 0, tzinfo=shanghai_timezone))

    assert stats["raw_message_count"] == 1
    assert stats["raw_group_count"] == 1
    assert stats["due_rollup_count"] == 1
    assert stats["report_count"] == 1
