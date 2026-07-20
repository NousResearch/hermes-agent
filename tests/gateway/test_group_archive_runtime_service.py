"""See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
import pytest


from gateway.group_archive_runtime_service import build_group_archive_runtime_summary
from gateway.group_runtime_platform_specs import GroupArchiveRuntimePlatformSpec


def test_build_group_archive_runtime_summary_merges_qq_and_weixin_stats():
    summary = build_group_archive_runtime_summary(
        load_qq_archive_stats_fn=lambda: {
            "raw_message_count": 42,
            "raw_group_count": 2,
            "due_rollup_count": 1,
            "report_count": 5,
            "oldest_raw_date": "2026-04-17",
            "newest_raw_date": "2026-04-18",
        },
        load_weixin_archive_stats_fn=lambda: {
            "raw_message_count": 8,
            "raw_scope_count": 1,
            "due_rollup_count": 2,
            "due_scope_count": 1,
            "report_count": 3,
            "oldest_raw_date": "2026-04-16",
            "newest_raw_date": "2026-04-19",
        },
    )

    assert summary["raw_message_count"] == 50
    assert summary["raw_scope_count"] == 3
    assert summary["due_rollup_count"] == 3
    assert summary["due_scope_count"] == 2
    assert summary["report_count"] == 8
    assert summary["oldest_raw_date"] == "2026-04-16"
    assert summary["newest_raw_date"] == "2026-04-19"
    assert summary["platforms"]["qq_napcat"]["raw_message_count"] == 42
    assert summary["platforms"]["weixin"]["raw_message_count"] == 8


def test_build_group_archive_runtime_summary_accepts_platform_specs():
    summary = build_group_archive_runtime_summary(
        platform_specs=[
            GroupArchiveRuntimePlatformSpec(
                platform="qq_napcat",
                load_runtime_stats=lambda: {
                    "raw_message_count": 42,
                    "raw_scope_count": 2,
                    "due_rollup_count": 1,
                    "report_count": 5,
                    "oldest_raw_date": "2026-04-17",
                    "newest_raw_date": "2026-04-18",
                },
            ),
            GroupArchiveRuntimePlatformSpec(
                platform="weixin",
                load_runtime_stats=lambda: {
                    "raw_message_count": 8,
                    "raw_scope_count": 1,
                    "due_rollup_count": 2,
                    "due_scope_count": 1,
                    "report_count": 3,
                    "oldest_raw_date": "2026-04-16",
                    "newest_raw_date": "2026-04-19",
                },
            ),
        ]
    )

    assert summary["raw_message_count"] == 50
    assert summary["raw_scope_count"] == 3
    assert summary["due_rollup_count"] == 3
    assert summary["due_scope_count"] == 2
    assert summary["platforms"]["weixin"]["report_count"] == 3
