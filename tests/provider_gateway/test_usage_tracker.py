from __future__ import annotations

import sqlite3
import time

from provider_gateway.usage_tracker import (
    SCHEMA_VERSION,
    ProviderUsageRecord,
    ProviderUsageTracker,
)


def test_usage_tracker_default_path_uses_hermes_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    tracker = ProviderUsageTracker()

    assert tracker.db_path == tmp_path / "provider_usage.db"


def test_record_usage_persists_request_and_summary(tmp_path) -> None:
    tracker = ProviderUsageTracker(tmp_path / "provider_usage.db")

    first_id = tracker.record_usage(
        ProviderUsageRecord(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            api_mode="chat_completions",
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            estimated_cost_usd=0.0123,
            latency_ms=850,
            status="success",
            session_id="sess-1",
        )
    )
    second_id = tracker.record_usage(
        ProviderUsageRecord(
            provider="openrouter",
            model="openai/gpt-5.4",
            api_mode="chat_completions",
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
            estimated_cost_usd=0.001,
            latency_ms=300,
            status="success",
            session_id="sess-1",
        )
    )

    assert first_id == 1
    assert second_id == 2

    with sqlite3.connect(tracker.db_path) as conn:
        rows = conn.execute(
            "SELECT provider, model, total_tokens, estimated_cost_usd FROM provider_usage ORDER BY id"
        ).fetchall()

    assert rows == [
        ("openrouter", "anthropic/claude-sonnet-4.6", 140, 0.0123),
        ("openrouter", "openai/gpt-5.4", 30, 0.001),
    ]

    assert tracker.summarize_by_provider() == [
        {
            "provider": "openrouter",
            "request_count": 2,
            "success_count": 2,
            "error_count": 0,
            "total_tokens": 170,
            "estimated_cost_usd": 0.0133,
            "avg_latency_ms": 575.0,
        }
    ]


def test_record_error_counts_without_tokens(tmp_path) -> None:
    tracker = ProviderUsageTracker(tmp_path / "provider_usage.db")

    tracker.record_usage(
        ProviderUsageRecord(
            provider="anthropic",
            model="claude-opus-4.6",
            api_mode="anthropic_messages",
            status="error",
            error_type="rate_limit",
            latency_ms=1200,
        )
    )

    assert tracker.summarize_by_provider() == [
        {
            "provider": "anthropic",
            "request_count": 1,
            "success_count": 0,
            "error_count": 1,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "avg_latency_ms": 1200.0,
        }
    ]


# --- New tests below ---


def test_schema_version_is_recorded(tmp_path) -> None:
    """Schema version table exists and records the current version."""
    tracker = ProviderUsageTracker(tmp_path / "provider_usage.db")

    assert tracker.get_schema_version() == SCHEMA_VERSION


def test_schema_version_is_idempotent_on_reopen(tmp_path) -> None:
    """Re-opening the same DB does not duplicate the version row."""
    db_path = tmp_path / "provider_usage.db"

    tracker1 = ProviderUsageTracker(db_path)
    assert tracker1.get_schema_version() == SCHEMA_VERSION

    tracker2 = ProviderUsageTracker(db_path)
    assert tracker2.get_schema_version() == SCHEMA_VERSION

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM provider_usage_schema_version"
        ).fetchone()[0]
    assert count == 1


def test_wal_mode_is_enabled(tmp_path) -> None:
    """Database uses WAL journal mode for concurrent read safety."""
    tracker = ProviderUsageTracker(tmp_path / "provider_usage.db")

    with sqlite3.connect(tracker.db_path) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert journal_mode.lower() == "wal"


def test_summarize_by_provider_with_time_window(tmp_path) -> None:
    """Time-window filter allows querying usage within a specific period."""
    tracker = ProviderUsageTracker(tmp_path / "provider_usage.db")

    now = time.time()
    old_time = now - 86400  # 24 hours ago
    recent_time = now - 3600  # 1 hour ago

    # Old record
    tracker.record_usage(
        ProviderUsageRecord(
            provider="openrouter",
            model="old-model",
            api_mode="chat_completions",
            total_tokens=100,
            status="success",
            created_at=old_time,
        )
    )
    # Recent record
    tracker.record_usage(
        ProviderUsageRecord(
            provider="openrouter",
            model="new-model",
            api_mode="chat_completions",
            total_tokens=200,
            status="success",
            created_at=recent_time,
        )
    )

    # All records
    all_summary = tracker.summarize_by_provider()
    assert len(all_summary) == 1
    assert all_summary[0]["request_count"] == 2
    assert all_summary[0]["total_tokens"] == 300

    # Only recent (since 2 hours ago)
    cutoff = now - 7200
    recent_summary = tracker.summarize_by_provider(since=cutoff)
    assert len(recent_summary) == 1
    assert recent_summary[0]["request_count"] == 1
    assert recent_summary[0]["total_tokens"] == 200

    # Only old (until 2 hours ago)
    old_summary = tracker.summarize_by_provider(until=cutoff)
    assert len(old_summary) == 1
    assert old_summary[0]["request_count"] == 1
    assert old_summary[0]["total_tokens"] == 100

    # Empty window
    empty = tracker.summarize_by_provider(since=now + 1000)
    assert empty == []


def test_concurrent_writes_do_not_lose_records(tmp_path) -> None:
    """Multiple tracker instances writing to the same DB should not lose data."""
    db_path = tmp_path / "provider_usage.db"

    tracker_a = ProviderUsageTracker(db_path)
    tracker_b = ProviderUsageTracker(db_path)

    for i in range(5):
        tracker_a.record_usage(
            ProviderUsageRecord(
                provider="provider-a",
                model=f"model-{i}",
                api_mode="chat_completions",
                total_tokens=10,
                status="success",
            )
        )
        tracker_b.record_usage(
            ProviderUsageRecord(
                provider="provider-b",
                model=f"model-{i}",
                api_mode="chat_completions",
                total_tokens=20,
                status="success",
            )
        )

    summary = tracker_a.summarize_by_provider()
    providers = {row["provider"]: row for row in summary}

    assert providers["provider-a"]["request_count"] == 5
    assert providers["provider-a"]["total_tokens"] == 50
    assert providers["provider-b"]["request_count"] == 5
    assert providers["provider-b"]["total_tokens"] == 100
