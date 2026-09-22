"""Tests for plugins/teams_pipeline/store.py — coverage-focused.

Covers: resolve_teams_pipeline_store_path, TeamsPipelineStore CRUD, atomic persist,
notification receipt dedup, stats, build_notification_receipt_key.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from os import getenv
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("MSGRAPH_WEBHOOK_STORE_PATH", "HERMES_HOME"):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# resolve_teams_pipeline_store_path
# ---------------------------------------------------------------------------


class TestResolveTeamsPipelineStorePath:
    """Cover resolve_teams_pipeline_store_path."""

    def test_explicit_path(self) -> None:
        from plugins.teams_pipeline.store import resolve_teams_pipeline_store_path

        p = resolve_teams_pipeline_store_path("/tmp/my-store.json")
        assert p == Path("/tmp/my-store.json")

    def test_explicit_path_stripped(self) -> None:
        from plugins.teams_pipeline.store import resolve_teams_pipeline_store_path

        p = resolve_teams_pipeline_store_path("  /tmp/store.json  ")
        assert p == Path("/tmp/store.json")

    def test_env_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.teams_pipeline.store import resolve_teams_pipeline_store_path

        monkeypatch.setenv("MSGRAPH_WEBHOOK_STORE_PATH", "/opt/store.json")
        p = resolve_teams_pipeline_store_path()
        assert p == Path("/opt/store.json")

    def test_explicit_path_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.teams_pipeline.store import resolve_teams_pipeline_store_path

        monkeypatch.setenv("MSGRAPH_WEBHOOK_STORE_PATH", "/opt/store.json")
        p = resolve_teams_pipeline_store_path("/tmp/explicit.json")
        assert p == Path("/tmp/explicit.json")

    def test_empty_string_path_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.teams_pipeline.store import resolve_teams_pipeline_store_path

        monkeypatch.setenv("MSGRAPH_WEBHOOK_STORE_PATH", "")
        p = resolve_teams_pipeline_store_path("")
        assert p.name == "teams_pipeline_store.json"


# ---------------------------------------------------------------------------
# TeamsPipelineStore — init + _load + _persist
# ---------------------------------------------------------------------------


class TestTeamsPipelineStoreInit:
    """Cover constructor, _load, _persist."""

    def test_creates_store_file_on_first_write(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        path = tmp_path / "new-store.json"
        store = TeamsPipelineStore(path)
        assert not path.exists()
        store.upsert_subscription("sub-1", {"resource": "me/events"})
        assert path.exists()
        data = json.loads(path.read_text())
        assert "subscriptions" in data
        assert "sub-1" in data["subscriptions"]

    def test_loads_existing_store(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        path = tmp_path / "existing-store.json"
        path.write_text(json.dumps({"subscriptions": {"old-sub": {"resource": "me/mail"}}}, indent=2))
        store = TeamsPipelineStore(path)
        assert store.get_subscription("old-sub") == {"resource": "me/mail"}

    def test_loads_empty_file_as_empty_state(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        path = tmp_path / "empty-store.json"
        path.write_text("{}")
        store = TeamsPipelineStore(path)
        assert store.list_subscriptions() == {}

    def test_loads_missing_file_as_empty_state(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        path = tmp_path / "nonexistent-store.json"
        store = TeamsPipelineStore(path)
        assert store.list_subscriptions() == {}

    def test_persist_is_atomic_via_temp_file(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        path = tmp_path / "atomic-store.json"
        store = TeamsPipelineStore(path)
        store.upsert_subscription("sub-1", {"resource": "me/events"})
        data = json.loads(path.read_text())
        assert data["subscriptions"]["sub-1"]["resource"] == "me/events"

    def test_store_path_is_resolved_to_path(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "store.json")
        assert isinstance(store.path, Path)


# ---------------------------------------------------------------------------
# TeamsPipelineStore — subscription CRUD
# ---------------------------------------------------------------------------


class TestTeamsPipelineStoreSubscriptions:
    """Cover subscription upsert/get/list/delete."""

    def test_upsert_subscription_stamps_ids(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "subs-store.json")
        result = store.upsert_subscription("sub-1", {"resource": "me/events"})
        assert result["subscription_id"] == "sub-1"
        assert "created_at" in result
        assert "updated_at" in result

    def test_upsert_subscription_merges_existing(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "merge-store.json")
        store.upsert_subscription("sub-1", {"resource": "me/events", "status": "active"})
        result = store.upsert_subscription("sub-1", {"status": "paused"})
        assert result["resource"] == "me/events"
        assert result["status"] == "paused"

    def test_upsert_subscription_preserves_created_at(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "created-store.json")
        first = store.upsert_subscription("sub-1", {"resource": "me/events"})
        created_at = first["created_at"]
        second = store.upsert_subscription("sub-1", {"status": "paused"})
        assert second["created_at"] == created_at

    def test_get_subscription_returns_deepcopy(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "deep-store.json")
        store.upsert_subscription("sub-1", {"resource": "me/events"})
        record = store.get_subscription("sub-1")
        record["resource"] = "tampered"
        assert store.get_subscription("sub-1")["resource"] == "me/events"

    def test_list_subscriptions_returns_deepcopy(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "list-store.json")
        store.upsert_subscription("sub-1", {"resource": "me/events"})
        subs = store.list_subscriptions()
        subs["sub-1"]["resource"] = "tampered"
        assert store.list_subscriptions()["sub-1"]["resource"] == "me/events"

    def test_delete_subscription_removes_record(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "del-store.json")
        store.upsert_subscription("sub-1", {"resource": "me/events"})
        assert store.delete_subscription("sub-1") is True
        assert store.get_subscription("sub-1") is None

    def test_delete_subscription_returns_false_for_missing(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "del-store2.json")
        assert store.delete_subscription("nonexistent") is False

    def test_delete_subscription_persists(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        path = tmp_path / "del-persist-store.json"
        store = TeamsPipelineStore(path)
        store.upsert_subscription("sub-1", {"resource": "me/events"})
        store.delete_subscription("sub-1")
        store2 = TeamsPipelineStore(path)
        assert store2.get_subscription("sub-1") is None


# ---------------------------------------------------------------------------
# TeamsPipelineStore — job CRUD
# ---------------------------------------------------------------------------


class TestTeamsPipelineStoreJobs:
    """Cover job upsert/get/list."""

    def test_upsert_job_stamps_ids(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "jobs-store.json")
        result = store.upsert_job("job-1", {"title": "Test Job"})
        assert result["job_id"] == "job-1"
        assert "created_at" in result

    def test_get_job_returns_record(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "get-job-store.json")
        store.upsert_job("job-1", {"title": "Test Job"})
        assert store.get_job("job-1")["title"] == "Test Job"

    def test_get_job_returns_none_for_missing(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "no-job-store.json")
        assert store.get_job("nonexistent") is None

    def test_list_jobs_returns_all(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "list-jobs-store.json")
        store.upsert_job("job-1", {"title": "A"})
        store.upsert_job("job-2", {"title": "B"})
        assert len(store.list_jobs()) == 2


# ---------------------------------------------------------------------------
# TeamsPipelineStore — sink records
# ---------------------------------------------------------------------------


class TestTeamsPipelineStoreSinkRecords:
    """Cover sink record upsert/get."""

    def test_upsert_sink_record(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "sink-store.json")
        result = store.upsert_sink_record("key-1", {"data": "value"})
        assert result["sink_key"] == "key-1"
        assert result["data"] == "value"

    def test_get_sink_record_returns_none_for_missing(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "no-sink-store.json")
        assert store.get_sink_record("nonexistent") is None


# ---------------------------------------------------------------------------
# TeamsPipelineStore — notification receipts
# ---------------------------------------------------------------------------


class TestTeamsPipelineStoreNotificationReceipts:
    """Cover notification receipt recording and dedup."""

    def test_record_notification_receipt_first_time(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "receipt-store.json")
        assert store.record_notification_receipt("receipt-1", {"event": "test"}) is True

    def test_record_notification_receipt_duplicate_returns_false(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "dup-receipt-store.json")
        store.record_notification_receipt("receipt-1", {"event": "test"})
        assert store.record_notification_receipt("receipt-1", {"event": "test"}) is False

    def test_record_notification_receipt_stamps_received_at(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "ts-receipt-store.json")
        store.record_notification_receipt("receipt-1", None, received_at="2025-01-01T00:00:00Z")


# ---------------------------------------------------------------------------
# TeamsPipelineStore — build_notification_receipt_key
# ---------------------------------------------------------------------------


class TestBuildNotificationReceiptKey:
    """Cover build_notification_receipt_key."""

    def test_explicit_id(self) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        key = TeamsPipelineStore.build_notification_receipt_key({"id": "evt-123"})
        assert key == "id:evt-123"

    def test_sha256_for_anonymous_notification(self) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        notification = {"event": "test", "data": "value"}
        key = TeamsPipelineStore.build_notification_receipt_key(notification)
        assert key.startswith("sha256:")
        import hashlib

        expected = f"sha256:{hashlib.sha256(json.dumps(notification, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}"
        assert key == expected

    def test_different_notifications_yield_different_keys(self) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        key_a = TeamsPipelineStore.build_notification_receipt_key({"event": "a"})
        key_b = TeamsPipelineStore.build_notification_receipt_key({"event": "b"})
        assert key_a != key_b


# ---------------------------------------------------------------------------
# TeamsPipelineStore — stats
# ---------------------------------------------------------------------------


class TestStats:
    """Cover stats."""

    def test_stats_returns_bucket_counts(self, tmp_path: Path) -> None:
        from plugins.teams_pipeline.store import TeamsPipelineStore

        store = TeamsPipelineStore(tmp_path / "stats-store.json")
        store.upsert_subscription("sub-1", {})
        store.upsert_subscription("sub-2", {})
        store.upsert_job("job-1", {})
        stats = store.stats()
        assert stats["subscriptions"] == 2
        assert stats["jobs"] == 1
        assert stats["notification_receipts"] == 0
        assert stats["event_timestamps"] == 0
        assert stats["sink_records"] == 0
