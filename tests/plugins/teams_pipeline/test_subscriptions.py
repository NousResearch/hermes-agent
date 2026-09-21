"""Tests for plugins/teams_pipeline/subscriptions.py — coverage-focused.

Covers: build_graph_client, _utc_now, utc_timestamp, sync_graph_subscription_record,
expected_client_state, is_managed_subscription, maintain_graph_subscriptions.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from os import getenv
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    """Provide a real temp file path for TeamsPipelineStore."""
    return tmp_path / "test_store.json"


@pytest.fixture
def store(store_path: Path):
    from plugins.teams_pipeline.store import TeamsPipelineStore
    return TeamsPipelineStore(store_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeGraphClient:
    """Fake MicrosoftGraphClient for tests."""

    def __init__(self, subscriptions: list[dict[str, Any]] = None):
        self.subscriptions = subscriptions or []
        self.patched: list[tuple[str, dict[str, Any]]] = []

    async def collect_paginated(self, path: str) -> list[dict[str, Any]]:
        return self.subscriptions

    async def patch_json(self, path: str, *, json_body: dict[str, Any]) -> dict[str, Any]:
        self.patched.append((path, json_body))
        return {}


def _make_subscription(
    sub_id: str = "sub-1",
    expiration_offset_hours: int = 12,
    client_state: str = "abc123",
    resource: str = "me/events",
    base: datetime | None = None,
) -> dict[str, Any]:
    now = base or datetime.now(timezone.utc)
    expiration = now + timedelta(hours=expiration_offset_hours)
    return {
        "id": sub_id,
        "subscription_id": sub_id,
        "resource": resource,
        "change_type": "created",
        "notification_url": f"https://graph.microsoft.com/v1.0/{sub_id}",
        "expirationDateTime": expiration.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clientState": client_state,
        "latestRenewalAt": None,
    }


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MSGRAPH_WEBHOOK_CLIENT_STATE", raising=False)
    monkeypatch.delenv("MSGRAPH_TENANT_ID", raising=False)
    monkeypatch.delenv("MSGRAPH_CLIENT_ID", raising=False)
    monkeypatch.delenv("MSGRAPH_CLIENT_SECRET", raising=False)


# ---------------------------------------------------------------------------
# _utc_now
# ---------------------------------------------------------------------------


class TestUtcNow:
    """Cover _utc_now."""

    def test_returns_utc_datetime(self) -> None:
        from plugins.teams_pipeline.subscriptions import _utc_now

        result = _utc_now()
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

    def test_within_recent_range(self) -> None:
        from plugins.teams_pipeline.subscriptions import _utc_now

        before = datetime.now(timezone.utc)
        result = _utc_now()
        after = datetime.now(timezone.utc)
        assert before <= result <= after


# ---------------------------------------------------------------------------
# utc_timestamp
# ---------------------------------------------------------------------------


class TestUtcTimestamp:
    """Cover utc_timestamp."""

    def test_default_now_plus_zero(self) -> None:
        from plugins.teams_pipeline.subscriptions import utc_timestamp

        ts = utc_timestamp()
        assert ts.endswith("Z")

    def test_with_hours_from_now(self) -> None:
        from plugins.teams_pipeline.subscriptions import utc_timestamp

        ts = utc_timestamp(hours_from_now=24)
        assert ts.endswith("Z")

    def test_with_base_datetime(self) -> None:
        from plugins.teams_pipeline.subscriptions import utc_timestamp

        base = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        ts = utc_timestamp(hours_from_now=2, base=base)
        expected = (base + timedelta(hours=2)).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert ts == expected


# ---------------------------------------------------------------------------
# sync_graph_subscription_record
# ---------------------------------------------------------------------------


class TestSyncGraphSubscriptionRecord:
    """Cover sync_graph_subscription_record."""

    def test_active_when_future_expiration(self, store, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.teams_pipeline.subscriptions import sync_graph_subscription_record, _utc_now

        monkeypatch.setattr("plugins.teams_pipeline.subscriptions._utc_now", lambda: datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc))
        store = store
        payload = _make_subscription(expiration_offset_hours=24)
        result = sync_graph_subscription_record(store, payload)
        assert result["status"] == "active"
        assert result["subscription_id"] == "sub-1"

    def test_expired_when_past_expiration(self, store, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.teams_pipeline.subscriptions import sync_graph_subscription_record, _utc_now

        monkeypatch.setattr("plugins.teams_pipeline.subscriptions._utc_now", lambda: datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc))
        payload = _make_subscription(expiration_offset_hours=-1, base=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc))
        result = sync_graph_subscription_record(store, payload)
        assert result["status"] == "expired"

    def test_renewed_sets_latest_renewal_at(self, store, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.teams_pipeline.subscriptions import sync_graph_subscription_record, _utc_now

        monkeypatch.setattr("plugins.teams_pipeline.subscriptions._utc_now", lambda: datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc))
        store = store
        payload = _make_subscription(expiration_offset_hours=24)
        result = sync_graph_subscription_record(store, payload, renewed=True)
        assert result["status"] == "active"
        assert "latest_renewal_at" in result
        assert result["latest_renewal_at"].endswith("Z")

    def test_explicit_status_override(self, store, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.teams_pipeline.subscriptions import sync_graph_subscription_record, _utc_now

        monkeypatch.setattr("plugins.teams_pipeline.subscriptions._utc_now", lambda: datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc))
        store = store
        payload = _make_subscription(expiration_offset_hours=24)
        result = sync_graph_subscription_record(store, payload, status="paused")
        assert result["status"] == "paused"


# ---------------------------------------------------------------------------
# expected_client_state
# ---------------------------------------------------------------------------


class TestExpectedClientState:
    """Cover expected_client_state."""

    def test_none_when_env_not_set(self) -> None:
        from plugins.teams_pipeline.subscriptions import expected_client_state

        assert expected_client_state() is None

    def test_returns_env_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.teams_pipeline.subscriptions import expected_client_state

        monkeypatch.setenv("MSGRAPH_WEBHOOK_CLIENT_STATE", "my-client-state")
        assert expected_client_state() == "my-client-state"

    def test_strips_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.teams_pipeline.subscriptions import expected_client_state

        monkeypatch.setenv("MSGRAPH_WEBHOOK_CLIENT_STATE", "  trimmed  ")
        assert expected_client_state() == "trimmed"

    def test_explicit_raw_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.teams_pipeline.subscriptions import expected_client_state

        monkeypatch.setenv("MSGRAPH_WEBHOOK_CLIENT_STATE", "env-value")
        assert expected_client_state(raw="explicit-value") == "explicit-value"

    def test_empty_string_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.teams_pipeline.subscriptions import expected_client_state

        monkeypatch.setenv("MSGRAPH_WEBHOOK_CLIENT_STATE", "")
        assert expected_client_state() is None


# ---------------------------------------------------------------------------
# is_managed_subscription
# ---------------------------------------------------------------------------


class TestIsManagedSubscription:
    """Cover is_managed_subscription."""

    def test_true_when_store_knows_it(self, store) -> None:
        from plugins.teams_pipeline.subscriptions import is_managed_subscription

        store = store
        store.upsert_subscription("known-sub", {"resource": "me/events"})
        payload = _make_subscription("known-sub")
        assert is_managed_subscription(store, payload, expected_client_state_value="other") is True

    def test_true_when_client_state_matches(self, store) -> None:
        from plugins.teams_pipeline.subscriptions import is_managed_subscription

        payload = _make_subscription(client_state="matching-state")
        assert is_managed_subscription(store, payload, expected_client_state_value="matching-state") is True

    def test_false_when_no_match(self, store) -> None:
        from plugins.teams_pipeline.subscriptions import is_managed_subscription

        payload = _make_subscription(client_state="different-state")
        assert is_managed_subscription(store, payload, expected_client_state_value="expected-state") is False

    def test_false_when_empty_client_state(self, store) -> None:
        from plugins.teams_pipeline.subscriptions import is_managed_subscription

        payload = _make_subscription(client_state="")
        assert is_managed_subscription(store, payload, expected_client_state_value="some-state") is False

    def test_handles_id_alias(self, store) -> None:
        from plugins.teams_pipeline.subscriptions import is_managed_subscription

        store.upsert_subscription("alias-sub", {"resource": "me/events"})
        payload = {"id": "alias-sub", "resource": "me/events"}
        assert is_managed_subscription(store, payload, expected_client_state_value="other") is True

    def test_uses_clientstate_field_when_id_missing(self, store) -> None:
        from plugins.teams_pipeline.subscriptions import is_managed_subscription

        payload = {
            "clientState": "cs-123",
            "resource": "me/events",
            "expirationDateTime": "2025-06-01T00:00:00Z",
        }
        assert is_managed_subscription(store, payload, expected_client_state_value="cs-123") is True


# ---------------------------------------------------------------------------
# maintain_graph_subscriptions (async) — mocked store + client
# ---------------------------------------------------------------------------


class TestMaintainGraphSubscriptions:
    """Cover maintain_graph_subscriptions with mocked store."""

    @pytest.mark.asyncio
    async def test_empty_remote_returns_empty_candidates(self) -> None:
        from plugins.teams_pipeline.subscriptions import maintain_graph_subscriptions

        client = FakeGraphClient([])
        store = MagicMock()
        store.list_subscriptions.return_value = []
        result = await maintain_graph_subscriptions(client=client, store=store, dry_run=True)
        assert result["success"] is True
        assert result["candidate_count"] == 0
        assert result["dry_run"] is True

    @pytest.mark.asyncio
    async def test_skips_unmanaged_subscriptions(self) -> None:
        from plugins.teams_pipeline.subscriptions import maintain_graph_subscriptions

        client = FakeGraphClient([_make_subscription("remote-1", expiration_offset_hours=12)])
        store = MagicMock()
        store.get_subscription.return_value = None
        store.list_subscriptions.return_value = []
        result = await maintain_graph_subscriptions(client=client, store=store, dry_run=True)
        assert result["candidate_count"] == 0
        assert any(s["reason"] == "not_managed_by_teams_pipeline" for s in result["skipped"])

    @pytest.mark.asyncio
    async def test_renews_expiring_subscription(self) -> None:
        from plugins.teams_pipeline.subscriptions import maintain_graph_subscriptions

        client = FakeGraphClient([_make_subscription("renew-me", expiration_offset_hours=1)])
        store = MagicMock()
        store.get_subscription.return_value = {"resource": "me/events"}
        store.upsert_subscription.return_value = {"subscription_id": "renew-me", "status": "active"}
        store.list_subscriptions.return_value = ["renew-me"]
        result = await maintain_graph_subscriptions(client=client, store=store, dry_run=False, renew_within_hours=24, extend_hours=24)
        assert result["candidate_count"] == 1
        assert result["renewed_count"] == 1
        assert result["dry_run"] is False

    @pytest.mark.asyncio
    async def test_dry_run_does_not_patch(self) -> None:
        from plugins.teams_pipeline.subscriptions import maintain_graph_subscriptions

        client = FakeGraphClient([_make_subscription("dry-run-sub", expiration_offset_hours=1)])
        store = MagicMock()
        store.get_subscription.return_value = {"resource": "me/events"}
        store.upsert_subscription.return_value = {"subscription_id": "dry-run-sub", "status": "active"}
        store.list_subscriptions.return_value = ["dry-run-sub"]
        result = await maintain_graph_subscriptions(client=client, store=store, dry_run=True, renew_within_hours=24, extend_hours=24)
        assert result["dry_run"] is True
        assert len(client.patched) == 0

    @pytest.mark.asyncio
    async def test_missing_expiration_skipped(self) -> None:
        from plugins.teams_pipeline.subscriptions import maintain_graph_subscriptions

        client = FakeGraphClient([{"id": "no-exp", "resource": "me/events"}])
        store = MagicMock()
        store.get_subscription.return_value = {"resource": "me/events"}
        store.upsert_subscription.return_value = {"subscription_id": "no-exp", "status": "active"}
        store.list_subscriptions.return_value = ["no-exp"]
        result = await maintain_graph_subscriptions(client=client, store=store, dry_run=True)
        assert any(s["reason"].startswith("failed_to_sync_local_store") for s in result["skipped"])

    @pytest.mark.asyncio
    async def test_not_due_skipped(self) -> None:
        from plugins.teams_pipeline.subscriptions import maintain_graph_subscriptions

        client = FakeGraphClient([_make_subscription("far-future", expiration_offset_hours=100)])
        store = MagicMock()
        store.get_subscription.return_value = {"resource": "me/events"}
        store.upsert_subscription.return_value = {"subscription_id": "far-future", "status": "active"}
        store.list_subscriptions.return_value = ["far-future"]
        result = await maintain_graph_subscriptions(client=client, store=store, dry_run=True, renew_within_hours=24)
        assert any(s["reason"] == "not_due" for s in result["skipped"])
