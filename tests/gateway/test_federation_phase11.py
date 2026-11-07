"""Tests for federation Phase 11 — Gateway API."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gateway.federation.federation_api import (
    FederationAPI,
    FederationAPIConfig,
    FederationStatus,
    PeerStatus,
    TaskStatus,
)


# ========================================================================
# Dataclass tests
# ========================================================================

class TestPeerStatus:
    def test_to_dict(self):
        peer = PeerStatus(
            device_id="dev-a",
            hostname="macbook-pro.local",
            status="online",
            last_seen=1234567890.0,
            latency_ms=12.5,
            compute_score=15.0,
            cpu_cores=10,
            memory_gb=32.0,
            is_leader=True,
        )
        d = peer.to_dict()
        assert d["device_id"] == "dev-a"
        assert d["status"] == "online"
        assert d["latency_ms"] == 12.5
        assert d["is_leader"] is True


class TestTaskStatus:
    def test_to_dict(self):
        task = TaskStatus(
            task_id="task-001",
            source_device="dev-a",
            target_device="dev-b",
            status="completed",
            created_at=1234567890.0,
            completed_at=1234567900.0,
            result="Success",
        )
        d = task.to_dict()
        assert d["task_id"] == "task-001"
        assert d["status"] == "completed"
        assert d["duration_sec"] == 10.0


class TestFederationStatus:
    def test_to_dict(self):
        status = FederationStatus(
            device_count=5,
            online_count=3,
            offline_count=2,
            leader="dev-a",
            mode="auto",
        )
        d = status.to_dict()
        assert d["device_count"] == 5
        assert d["online_count"] == 3
        assert d["tasks"]["total"] == 0


# ========================================================================
# FederationAPI tests
# ========================================================================

class TestFederationAPI:
    def _make_api(self):
        adapter = MagicMock()
        adapter._peers = {
            "dev-a": MagicMock(status="online", last_seen=1234567890.0),
            "dev-b": MagicMock(status="offline", last_seen=1234567800.0),
        }
        adapter.get_leader = MagicMock(return_value="dev-a")
        adapter._mode = "auto"
        adapter._relay = MagicMock()
        adapter._relay._tasks = {
            "task-001": {"status": "completed", "source_device": "dev-a", "target_device": "dev-b"},
            "task-002": {"status": "pending", "source_device": "dev-b", "target_device": ""},
        }
        config = FederationAPIConfig(enabled=False)  # Don't actually start server
        return FederationAPI(adapter=adapter, config=config, hermes_version="1.0.0")

    def test_init(self):
        api = self._make_api()
        assert api.config.enabled is False
        assert api.hermes_version == "1.0.0"

    def test_build_status(self):
        api = self._make_api()
        status = api._build_status()
        assert status.device_count == 2
        assert status.online_count == 1
        assert status.offline_count == 1
        assert status.leader == "dev-a"
        assert status.mode == "auto"

    def test_get_peers(self):
        api = self._make_api()
        peers = api.get_peers()
        assert len(peers) == 2
        assert any(p["device_id"] == "dev-a" for p in peers)

    def test_get_tasks(self):
        api = self._make_api()
        tasks = api.get_tasks()
        assert len(tasks) == 2
        assert any(t["task_id"] == "task-001" for t in tasks)

    def test_get_metrics(self):
        api = self._make_api()
        metrics = api.get_metrics()
        assert "hermes_federation_devices_total 2" in metrics
        assert "hermes_federation_devices_online 1" in metrics
        assert "hermes_federation_devices_offline 1" in metrics
        assert "hermes_federation_uptime_seconds" in metrics

    def test_disabled_api_does_not_start(self):
        api = self._make_api()
        import asyncio
        asyncio.get_event_loop().run_until_complete(api.start())
        assert api._server is None  # Not started because enabled=False
