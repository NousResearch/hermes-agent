"""Tests for federation v2 — protocol, adapter, and heartbeat loop.

Covers:
- Protocol: message serialization, signing, verification
- Connection: peer registration, state queries
- Adapter: task lifecycle, peer management
- Config: parsing, defaults
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.federation.federation_protocol import (
    FedMessage,
    MessageType,
    PeerInfo,
)
from gateway.config import FederationConfig


# ========================================================================
# Protocol tests
# ========================================================================


class TestFedMessage:
    """Regression: message serialization and integrity."""

    def test_roundtrip(self):
        msg = FedMessage(
            msg_type=MessageType.PEER_JOIN.value,
            sender_id="device-a",
            payload={"hello": "world"},
        )
        raw = msg.to_json()
        restored = FedMessage.from_json(raw)
        assert restored.msg_id == msg.msg_id
        assert restored.msg_type == msg.msg_type
        assert restored.sender_id == "device-a"
        assert restored.payload == {"hello": "world"}

    def test_sign_and_verify(self):
        msg = FedMessage(
            msg_type=MessageType.TASK_SUBMIT.value,
            sender_id="device-a",
            payload={"task_id": "T-001"},
        )
        msg.sign("secret-token")
        assert msg.signature
        assert msg.verify("secret-token")
        assert not msg.verify("wrong-token")

    def test_no_signature_when_token_empty(self):
        msg = FedMessage(msg_type=MessageType.PEER_PING.value)
        assert msg.signature == ""

    def test_expired_message(self):
        msg = FedMessage(
            msg_type=MessageType.PEER_PING.value,
            timestamp=time.time() - 600,
        )
        assert msg.is_expired(ttl=300)
        assert not msg.is_expired(ttl=900)

    def test_factory_methods(self):
        join = FedMessage.peer_join("dev-a", PeerInfo(device_id="dev-a", hostname="host1"))
        assert join.msg_type == MessageType.PEER_JOIN.value
        assert join.payload["peer_info"]["device_id"] == "dev-a"

        leave = FedMessage.peer_leave("dev-a", reason="shutdown")
        assert leave.msg_type == MessageType.PEER_LEAVE.value
        assert leave.payload["reason"] == "shutdown"

        submit = FedMessage.task_submit("dev-a", "T-001", "Test task", priority=1)
        assert submit.msg_type == MessageType.TASK_SUBMIT.value
        assert submit.payload["task_id"] == "T-001"
        assert submit.payload["priority"] == 1

        claim = FedMessage.task_claim("dev-b", "T-001")
        assert claim.msg_type == MessageType.TASK_CLAIM.value

        progress = FedMessage.task_progress("dev-b", "T-001", 0.5, "half done")
        assert progress.payload["progress"] == 0.5

        result = FedMessage.task_result("dev-b", "T-001", True, {"count": 42})
        assert result.payload["success"] is True

        heartbeat = FedMessage.task_heartbeat("dev-b", "T-001")
        assert heartbeat.msg_type == MessageType.TASK_HEARTBEAT.value


class TestPeerInfo:
    def test_compute_score_idle(self):
        info = PeerInfo(
            device_id="dev-a", hostname="host1",
            cpu_cores=8, memory_gb=16, load_avg=0.0,
        )
        assert info.compute_score > 0

    def test_compute_score_busy(self):
        idle = PeerInfo(
            device_id="dev-a", hostname="host1",
            cpu_cores=8, memory_gb=16, load_avg=0.0,
        )
        busy = PeerInfo(
            device_id="dev-b", hostname="host2",
            cpu_cores=8, memory_gb=16, load_avg=10.0,
            current_task_id="T-999",
        )
        # Busy device should have lower score
        assert busy.compute_score < idle.compute_score


class TestFederationConfig:
    def test_defaults(self):
        cfg = FederationConfig()
        assert cfg.enabled is False
        assert cfg.mode == "shared_db"
        assert cfg.ws_port == 18765
        assert cfg.peers == []

    def test_from_dict_lan_mode(self):
        data = {
            "enabled": True,
            "mode": "lan",
            "device_id": "my-device",
            "ws_port": 19000,
            "auth_token": "secret",
            "peers": ["ws://192.168.1.10:18765"],
        }
        cfg = FederationConfig.from_dict(data)
        assert cfg.enabled is True
        assert cfg.mode == "lan"
        assert cfg.device_id == "my-device"
        assert cfg.ws_port == 19000
        assert cfg.auth_token == "secret"
        assert cfg.peers == ["ws://192.168.1.10:18765"]

    def test_from_dict_invalid_peers_type(self):
        """Non-list peers should be normalized to empty list."""
        cfg = FederationConfig.from_dict({"peers": "not-a-list"})
        assert cfg.peers == []


# ========================================================================
# Adapter tests
# ========================================================================


class TestFederationAdapter:
    """Adapter lifecycle and task management."""

    def _make_adapter(self, mode="lan"):
        from gateway.federation.federation_adapter import FederationAdapter
        cfg = FederationConfig(
            enabled=True,
            mode=mode,
            device_id="test-device",
            ws_port=18765,
        )
        return FederationAdapter(cfg)

    def test_create_adapter(self):
        adapter = self._make_adapter()
        assert adapter.device_id == "test-device"
        assert adapter.config.mode == "lan"
        assert not adapter.is_connected()

    def test_shared_db_mode_no_connection(self):
        adapter = self._make_adapter(mode="shared_db")
        assert adapter.config.mode == "shared_db"
        assert adapter.get_peer_count() == 0

    def test_task_state_tracking(self):
        adapter = self._make_adapter()
        # Simulate receiving a task submit message
        msg = FedMessage.task_submit(
            "peer-a", "T-001", "Test task", "Do something", priority=1,
        )
        adapter._on_message(msg)

        state = adapter.get_task_state("T-001")
        assert state is not None
        assert state["status"] == "pending"
        assert state["title"] == "Test task"
        assert state["submitted_by"] == "peer-a"

    def test_task_claim_updates_state(self):
        adapter = self._make_adapter()
        # First submit
        adapter._on_message(
            FedMessage.task_submit("peer-a", "T-001", "Test task")
        )
        # Then claim
        adapter._on_message(
            FedMessage.task_claim("peer-b", "T-001")
        )

        state = adapter.get_task_state("T-001")
        assert state["status"] == "claimed"
        assert state["claimed_by"] == "peer-b"

    def test_task_progress_tracking(self):
        adapter = self._make_adapter()
        adapter._on_message(
            FedMessage.task_submit("peer-a", "T-001", "Test task")
        )
        adapter._on_message(
            FedMessage.task_progress("peer-b", "T-001", 0.75, "Almost done")
        )

        state = adapter.get_task_state("T-001")
        assert state["progress"] == 0.75
        assert state["progress_note"] == "Almost done"

    def test_task_result_tracking(self):
        adapter = self._make_adapter()
        adapter._on_message(
            FedMessage.task_submit("peer-a", "T-001", "Test task")
        )
        adapter._on_message(
            FedMessage.task_result("peer-b", "T-001", True, {"output": "done"})
        )

        state = adapter.get_task_state("T-001")
        assert state["status"] == "completed"
        assert state["result_data"]["output"] == "done"

    def test_task_failure_tracking(self):
        adapter = self._make_adapter()
        adapter._on_message(
            FedMessage.task_submit("peer-a", "T-001", "Test task")
        )
        adapter._on_message(
            FedMessage.task_result("peer-b", "T-001", False, error_info="Timeout")
        )

        state = adapter.get_task_state("T-001")
        assert state["status"] == "failed"
        assert state["error_info"] == "Timeout"

    def test_task_heartbeat_tracking(self):
        adapter = self._make_adapter()
        adapter._on_message(
            FedMessage.task_submit("peer-a", "T-001", "Test task")
        )
        adapter._on_message(
            FedMessage.task_claim("peer-b", "T-001")
        )
        adapter._on_message(
            FedMessage.task_heartbeat("peer-b", "T-001")
        )

        state = adapter.get_task_state("T-001")
        assert "executor_heartbeat_at" in state

    def test_all_task_states(self):
        adapter = self._make_adapter()
        adapter._on_message(
            FedMessage.task_submit("peer-a", "T-001", "Task 1")
        )
        adapter._on_message(
            FedMessage.task_submit("peer-a", "T-002", "Task 2")
        )

        states = adapter.get_all_task_states()
        assert len(states) == 2
        assert "T-001" in states
        assert "T-002" in states


# ========================================================================
# Heartbeat loop tests (sync parts)
# ========================================================================


class TestHeartbeatLoop:
    """Regression: heartbeat loop initialization and mode dispatch."""

    def test_disabled_returns_immediately(self):
        from gateway.federation.federation_heartbeat import federation_heartbeat_loop

        cfg = FederationConfig(enabled=False)
        # Should return without error
        import asyncio
        async def run():
            await federation_heartbeat_loop(cfg)
        asyncio.get_event_loop().run_until_complete(run())
