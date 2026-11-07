"""Tests for federation Phase 10 — leader election + config sync."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.federation.federation_cluster import (
    ElectionState,
    FederationConfigSync,
    FederationLeaderElection,
)


# ========================================================================
# ElectionState tests
# ========================================================================

class TestElectionState:
    def test_roundtrip(self):
        state = ElectionState(
            election_id="el-001",
            initiator="dev-a",
            candidates={"dev-a": 15.0, "dev-b": 12.0},
            leader="dev-a",
            status="finished",
        )
        d = state.to_dict()
        restored = ElectionState.from_dict(d)
        assert restored.election_id == "el-001"
        assert restored.leader == "dev-a"
        assert restored.candidates["dev-b"] == 12.0


# ========================================================================
# FederationLeaderElection tests
# ========================================================================

class TestFederationLeaderElection:
    def _make_election(self, score=15.0):
        adapter = MagicMock()
        adapter.send = AsyncMock()
        return FederationLeaderElection(
            device_id="dev-a",
            adapter=adapter,
            compute_score=score,
            election_timeout=0.1,
        )

    def test_init(self):
        e = self._make_election()
        assert e.device_id == "dev-a"
        assert e.compute_score == 15.0
        assert not e.has_leader

    def test_initiate_election_broadcasts(self):
        e = self._make_election()
        import asyncio
        asyncio.get_event_loop().run_until_complete(e.initiate_election())
        # initiate_election sends election + (if alone) victory
        assert e.adapter.send.call_count >= 1
        msg = e.adapter.send.call_args_list[0][0][0]
        assert msg.msg_type == "election"
        assert msg.payload["election_id"] != ""

    def test_handle_election_registers_candidate(self):
        e = self._make_election()
        msg = MagicMock()
        msg.sender_id = "dev-b"
        msg.payload = {
            "election_id": "el-001",
            "initiator": "dev-b",
            "score": 20.0,
        }
        import asyncio
        asyncio.get_event_loop().run_until_complete(e.handle_election(msg))
        assert e._current_election is not None
        assert "dev-b" in e._current_election.candidates

    def test_election_highest_score_wins(self):
        e = self._make_election(score=10.0)  # Lower score
        # Simulate election with higher-scoring peer
        msg = MagicMock()
        msg.sender_id = "dev-b"
        msg.payload = {
            "election_id": "el-002",
            "initiator": "dev-b",
            "score": 25.0,
        }
        import asyncio
        asyncio.get_event_loop().run_until_complete(e.handle_election(msg))
        e._current_election.candidates["dev-b"] = 25.0
        asyncio.get_event_loop().run_until_complete(e._wait_for_election_result())

        assert e.get_leader() == "dev-b"  # Higher score wins

    def test_is_leader(self):
        e = self._make_election(score=30.0)  # High score
        e._current_leader = "dev-a"
        assert e.is_leader()

    def test_victory_sets_leader(self):
        e = self._make_election()
        msg = MagicMock()
        msg.sender_id = "dev-c"
        msg.payload = {
            "election_id": "el-003",
            "leader": "dev-c",
        }
        import asyncio
        asyncio.get_event_loop().run_until_complete(e.handle_victory(msg))
        assert e.get_leader() == "dev-c"
        assert e._missed_heartbeats == 0

    def test_coordinate_updates_heartbeat(self):
        e = self._make_election()
        e._current_leader = "dev-c"
        msg = MagicMock()
        msg.sender_id = "dev-c"
        msg.payload = {
            "leader": "dev-c",
            "timestamp": 1234567890.0,
        }
        import asyncio
        asyncio.get_event_loop().run_until_complete(e.handle_coordinate(msg))
        assert e._missed_heartbeats == 0


# ========================================================================
# FederationConfigSync tests
# ========================================================================

class TestFederationConfigSync:
    def _make_sync(self, tmp_path: Path):
        adapter = MagicMock()
        adapter.send = AsyncMock()
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: gpt-4\nprovider: openai\n")
        return FederationConfigSync(
            device_id="dev-a",
            adapter=adapter,
            config_path=config_file,
        )

    def test_init(self, tmp_path):
        s = self._make_sync(tmp_path)
        assert s.device_id == "dev-a"
        # Hash is computed on start(); after start it should be non-empty
        import asyncio
        asyncio.get_event_loop().run_until_complete(s.start())
        assert s._local_config_hash != ""

    def test_compute_hash(self, tmp_path):
        s = self._make_sync(tmp_path)
        h1 = s._compute_config_hash()
        h2 = s._compute_config_hash()
        assert h1 == h2  # Same content = same hash

    def test_sync_config_broadcasts(self, tmp_path):
        s = self._make_sync(tmp_path)
        import asyncio
        asyncio.get_event_loop().run_until_complete(s.sync_config())
        s.adapter.send.assert_called_once()
        msg = s.adapter.send.call_args[0][0]
        assert msg.msg_type == "config_sync"
        assert "config" in msg.payload

    def test_handle_config_sync_applies(self, tmp_path):
        s = self._make_sync(tmp_path)
        original_hash = s._local_config_hash

        msg = MagicMock()
        msg.sender_id = "dev-b"
        msg.payload = {
            "action": "update",
            "config_hash": "new_hash_123",
            "config": "model: claude-3\nprovider: anthropic\n",
        }
        import asyncio
        asyncio.get_event_loop().run_until_complete(s.handle_config_sync(msg))

        # Config was applied
        assert s._local_config_hash == "new_hash_123"
        config_content = s.config_path.read_text()
        assert "claude-3" in config_content

    def test_handle_config_sync_skip_same_hash(self, tmp_path):
        s = self._make_sync(tmp_path)
        original_hash = s._local_config_hash

        msg = MagicMock()
        msg.sender_id = "dev-b"
        msg.payload = {
            "action": "update",
            "config_hash": original_hash,
            "config": "model: different\n",
        }
        import asyncio
        asyncio.get_event_loop().run_until_complete(s.handle_config_sync(msg))

        # Same hash → skip: neither the hash nor the config file changes,
        # and no ack is sent. Regression: the coroutine was previously
        # never awaited, so this test passed vacuously without ever
        # exercising the same-hash skip path (pytest emitted
        # RuntimeWarning: coroutine ... was never awaited).
        assert s._local_config_hash == original_hash
        config_content = s.config_path.read_text()
        assert "gpt-4" in config_content  # Original content preserved
        assert "different" not in config_content  # Remote config NOT applied
        s.adapter.send.assert_not_called()  # Skip path sends no CONFIG_ACK

    def test_apply_remote_config_atomic(self, tmp_path):
        s = self._make_sync(tmp_path)
        s._apply_remote_config("new: config\n", "hash456")

        assert s.config_path.exists()
        assert s.config_path.read_text() == "new: config\n"
        # No temp file should remain
        assert not s.config_path.with_suffix(".yaml.tmp").exists()
