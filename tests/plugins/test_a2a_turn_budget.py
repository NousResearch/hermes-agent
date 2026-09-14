"""Tests for turn budget surfacing, transport failure refunds, and per-peer config."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent to path so we can import the a2a plugin
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugins.platforms.a2a import protocol, adapter


class TestTurnBudgetMetadata:
    """Test that turn budget is surfaced in A2A responses."""

    def test_build_task_with_turn_budget(self):
        """build_task includes metadata.turnBudget when turn/max_turns provided."""
        task = protocol.build_task(
            "task-1", "ctx-1", protocol.STATE_COMPLETED,
            "Hello", turn=2, max_turns=5,
        )
        assert "metadata" in task
        assert "turnBudget" in task["metadata"]
        budget = task["metadata"]["turnBudget"]
        assert budget["current"] == 2
        assert budget["max"] == 5
        assert budget["remaining"] == 3

    def test_build_task_without_turn_budget(self):
        """build_task omits metadata when turn/max_turns not provided."""
        task = protocol.build_task(
            "task-1", "ctx-1", protocol.STATE_COMPLETED, "Hello",
        )
        assert "metadata" not in task

    def test_build_task_remaining_floor_zero(self):
        """remaining is floored at 0 when turn exceeds max."""
        task = protocol.build_task(
            "task-1", "ctx-1", protocol.STATE_COMPLETED,
            "Hello", turn=10, max_turns=5,
        )
        assert task["metadata"]["turnBudget"]["remaining"] == 0


class TestTransportFailureRefund:
    """Test that transport failures refund turns."""

    def test_refund_decrements_count(self):
        """refund() decrements the turn count."""
        tracker = protocol.TurnTracker()
        tracker.track("ctx-1")
        tracker.track("ctx-1")
        assert tracker.get_count("ctx-1") == 2
        tracker.refund("ctx-1")
        assert tracker.get_count("ctx-1") == 1

    def test_refund_floor_zero(self):
        """refund() floors at 0, never negative."""
        tracker = protocol.TurnTracker()
        tracker.track("ctx-1")
        tracker.refund("ctx-1")
        tracker.refund("ctx-1")  # second refund
        assert tracker.get_count("ctx-1") == 0

    def test_refund_nonexistent_context(self):
        """refund() on nonexistent context returns 0."""
        tracker = protocol.TurnTracker()
        result = tracker.refund("ctx-nonexistent")
        assert result == 0

    def test_is_transport_failure_timeout(self):
        """_is_transport_failure detects timeout."""
        assert adapter.A2AAdapter._is_transport_failure("[agent did not reply in time]")

    def test_is_transport_failure_disconnect(self):
        """_is_transport_failure detects client disconnect."""
        assert adapter.A2AAdapter._is_transport_failure("[client disconnected]")

    def test_is_transport_failure_dispatch_error(self):
        """_is_transport_failure detects dispatch failures."""
        assert adapter.A2AAdapter._is_transport_failure("Dispatch failed: connection refused")
        assert adapter.A2AAdapter._is_transport_failure("Agent gateway not ready")

    def test_is_transport_failure_empty_reply(self):
        """_is_transport_failure treats empty reply as transport failure."""
        assert adapter.A2AAdapter._is_transport_failure("")

    def test_is_not_transport_failure_genuine_error(self):
        """_is_transport_failure returns False for genuine agent errors."""
        assert not adapter.A2AAdapter._is_transport_failure("I don't understand that request")
        assert not adapter.A2AAdapter._is_transport_failure("Task failed: invalid input")


class TestPerPeerTurnConfig:
    """Test per-peer turn limit configuration."""

    def test_max_pingpong_turns_global_default(self, monkeypatch):
        """max_pingpong_turns() returns global default when no peer specified."""
        monkeypatch.delenv("A2A_MAX_PINGPONG_TURNS", raising=False)
        assert protocol.max_pingpong_turns() == 5

    def test_max_pingpong_turns_global_env(self, monkeypatch):
        """max_pingpong_turns() respects global env var."""
        monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "10")
        assert protocol.max_pingpong_turns() == 10

    def test_max_pingpong_turns_peer_override(self, monkeypatch, tmp_path):
        """max_pingpong_turns(peer) reads per-peer config."""
        monkeypatch.delenv("A2A_MAX_PINGPONG_TURNS", raising=False)
        
        # Create a temp config with per-peer max_turns
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
a2a_agents:
  researcher:
    url: "http://localhost:9991"
    max_turns: 15
  coder:
    url: "http://localhost:9992"
""")
        
        # Mock _load_config to return our test config
        original_load = protocol._load_config if hasattr(protocol, '_load_config') else None
        
        def mock_load_config():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        # Patch at module level
        import plugins.platforms.a2a.tools as tools_module
        original = tools_module._load_config
        tools_module._load_config = mock_load_config
        
        try:
            # researcher has max_turns: 15
            assert protocol.max_pingpong_turns("researcher") == 15
            # coder has no max_turns, falls back to global (5)
            assert protocol.max_pingpong_turns("coder") == 5
            # unknown peer falls back to global
            assert protocol.max_pingpong_turns("unknown") == 5
        finally:
            tools_module._load_config = original

    def test_max_pingpong_turns_peer_hard_cap(self, monkeypatch, tmp_path):
        """Per-peer max_turns respects hard cap of 20."""
        monkeypatch.delenv("A2A_MAX_PINGPONG_TURNS", raising=False)
        
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
a2a_agents:
  high_limit:
    url: "http://localhost:9991"
    max_turns: 50
""")
        
        def mock_load_config():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        import plugins.platforms.a2a.tools as tools_module
        original = tools_module._load_config
        tools_module._load_config = mock_load_config
        
        try:
            # 50 is capped at hard max of 20
            assert protocol.max_pingpong_turns("high_limit") == 20
        finally:
            tools_module._load_config = original

    def test_max_pingpong_turns_peer_min_one(self, monkeypatch, tmp_path):
        """Per-peer max_turns has minimum of 1."""
        monkeypatch.delenv("A2A_MAX_PINGPONG_TURNS", raising=False)
        
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
a2a_agents:
  zero_limit:
    url: "http://localhost:9991"
    max_turns: 0
""")
        
        def mock_load_config():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        import plugins.platforms.a2a.tools as tools_module
        original = tools_module._load_config
        tools_module._load_config = mock_load_config
        
        try:
            # 0 is floored at 1
            assert protocol.max_pingpong_turns("zero_limit") == 1
        finally:
            tools_module._load_config = original
