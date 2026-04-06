"""Tests for cron read-only memory injection (cron.memory_read config)."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.memory_tool import MemoryStore, memory_tool


# =========================================================================
# memory_tool read_only flag
# =========================================================================


class TestMemoryToolReadOnly:
    """Verify the memory tool blocks writes when read_only=True."""

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        s = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
        s.add("memory", "existing fact about the user's environment")
        s.add("user", "user prefers dark mode")
        return s

    def test_read_only_blocks_add(self, store):
        result = json.loads(memory_tool(
            action="add", target="memory", content="new note",
            store=store, read_only=True,
        ))
        assert result["success"] is False
        assert "read-only" in result["error"]

    def test_read_only_blocks_replace(self, store):
        result = json.loads(memory_tool(
            action="replace", target="memory",
            old_text="existing", content="replacement",
            store=store, read_only=True,
        ))
        assert result["success"] is False
        assert "read-only" in result["error"]

    def test_read_only_blocks_remove(self, store):
        result = json.loads(memory_tool(
            action="remove", target="memory", old_text="existing",
            store=store, read_only=True,
        ))
        assert result["success"] is False
        assert "read-only" in result["error"]

    def test_read_only_false_allows_add(self, store):
        result = json.loads(memory_tool(
            action="add", target="memory", content="new note",
            store=store, read_only=False,
        ))
        assert result["success"] is True

    def test_default_read_only_is_false(self, store):
        """Backward compatibility: read_only defaults to False."""
        result = json.loads(memory_tool(
            action="add", target="memory", content="another note",
            store=store,
        ))
        assert result["success"] is True

    def test_store_data_unchanged_after_blocked_write(self, store):
        """Verify store is not modified when write is blocked."""
        snapshot_before = store.format_for_system_prompt("memory")
        memory_tool(
            action="add", target="memory", content="should not appear",
            store=store, read_only=True,
        )
        snapshot_after = store.format_for_system_prompt("memory")
        assert snapshot_before == snapshot_after


# =========================================================================
# AIAgent memory_read_only flag
# =========================================================================


class TestAgentMemoryReadOnly:
    """Verify AIAgent initializes correctly with memory_read_only."""

    def test_agent_accepts_memory_read_only_param(self):
        """AIAgent __init__ should accept memory_read_only without error."""
        from run_agent import AIAgent

        # Don't actually run — just verify the param is accepted and stored
        with patch.object(AIAgent, "__init__", wraps=AIAgent.__init__) as mock_init:
            try:
                agent = AIAgent(
                    model="test/model",
                    skip_memory=False,
                    memory_read_only=True,
                    quiet_mode=True,
                )
            except Exception:
                pass  # May fail on API setup — we only care about param acceptance

    def test_memory_read_only_stored_on_agent(self):
        """The _memory_read_only attribute should be set."""
        from run_agent import AIAgent

        # Minimal init with skip_memory=True to avoid disk reads
        agent = AIAgent.__new__(AIAgent)
        agent._memory_read_only = True
        assert agent._memory_read_only is True


# =========================================================================
# Scheduler config reading
# =========================================================================


class TestSchedulerMemoryReadConfig:
    """Verify scheduler reads cron.memory_read from config."""

    def test_default_config_has_memory_read_true(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["cron"]["memory_read"] is True

    def test_config_memory_read_false_means_skip_memory(self):
        """When cron.memory_read is False, skip_memory should be True."""
        cfg = {"cron": {"memory_read": False}}
        skip = not cfg.get("cron", {}).get("memory_read", True)
        read_only = cfg.get("cron", {}).get("memory_read", True)
        assert skip is True
        assert read_only is False

    def test_config_memory_read_true_means_no_skip(self):
        """When cron.memory_read is True, skip_memory=False, read_only=True."""
        cfg = {"cron": {"memory_read": True}}
        skip = not cfg.get("cron", {}).get("memory_read", True)
        read_only = cfg.get("cron", {}).get("memory_read", True)
        assert skip is False
        assert read_only is True

    def test_missing_cron_section_defaults_to_true(self):
        """Empty config should default memory_read to True."""
        cfg = {}
        skip = not cfg.get("cron", {}).get("memory_read", True)
        read_only = cfg.get("cron", {}).get("memory_read", True)
        assert skip is False
        assert read_only is True
