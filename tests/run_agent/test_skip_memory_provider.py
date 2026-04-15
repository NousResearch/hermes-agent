"""Tests for AIAgent skip_memory_provider parameter."""

from unittest.mock import patch, MagicMock

import pytest

from run_agent import AIAgent


class TestSkipMemoryProvider:
    """Verify that skip_memory_provider controls external memory provider initialization
    independently from skip_memory (which controls local MEMORY.md/USER.md)."""

    @pytest.fixture
    def minimal_agent_kwargs(self):
        return {
            "model": "test/model",
            "quiet_mode": True,
            "skip_memory": True,
            "skip_context_files": True,
        }

    def test_skip_memory_provider_false_initializes_manager(self, minimal_agent_kwargs):
        """When skip_memory_provider=False and a provider is configured, _memory_manager is initialized."""
        with patch("hermes_cli.config.load_config", return_value={"memory": {"provider": "mem0"}}), \
             patch("plugins.memory.load_memory_provider") as mock_load, \
             patch("agent.memory_manager.MemoryManager") as mock_mgr_cls:
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            mock_load.return_value = mock_provider
            mock_mgr = MagicMock()
            mock_mgr.providers = [mock_provider]
            mock_mgr_cls.return_value = mock_mgr

            agent = AIAgent(**minimal_agent_kwargs, skip_memory_provider=False)

        assert agent._memory_manager is not None

    def test_skip_memory_provider_true_skips_manager(self, minimal_agent_kwargs):
        """When skip_memory_provider=True, _memory_manager remains None even if provider is configured."""
        with patch("hermes_cli.config.load_config", return_value={"memory": {"provider": "mem0"}}), \
             patch("plugins.memory.load_memory_provider") as mock_load, \
             patch("agent.memory_manager.MemoryManager") as mock_mgr_cls:
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            mock_load.return_value = mock_provider

            agent = AIAgent(**minimal_agent_kwargs, skip_memory_provider=True)

        assert agent._memory_manager is None
        mock_load.assert_not_called()
        mock_mgr_cls.assert_not_called()

    def test_skip_memory_true_still_allows_provider_when_flag_is_false(self, minimal_agent_kwargs):
        """The common cron pattern: skip_memory=True (no local files) + skip_memory_provider=False (allow mem0)."""
        with patch("hermes_cli.config.load_config", return_value={"memory": {"provider": "mem0"}}), \
             patch("plugins.memory.load_memory_provider") as mock_load, \
             patch("agent.memory_manager.MemoryManager") as mock_mgr_cls:
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            mock_load.return_value = mock_provider
            mock_mgr = MagicMock()
            mock_mgr.providers = [mock_provider]
            mock_mgr_cls.return_value = mock_mgr

            agent = AIAgent(**minimal_agent_kwargs, skip_memory_provider=False)

        assert agent._memory_store is None  # local memory skipped
        assert agent._memory_manager is not None  # external provider allowed

    def test_skip_memory_provider_defaults_to_skip_memory(self):
        """When skip_memory_provider is omitted, it defaults to the value of skip_memory
        to preserve backward compatibility for existing call sites."""
        with patch("hermes_cli.config.load_config", return_value={"memory": {"provider": "mem0"}}), \
             patch("plugins.memory.load_memory_provider") as mock_load, \
             patch("agent.memory_manager.MemoryManager") as mock_mgr_cls:
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            mock_load.return_value = mock_provider

            # Existing pattern: only skip_memory=True means skip everything
            agent = AIAgent(model="test/model", quiet_mode=True, skip_memory=True)

        assert agent._memory_manager is None
        mock_load.assert_not_called()
        mock_mgr_cls.assert_not_called()
