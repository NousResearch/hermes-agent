"""Tests that skip_memory=True removes the memory tool from the agent surface."""

import pytest
from run_agent import AIAgent


class TestSkipMemoryRemovesTool:
    """When skip_memory=True, the memory tool should not be in agent.tools."""

    def test_memory_tool_present_without_skip(self):
        """Without skip_memory, the memory tool should be available."""
        agent = AIAgent(
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            model="test/model",
            quiet_mode=True,
            skip_memory=False,
            skip_context_files=True,
        )
        tool_names = {t["function"]["name"] for t in (agent.tools or [])}
        assert "memory" in tool_names, "memory tool should be present when skip_memory=False"

    def test_memory_tool_absent_with_skip(self):
        """With skip_memory=True, the memory tool should be removed."""
        agent = AIAgent(
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            model="test/model",
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
        )
        tool_names = {t["function"]["name"] for t in (agent.tools or [])}
        assert "memory" not in tool_names, (
            "memory tool must be removed when skip_memory=True; "
            "leaving it causes runtime failures in cron-like sessions"
        )

    def test_valid_tool_names_excludes_memory_with_skip(self):
        """valid_tool_names should also exclude memory when skip_memory=True."""
        agent = AIAgent(
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            model="test/model",
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
        )
        assert "memory" not in agent.valid_tool_names

    def test_skip_memory_false_preserves_valid_tool_names(self):
        """valid_tool_names should include memory when skip_memory=False."""
        agent = AIAgent(
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            model="test/model",
            quiet_mode=True,
            skip_memory=False,
            skip_context_files=True,
        )
        assert "memory" in agent.valid_tool_names

    def test_other_tools_unaffected_by_skip_memory(self):
        """Core tools like read_file, write_file, terminal should remain."""
        agent = AIAgent(
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            model="test/model",
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
        )
        tool_names = {t["function"]["name"] for t in (agent.tools or [])}
        # These tools should always be present regardless of skip_memory
        for expected in ("read_file", "write_file", "terminal", "web_search"):
            assert expected in tool_names, (
                f"{expected} tool should still be present when skip_memory=True"
            )
