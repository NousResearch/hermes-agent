"""Integration tests for prompt caching v2 in run_agent.py.

Tests the wiring between _build_system_prompt_blocks(), API message assembly,
and apply_anthropic_cache_control_v2.
"""

import copy
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent
from agent.prompt_caching import SystemPromptBlock, build_system_content_blocks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    """Minimal AIAgent with prompt caching enabled."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


@pytest.fixture()
def caching_agent():
    """Agent with prompt caching enabled (simulating Claude via OpenRouter)."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search", "memory")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key",
            model="anthropic/claude-sonnet-4-20250514",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._use_prompt_caching = True
        return a


# ---------------------------------------------------------------------------
# Tests: _build_system_prompt_blocks
# ---------------------------------------------------------------------------

class TestBuildSystemPromptBlocks:
    def test_returns_list_of_blocks(self, agent):
        """_build_system_prompt_blocks() should return SystemPromptBlock list."""
        if not hasattr(agent, '_build_system_prompt_blocks'):
            pytest.skip("_build_system_prompt_blocks not implemented yet")
        blocks = agent._build_system_prompt_blocks()
        assert isinstance(blocks, list)
        assert all(isinstance(b, SystemPromptBlock) for b in blocks)

    def test_has_three_blocks(self, agent):
        """Should have static, session, and ephemeral blocks."""
        if not hasattr(agent, '_build_system_prompt_blocks'):
            pytest.skip("_build_system_prompt_blocks not implemented yet")
        blocks = agent._build_system_prompt_blocks()
        labels = [b.label for b in blocks]
        assert "static" in labels
        assert "session" in labels
        assert "ephemeral" in labels

    def test_static_block_has_1h_ttl(self, agent):
        """Static block (identity + skills + guidance) should have 1h TTL."""
        if not hasattr(agent, '_build_system_prompt_blocks'):
            pytest.skip("_build_system_prompt_blocks not implemented yet")
        blocks = agent._build_system_prompt_blocks()
        static = [b for b in blocks if b.label == "static"][0]
        assert static.cache_ttl == "1h"

    def test_session_block_has_5m_ttl(self, agent):
        """Session block (memory + context) should have 5m TTL."""
        if not hasattr(agent, '_build_system_prompt_blocks'):
            pytest.skip("_build_system_prompt_blocks not implemented yet")
        blocks = agent._build_system_prompt_blocks()
        session = [b for b in blocks if b.label == "session"][0]
        assert session.cache_ttl == "5m"

    def test_ephemeral_block_has_no_ttl(self, agent):
        """Ephemeral block (timestamp + platform) should have no cache TTL."""
        if not hasattr(agent, '_build_system_prompt_blocks'):
            pytest.skip("_build_system_prompt_blocks not implemented yet")
        blocks = agent._build_system_prompt_blocks()
        ephemeral = [b for b in blocks if b.label == "ephemeral"][0]
        assert ephemeral.cache_ttl is None

    def test_timestamp_in_ephemeral_block(self, agent):
        """Timestamp should be in the ephemeral block, not in cached blocks."""
        if not hasattr(agent, '_build_system_prompt_blocks'):
            pytest.skip("_build_system_prompt_blocks not implemented yet")
        blocks = agent._build_system_prompt_blocks()
        ephemeral = [b for b in blocks if b.label == "ephemeral"][0]
        assert "Conversation started:" in ephemeral.text

        # Verify timestamp is NOT in cached blocks
        static = [b for b in blocks if b.label == "static"][0]
        session = [b for b in blocks if b.label == "session"][0]
        assert "Conversation started:" not in static.text
        if session.text:
            assert "Conversation started:" not in session.text

    def test_identity_in_static_block(self, agent):
        """Agent identity should be in the static block."""
        if not hasattr(agent, '_build_system_prompt_blocks'):
            pytest.skip("_build_system_prompt_blocks not implemented yet")
        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
        blocks = agent._build_system_prompt_blocks()
        static = [b for b in blocks if b.label == "static"][0]
        assert DEFAULT_AGENT_IDENTITY in static.text

    def test_backward_compat_build_system_prompt_returns_string(self, agent):
        """_build_system_prompt() should still return a string (backward compat)."""
        prompt = agent._build_system_prompt()
        assert isinstance(prompt, str)
        assert "Conversation started:" in prompt

    def test_blocks_join_equals_string(self, agent):
        """Joining blocks should produce the same content as _build_system_prompt()."""
        if not hasattr(agent, '_build_system_prompt_blocks'):
            pytest.skip("_build_system_prompt_blocks not implemented yet")
        blocks = agent._build_system_prompt_blocks()
        joined = "\n\n".join(b.text for b in blocks if b.text)
        prompt_str = agent._build_system_prompt()
        assert joined == prompt_str


# ---------------------------------------------------------------------------
# Tests: Cached blocks
# ---------------------------------------------------------------------------

class TestCachedBlocks:
    def test_cached_system_blocks_stored(self, caching_agent):
        """After building, blocks should be cached on the agent."""
        if not hasattr(caching_agent, '_cached_system_blocks'):
            pytest.skip("_cached_system_blocks not implemented yet")
        caching_agent._build_system_prompt()
        assert caching_agent._cached_system_blocks is not None
        assert len(caching_agent._cached_system_blocks) >= 3

    def test_invalidate_clears_blocks(self, caching_agent):
        """_invalidate_system_prompt_cache should clear blocks too."""
        if not hasattr(caching_agent, '_cached_system_blocks'):
            pytest.skip("_cached_system_blocks not implemented yet")
        caching_agent._build_system_prompt()
        caching_agent._invalidate_system_prompt()
        assert caching_agent._cached_system_prompt is None
        assert caching_agent._cached_system_blocks is None


# ---------------------------------------------------------------------------
# Tests: Non-Anthropic models unaffected
# ---------------------------------------------------------------------------

class TestNonAnthropicUnaffected:
    def test_non_claude_model_no_caching(self):
        """Non-Claude models should not have prompt caching enabled."""
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-key",
                model="gpt-4o",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        assert a._use_prompt_caching is False

    def test_system_prompt_still_string_for_non_caching(self, agent):
        """When caching is off, system prompt is a plain string."""
        agent._use_prompt_caching = False
        prompt = agent._build_system_prompt()
        assert isinstance(prompt, str)
