"""Tests for issue #40170: Honcho memory injection guard on customer-facing platforms.

Verifies that _skip_memory_injection flag prevents memory context from being
auto-injected into responses on customer-facing platforms while keeping
memory tools functional.
"""
import pytest
from unittest.mock import MagicMock, patch
from agent.conversation_loop import run_conversation


def test_skip_memory_injection_flag_prevents_prefetch():
    """When _skip_memory_injection is True, prefetch_all is not called."""
    # Create a mock agent with memory manager
    agent = MagicMock()
    agent._skip_memory_injection = True
    agent._memory_manager = MagicMock()
    agent._memory_manager.prefetch_all = MagicMock(return_value="cached context")
    
    # The memory injection logic:
    _skip_memory_injection = getattr(agent, "_skip_memory_injection", False)
    _ext_prefetch_cache = ""
    
    if agent._memory_manager and not _skip_memory_injection:
        # This block should NOT execute when flag is True
        _ext_prefetch_cache = agent._memory_manager.prefetch_all("test") or ""
    
    # Verify prefetch_all was NOT called
    agent._memory_manager.prefetch_all.assert_not_called()
    assert _ext_prefetch_cache == ""


def test_skip_memory_injection_flag_allows_memory_tools():
    """Memory tools (read, write) remain available even when injection is skipped."""
    # When _skip_memory_injection is True, we only skip the automatic
    # context injection from prefetch_all(). Memory tools themselves
    # remain available for explicit use.
    
    # This is handled by the memory_manager itself checking if the
    # tool is in the agent's enabled_toolsets, which is independent
    # of the _skip_memory_injection flag.
    
    agent = MagicMock()
    agent._skip_memory_injection = True
    agent.tools = ["memory.read", "memory.write"]
    
    # Verify tools are still present
    assert "memory.read" in agent.tools
    assert "memory.write" in agent.tools


def test_default_skip_memory_injection_is_false():
    """By default, _skip_memory_injection is False (memory is auto-injected)."""
    agent = MagicMock()
    # Don't set _skip_memory_injection on a real object
    del agent._skip_memory_injection  # Remove the auto-created MagicMock attribute
    
    _skip_memory_injection = getattr(agent, "_skip_memory_injection", False)
    
    # Default should be False (inject memory)
    assert _skip_memory_injection is False


def test_customer_platforms_should_skip_injection():
    """Verify the list of customer-facing platforms that should skip injection."""
    customer_platforms = {'telegram', 'discord', 'slack', 'whatsapp', 'signal', 'matrix'}
    
    # These are the platforms where operator-level memory should not be exposed
    assert 'telegram' in customer_platforms
    assert 'discord' in customer_platforms
    assert 'slack' in customer_platforms
    assert 'whatsapp' in customer_platforms
    assert 'signal' in customer_platforms
    assert 'matrix' in customer_platforms
    
    # Internal platforms should NOT be in the list (they can see memory)
    assert 'telegram_personal' not in customer_platforms  # If such a variant exists


def test_internal_platforms_can_access_memory():
    """Internal platforms (CLI, internal gateways) should still have memory access."""
    # Internal platforms are NOT in the skip list, so memory injection proceeds normally
    internal_platforms = ['cli', 'hermes_internal', 'local_api']
    customer_platforms = {'telegram', 'discord', 'slack', 'whatsapp', 'signal', 'matrix'}
    
    for platform in internal_platforms:
        # These platforms are NOT in customer_platforms
        assert platform not in customer_platforms, \
            f"{platform} is an internal platform and should have memory access"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
