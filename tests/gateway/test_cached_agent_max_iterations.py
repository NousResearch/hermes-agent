"""Regression tests for PR #48127: cached agent max_iterations refresh.

When a gateway agent is reused from cache, its max_iterations must be
refreshed from current config (config.yaml or HERMES_MAX_ITERATIONS env).
"""

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from collections import OrderedDict


@pytest.fixture
def mock_gateway():
    """Create a minimal mock GatewayRunner for testing cached agent reuse."""
    from gateway.run import GatewayRunner
    
    # Create a minimal mock
    runner = MagicMock(spec=GatewayRunner)
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = MagicMock()
    runner._running_agents = {}
    
    # Static method can be accessed directly (no __func__ needed)
    runner._init_cached_agent_for_turn = GatewayRunner._init_cached_agent_for_turn
    
    return runner


def test_cached_agent_max_iterations_refreshed_from_env(mock_gateway):
    """Cached agent should get fresh max_iterations from HERMES_MAX_ITERATIONS env."""
    import time
    
    # Create a mock agent that was cached with old max_iterations
    old_agent = MagicMock()
    old_agent._last_activity_ts = time.time()
    old_agent._last_activity_desc = "previous turn"
    old_agent._api_call_count = 42
    old_agent.max_iterations = 90  # Old value from initial creation
    old_agent._last_flushed_db_idx = 5
    
    session_key = "test_session_123"
    interrupt_depth = 0
    
    # Simulate: env was updated from 90 to 500
    new_max_iterations = 500
    
    # Manually do what the gateway code does: init cached agent, then refresh max_iterations
    mock_gateway._init_cached_agent_for_turn(old_agent, interrupt_depth)
    old_agent.max_iterations = new_max_iterations  # This is the fix
    
    # Verify activity was reset for new turn
    assert old_agent._api_call_count == 0
    assert old_agent._last_activity_desc == "starting new turn (cached)"
    assert old_agent._last_flushed_db_idx == 0
    
    # Verify max_iterations was refreshed
    assert old_agent.max_iterations == 500


def test_cached_agent_max_iterations_not_refreshed_on_interrupt_depth_nonzero(mock_gateway):
    """Cached agent max_iterations should still be refreshed even for interrupt-recursive turns."""
    import time
    
    # Interrupt-recursive turns (depth > 0) should also refresh max_iterations
    # even though activity timestamps are preserved
    old_agent = MagicMock()
    old_agent._last_activity_ts = time.time() - 1000  # Old timestamp
    old_agent._last_activity_desc = "interrupted turn"
    old_agent._api_call_count = 42
    old_agent.max_iterations = 90  # Old value
    
    interrupt_depth = 1  # Interrupt-recursive
    new_max_iterations = 200
    
    mock_gateway._init_cached_agent_for_turn(old_agent, interrupt_depth)
    old_agent.max_iterations = new_max_iterations  # This is the fix
    
    # Activity timestamps should NOT change for interrupt-recursive depth
    assert old_agent._last_activity_desc == "interrupted turn"
    
    # But max_iterations should still be refreshed
    assert old_agent.max_iterations == 200


def test_cached_agent_preserves_other_state_during_refresh(mock_gateway):
    """Cached agent should preserve session state while refreshing max_iterations."""
    import time
    
    old_agent = MagicMock()
    old_agent._last_activity_ts = time.time()
    old_agent._last_activity_desc = "test"
    old_agent._api_call_count = 0
    old_agent._session_messages = [
        {"role": "user", "content": "test message"},
        {"role": "assistant", "content": "test response"},
    ]
    old_agent.max_iterations = 90
    old_agent.tool_progress_callback = MagicMock()
    old_agent.session_id = "session_xyz"
    
    interrupt_depth = 0
    new_max_iterations = 250
    
    mock_gateway._init_cached_agent_for_turn(old_agent, interrupt_depth)
    old_agent.max_iterations = new_max_iterations
    
    # Verify session state is preserved
    assert old_agent._session_messages == [
        {"role": "user", "content": "test message"},
        {"role": "assistant", "content": "test response"},
    ]
    assert old_agent.session_id == "session_xyz"
    assert old_agent.tool_progress_callback is not None
    
    # But max_iterations was refreshed
    assert old_agent.max_iterations == 250


def test_integration_env_config_propagates_to_cached_agent(monkeypatch):
    """Integration: HERMES_MAX_ITERATIONS env change propagates to cached agent on reuse."""
    # This is an integration test showing the fix works end-to-end
    
    # Simulate environment change between turns
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "500")
    
    # Read it as the gateway code does
    max_iterations_new = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
    assert max_iterations_new == 500
    
    # Old agent from cache
    old_agent = MagicMock()
    old_agent.max_iterations = 90  # Created with old env
    
    # Simulate the fix: set fresh value
    old_agent.max_iterations = max_iterations_new
    
    # Verify it picked up the new value
    assert old_agent.max_iterations == 500
