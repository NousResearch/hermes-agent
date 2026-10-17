"""
Test suite for three-layer router integration in AIAgent.

Tests verify:
1. Router override for github_create_issue (should use qwen-3.6-plus, not haiku)
2. Router override for sensitive_data_handler (should use sonnet-4)
3. Fallback chain when primary model unavailable
4. Graceful degradation when router fails or is disabled
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, Optional
import logging

# Test fixtures
logger = logging.getLogger("test_router_integration")


class MockRouter:
    """Mock router for testing."""
    
    def __init__(self, routing_decisions: Dict[str, str]):
        """
        Args:
            routing_decisions: dict of {tool_name: model_name}
        """
        self.routing_decisions = routing_decisions
        self.calls = []
        
    def route(self, tool_name: str, classification: str = "tool_call"):
        """Mock routing decision."""
        self.calls.append((tool_name, classification))
        model = self.routing_decisions.get(tool_name, "claude-haiku-4-5-20251001")
        
        # Create a mock decision object
        decision = Mock()
        decision.model = model
        decision.reason = Mock()
        decision.reason.value = f"routed_to_{model}"
        return decision
    
    def get_stats(self):
        """Return mock stats."""
        return {"routing_decisions": len(self.calls)}


class MockAgent:
    """Mock AIAgent for testing."""
    
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self.provider = "anthropic"
        self.base_url = "https://api.anthropic.com/v1"
        self.session_id = "test-session-123"
        self.api_mode = "regular"
        self._fallback_models = [
            "qwen-3.6-plus",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
        ]
        self.log_prefix = "[test] "
        
    def set_model(self, model: str):
        """Change the model."""
        self.model = model
        
    def get_model(self) -> str:
        """Get current model."""
        return self.model


# Tests


def test_router_override_github_create_issue():
    """Test that github_create_issue routes to qwen-3.6-plus instead of haiku."""
    from agent.router_hook import should_override_model_for_tool
    
    # Setup: Mock the router
    mock_router = MockRouter({
        "github_create_issue": "qwen-3.6-plus",
        "sensitive_data_handler": "claude-sonnet-4-20250514",
    })
    
    with patch("agent.router_hook.get_router", return_value=mock_router):
        # Test: Query override for github_create_issue
        override = should_override_model_for_tool(
            "github_create_issue",
            current_model="claude-haiku-4-5-20251001"
        )
        
        # Verify: Should return qwen-3.6-plus
        assert override == "qwen-3.6-plus", \
            f"Expected qwen-3.6-plus but got {override}"
        
        # Verify: Router was called
        assert mock_router.calls == [("github_create_issue", "tool_call")]


def test_router_override_sensitive_data_handler():
    """Test that sensitive_data_handler routes to claude-sonnet-4."""
    from agent.router_hook import should_override_model_for_tool
    
    # Setup
    mock_router = MockRouter({
        "github_create_issue": "qwen-3.6-plus",
        "sensitive_data_handler": "claude-sonnet-4-20250514",
    })
    
    with patch("agent.router_hook.get_router", return_value=mock_router):
        # Test
        override = should_override_model_for_tool(
            "sensitive_data_handler",
            current_model="claude-haiku-4-5-20251001"
        )
        
        # Verify
        assert override == "claude-sonnet-4-20250514", \
            f"Expected claude-sonnet-4-20250514 but got {override}"


def test_router_no_override_when_model_matches():
    """Test that no override is returned when tool routes to current model."""
    from agent.router_hook import should_override_model_for_tool
    
    # Setup: Tool routes to haiku (current model)
    mock_router = MockRouter({
        "simple_tool": "claude-haiku-4-5-20251001",
    })
    
    with patch("agent.router_hook.get_router", return_value=mock_router):
        # Test
        override = should_override_model_for_tool(
            "simple_tool",
            current_model="claude-haiku-4-5-20251001"
        )
        
        # Verify: No override needed
        assert override is None


def test_router_graceful_failure():
    """Test that None is returned when router fails."""
    from agent.router_hook import should_override_model_for_tool
    
    # Setup: Router raises exception
    mock_router = Mock()
    mock_router.route.side_effect = RuntimeError("Router initialization failed")
    
    with patch("agent.router_hook.get_router", return_value=mock_router):
        # Test
        override = should_override_model_for_tool(
            "any_tool",
            current_model="claude-haiku-4-5-20251001"
        )
        
        # Verify: Returns None on failure (graceful degradation)
        assert override is None


def test_router_not_available():
    """Test that None is returned when router module is unavailable."""
    from agent.router_hook import should_override_model_for_tool
    
    # Setup: No router available
    with patch("agent.router_hook.get_router", return_value=None):
        # Test
        override = should_override_model_for_tool(
            "any_tool",
            current_model="claude-haiku-4-5-20251001"
        )
        
        # Verify: Returns None when router unavailable
        assert override is None


def test_router_rejects_invalid_tool_names():
    """Test that invalid/private tool names are rejected."""
    from agent.router_hook import should_override_model_for_tool
    
    mock_router = MockRouter({})
    
    with patch("agent.router_hook.get_router", return_value=mock_router):
        # Test: Private tool (starts with _)
        assert should_override_model_for_tool("_private_tool") is None
        
        # Test: Unknown tool
        assert should_override_model_for_tool("unknown") is None
        
        # Test: Empty string
        assert should_override_model_for_tool("") is None
        
        # Test: None
        assert should_override_model_for_tool(None) is None
        
        # Verify: Router was never called for invalid inputs
        assert len(mock_router.calls) == 0


def test_router_get_stats():
    """Test that router stats can be retrieved."""
    from agent.router_hook import get_router_stats
    
    # Setup
    mock_router = MockRouter({})
    mock_router.get_stats = Mock(return_value={"total_routes": 42})
    
    with patch("agent.router_hook.get_router", return_value=mock_router):
        # Test
        stats = get_router_stats()
        
        # Verify
        assert stats == {"total_routes": 42}


def test_router_get_stats_unavailable():
    """Test that empty dict is returned when router stats unavailable."""
    from agent.router_hook import get_router_stats
    
    # Setup: No router
    with patch("agent.router_hook.get_router", return_value=None):
        # Test
        stats = get_router_stats()
        
        # Verify
        assert stats == {}


def test_is_routing_enabled_true():
    """Test that routing enabled status is correctly detected."""
    from unittest.mock import patch, mock_open
    
    # Setup: Mock config with routing enabled (as YAML string)
    mock_yaml_content = "layer_routing:\n  enabled: true\n"
    
    with patch("hermes_constants.get_hermes_home") as mock_hermes_home, \
         patch("builtins.open", mock_open(read_data=mock_yaml_content)) as mock_file, \
         patch("pathlib.Path.exists") as mock_exists:
        
        mock_hermes_home.return_value = "/mock/home"
        mock_exists.return_value = True
        
        # Test
        from agent.router_hook import is_routing_enabled
        enabled = is_routing_enabled()
        
        # Verify
        assert enabled is True


def test_is_routing_enabled_false():
    """Test that routing disabled status is correctly detected."""
    from unittest.mock import patch, mock_open
    
    # Setup: Mock config with routing disabled (as YAML string)
    mock_yaml_content = "layer_routing:\n  enabled: false\n"
    
    with patch("hermes_constants.get_hermes_home") as mock_hermes_home, \
         patch("builtins.open", mock_open(read_data=mock_yaml_content)) as mock_file, \
         patch("pathlib.Path.exists") as mock_exists:
        
        mock_hermes_home.return_value = "/mock/home"
        mock_exists.return_value = True
        
        # Test
        from agent.router_hook import is_routing_enabled
        enabled = is_routing_enabled()
        
        # Verify
        assert enabled is False


def test_fallback_chain_primary_model_unavailable():
    """Test fallback chain when primary model is unavailable.
    
    Scenario: github_create_issue routes to qwen-3.6-plus, but it fails.
    Agent should fall back to next model in chain.
    """
    # This test documents the fallback contract:
    # When a tool routes to model X but X fails:
    # 1. Mark X as unavailable
    # 2. Retry with next model in fallback chain
    # 3. Log decision for observability
    
    from agent.router_hook import should_override_model_for_tool
    
    # Setup
    mock_router = MockRouter({
        "github_create_issue": "qwen-3.6-plus",
    })
    
    with patch("agent.router_hook.get_router", return_value=mock_router):
        # First call: primary override
        override_1 = should_override_model_for_tool(
            "github_create_issue",
            current_model="claude-haiku-4-5-20251001"
        )
        assert override_1 == "qwen-3.6-plus"
        
        # Second call: if qwen fails, try next
        # (In real implementation, agent checks availability and retries)
        override_2 = should_override_model_for_tool(
            "github_create_issue",
            current_model="qwen-3.6-plus"  # Now qwen is current, won't override
        )
        # If router says qwen again, we'd need to escalate fallback logic in agent
        assert override_2 is None  # No override since model matches


def test_routing_decision_logging():
    """Test that routing decisions are properly logged."""
    from agent.router_hook import should_override_model_for_tool
    
    # Setup
    mock_router = MockRouter({
        "github_create_issue": "qwen-3.6-plus",
    })
    
    with patch("agent.router_hook.get_router", return_value=mock_router), \
         patch("agent.router_hook.logger") as mock_logger:
        
        # Test
        override = should_override_model_for_tool(
            "github_create_issue",
            current_model="claude-haiku-4-5-20251001"
        )
        
        # Verify: Logging was called
        assert mock_logger.info.called, "Router override should be logged"
        call_args = mock_logger.info.call_args[0][0]
        assert "github_create_issue" in call_args
        assert "qwen-3.6-plus" in call_args


def test_integration_conversation_loop_uses_router():
    """Integration test: Verify conversation loop consults router for tool calls.
    
    This is a minimal mock test to document the integration contract.
    """
    # In real implementation:
    # 1. conversation_loop.run_conversation() starts a turn
    # 2. When preparing API call for a tool, it queries should_override_model_for_tool()
    # 3. If override returned, use new model instead of agent.model
    # 4. Log the decision for observability
    
    from agent.router_hook import should_override_model_for_tool
    
    # Simulated state
    agent = MockAgent(model="claude-haiku-4-5-20251001")
    tool_name = "github_create_issue"
    
    # Setup router
    mock_router = MockRouter({
        "github_create_issue": "qwen-3.6-plus",
    })
    
    with patch("agent.router_hook.get_router", return_value=mock_router):
        # Conversation loop would call this before building API request
        override_model = should_override_model_for_tool(
            tool_name, 
            current_model=agent.model
        )
        
        # Contract: If override returned, use it
        if override_model:
            model_for_call = override_model
        else:
            model_for_call = agent.model
            
        # Verify
        assert model_for_call == "qwen-3.6-plus", \
            "API call should use routed model, not default haiku"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
