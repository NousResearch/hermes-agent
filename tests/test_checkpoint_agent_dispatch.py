"""Verify that checkpoint is routed through the agent loop, not the registry."""
import json
import pytest
from model_tools import handle_function_call, _AGENT_LOOP_TOOLS


def test_checkpoint_is_agent_loop_tool():
    """checkpoint must be in _AGENT_LOOP_TOOLS so it gets agent-level state."""
    assert "checkpoint" in _AGENT_LOOP_TOOLS


def test_checkpoint_registry_dispatch_returns_error():
    """Calling handle_function_call for checkpoint should return agent-loop error."""
    result = handle_function_call("checkpoint", {"action": "read"})
    data = json.loads(result)
    assert "must be handled" in data["error"].lower()