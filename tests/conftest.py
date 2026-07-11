"""Pytest fixtures for tool_input_repair tests."""

import pytest

# Discover tools before running tests so schemas are available
@pytest.fixture(autouse=True)
def discover_tools(request):
    """Auto-discover tools before each test."""
    import sys
    # Add both main repo and worktree to path
    sys.path.insert(0, "/home/hermes/hermes-agent")
    sys.path.insert(0, "/home/hermes/hermes-agent/.worktrees/tool-input-repair")

    import tools.registry
    tools.registry.discover_builtin_tools()

    yield