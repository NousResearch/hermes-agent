"""External personal memory must not be borrowed by machine-origin workers."""
from types import SimpleNamespace
from agent.memory_manager import MemoryManager

import pytest
from agent.agent_init import _init_memory
from gateway.session_context import _SESSION_SOURCE


@pytest.mark.parametrize("source,platform", [("kanban", "cli"), ("tool", "cli"), ("desktop", "subagent"), ("subagent", "cli")])
def test_worker_does_not_reuse_human_memory_manager(source, platform):
    human = MemoryManager()
    agent = SimpleNamespace(enabled_toolsets=[], disabled_toolsets=[], tools=[], valid_tool_names=set())
    token = _SESSION_SOURCE.set(source)
    try:
        _init_memory(agent, {}, False, platform, memory_manager=human)
    finally:
        _SESSION_SOURCE.reset(token)
    # The ordinary agent teardown shuts its manager down. A borrowed manager must survive.
    if agent._memory_manager is not None:
        agent._memory_manager.shutdown_all()
    assert not human._shutting_down
    assert agent._memory_manager is None


@pytest.mark.parametrize("source", ["cli", "desktop", "oneshot", "api_server"])
def test_human_supplied_memory_manager_remains_available(source):
    human = MemoryManager()
    agent = SimpleNamespace(enabled_toolsets=[], disabled_toolsets=[], tools=[], valid_tool_names=set())
    token = _SESSION_SOURCE.set(source)
    try:
        _init_memory(agent, {}, False, "cli", memory_manager=human)
    finally:
        _SESSION_SOURCE.reset(token)
    assert agent._memory_manager is human
