"""Keep a standalone AG-UI conversation's admitted tool contract intact."""

import copy
from functools import wraps


def pin_tool_snapshot(agent):
    """Restore this agent's frozen schemas before a prompt rebuild.

    Hermes deliberately refreshes registry-derived tools at compression's
    commit boundary. Client/state-writer schemas are merged after construction
    and are not in that selection, so refresh otherwise drops them (or takes
    another conversation's placeholder schema). AG-UI admits a fixed toolset
    for a conversation; rebuilding the prompt must not change that contract.

    This is an instance-local compatibility patch to an existing method, not
    a global patch or a new AIAgent protocol. The real prompt builder still
    runs, including compression's normal cache invalidation/reconstruction.
    """
    from agui_adapter.translate import json_contract as json_key

    tools = copy.deepcopy(agent.tools)
    names = set(agent.valid_tool_names)
    build_prompt = agent._build_system_prompt

    @wraps(build_prompt)
    def build_with_admitted_tools(*args, **kwargs):
        if json_key(agent.tools) != json_key(tools) or agent.valid_tool_names != names:
            agent.tools = copy.deepcopy(tools)
            agent.valid_tool_names = set(names)
            # The registry refresh may have persisted its intermediate snapshot.
            # Re-pin the actual admitted list, using Hermes' existing writer.
            from tools.mcp_tool_agent import persist_agent_tool_names

            persist_agent_tool_names(agent)
        return build_prompt(*args, **kwargs)

    agent._build_system_prompt = build_with_admitted_tools
