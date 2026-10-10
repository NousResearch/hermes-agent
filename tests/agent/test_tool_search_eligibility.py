"""Deferred executor eligibility follows the same reprobed catalog as the bridge."""

from copy import deepcopy
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("initially_available", [False, True])
@pytest.mark.parametrize("exclude_by", ["enabled", "disabled"])
def test_executor_eligibility_follows_reprobe_without_changing_prompt(
    initially_available, exclude_by, monkeypatch, tmp_path,
):
    import socket

    def forbidden(*args, **kwargs):
        pytest.fail("Eligibility resolution must not execute handlers or use the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    import model_tools
    from agent.tool_executor import _unwrap_tool_search_call
    from tools.mcp_tool_agent import reprobe_tool_availability
    from tools.registry import registry
    from tools.tool_search import scoped_deferrable_names

    available = initially_available
    probes = []

    def check():
        probes.append(available)
        return available

    name, outside = "eligibility_probe", "eligibility_outside"
    toolset, excluded = "eligibility-probe", "eligibility-outside"
    for tool_name, group, check_fn in ((name, toolset, check), (outside, excluded, None)):
        registry.register(
            name=tool_name, toolset=group, handler=forbidden, check_fn=check_fn,
            schema={"name": tool_name, "description": "Offline eligibility fixture",
                    "parameters": {"type": "object", "properties": {}}},
        )
    agent = SimpleNamespace(
        enabled_toolsets=[toolset] if exclude_by == "enabled" else [toolset, excluded],
        disabled_toolsets=None if exclude_by == "enabled" else [excluded],
        tools=[{"type": "function", "function": {"name": "tool_call"}}],
        _cached_system_prompt="Stable conversation prefix",
    )
    frozen_tools, frozen_prompt = deepcopy(agent.tools), agent._cached_system_prompt
    tools_identity = agent.tools
    generation = registry._generation
    envelope = {"calls": [{"name": name, "arguments": {}}]}
    outside_envelope = {"calls": [{"name": outside, "arguments": {}}]}

    try:
        reprobe_tool_availability()
        for expected in (initially_available, not initially_available):
            available = expected
            reprobe_tool_availability()
            assert registry._generation == generation
            definitions = model_tools.get_tool_definitions(
                enabled_toolsets=agent.enabled_toolsets,
                disabled_toolsets=agent.disabled_toolsets,
                quiet_mode=True, skip_tool_search_assembly=True,
            )
            assert (name in scoped_deferrable_names(definitions)) is expected
            assert outside not in scoped_deferrable_names(definitions)
            bridge_result, resolved = model_tools._dispatch_bridge_tool(
                "tool_call", envelope, agent.enabled_toolsets, agent.disabled_toolsets,
            )
            assert (resolved == (name, {})) is expected
            if not expected:
                assert "not available in this session" in bridge_result

            probe_count = len(probes)
            for _ in range(3):
                unwrapped = _unwrap_tool_search_call(agent, "tool_call", envelope)
                denied = _unwrap_tool_search_call(agent, "tool_call", outside_envelope)
                assert "not available in this session" in denied[2]
                assert len(probes) == probe_count  # reuse the definitions cache
                assert agent.tools is tools_identity
                assert agent.tools == frozen_tools
                assert agent._cached_system_prompt == frozen_prompt
                assert (unwrapped == (name, {}, None)) is expected
                if not expected:
                    assert "not available in this session" in unwrapped[2]
        assert probes == [initially_available, not initially_available]
    finally:
        registry.deregister(name)
        registry.deregister(outside)
        reprobe_tool_availability()
