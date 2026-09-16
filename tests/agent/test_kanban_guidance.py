"""Behavioral contracts for Kanban guidance during real agent tool loading."""

from contextlib import nullcontext

import pytest

from agent import agent_init, prompt_builder as pb
from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
from agent.system_prompt import _tool_guidance_block
from hermes_cli import plugins


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.mark.parametrize("task", [None, "", "   "])
@pytest.mark.parametrize("context", ["worker", "delegate", "cron"])
@pytest.mark.parametrize("has_tool", [True, False])
def test_load_tools_only_caches_guidance_for_a_real_worker(
    monkeypatch, task, context, has_tool, isolated_hermes_home
):
    if task is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    scope = {
        "worker": nullcontext(),
        "delegate": delegated_child_context(),
        "cron": non_dispatcher_owned_context(),
    }[context]
    tools = ([{"type": "function", "function": {"name": "kanban_show"}}]
             if has_tool else [])
    agent = type("Agent", (), {
        "quiet_mode": True,
        "save_trajectories": False,
        "ephemeral_system_prompt": "",
        "_use_prompt_caching": False,
        "_use_native_cache_layout": False,
        "provider": "",
        "_cache_ttl": "",
    })()
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_: tools)
    with scope:
        agent_init._load_tools(agent, [], [])
    expected = (
        pb.KANBAN_GUIDANCE
        if has_tool and context == "worker" and task is not None and task.strip()
        else ""
    )
    assert agent._kanban_worker_guidance == expected
    assert agent.valid_tool_names == ({"kanban_show"} if has_tool else set())
    assert agent.tools == tools


@pytest.mark.parametrize("task", [None, "", "   ", "task-1"])
def test_tool_guidance_fallback_is_session_static_without_cached_attribute(
    monkeypatch, task, isolated_hermes_home
):
    if task is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    agent = type("Agent", (), {"valid_tool_names": {"kanban_show"}})()
    with nullcontext():
        first = _tool_guidance_block(agent)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "changed-after-assembly")
    with non_dispatcher_owned_context():
        second = _tool_guidance_block(agent)
    expected = pb.KANBAN_GUIDANCE if task is not None and task.strip() else None
    assert first == second == expected

    agent = type("Agent", (), {"valid_tool_names": {"kanban_show"}})()
    with non_dispatcher_owned_context():
        first = _tool_guidance_block(agent)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "changed-after-assembly")
    with nullcontext():
        second = _tool_guidance_block(agent)
    assert first == second == None
