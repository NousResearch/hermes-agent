"""Worker instructions require task ownership, not merely Kanban tool access."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from agent.agent_init import _load_tools
from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
from agent.prompt_builder import KANBAN_GUIDANCE
from agent.system_prompt import _tool_guidance_block


@pytest.mark.parametrize("fallback", [False, True], ids=["init", "fallback"])
@pytest.mark.parametrize("task", [None, "", "test-worker-task"])
def test_kanban_tool_access_does_not_imply_worker_identity(monkeypatch, task, fallback):
    if task is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", task)

    # Exercise real tool discovery under the per-test HERMES_HOME sandbox.
    agent = SimpleNamespace(quiet_mode=True)
    _load_tools(agent, enabled_toolsets=["kanban"], disabled_toolsets=None)
    assert "kanban_show" in agent.valid_tool_names
    if fallback:
        del agent._kanban_worker_guidance

    guidance = _tool_guidance_block(agent) or ""
    assert (KANBAN_GUIDANCE in guidance) == bool(task)
    # Suppressing worker instructions must not remove interactive board access.
    assert "kanban_show" in agent.valid_tool_names


@pytest.mark.parametrize("fallback", [False, True], ids=["init", "fallback"])
@pytest.mark.parametrize("context", [non_dispatcher_owned_context, delegated_child_context])
def test_inherited_task_does_not_assign_worker_instructions(monkeypatch, context, fallback):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "parent-worker-task")
    with context():
        agent = SimpleNamespace(quiet_mode=True)
        _load_tools(agent, enabled_toolsets=["kanban"], disabled_toolsets=None)
        if context is non_dispatcher_owned_context:
            # Cron can opt into Kanban tools without owning the inherited task.
            assert "kanban_show" in agent.valid_tool_names
        if fallback:
            del agent._kanban_worker_guidance
        assert KANBAN_GUIDANCE not in (_tool_guidance_block(agent) or "")


@pytest.mark.parametrize("worker", [False, True])
def test_worker_guidance_snapshot_survives_environment_changes(monkeypatch, worker):
    if worker:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "test-worker-task")
    else:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    agent = SimpleNamespace(quiet_mode=True)
    _load_tools(agent, enabled_toolsets=["kanban"], disabled_toolsets=None)
    before = _tool_guidance_block(agent)
    assert (KANBAN_GUIDANCE in (before or "")) == worker

    if worker:
        monkeypatch.delenv("HERMES_KANBAN_TASK")
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "other-worker-task")
    with non_dispatcher_owned_context() if worker else nullcontext():
        assert _tool_guidance_block(agent) == before


def test_task_without_kanban_tool_has_no_worker_guidance(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "test-worker-task")
    agent = SimpleNamespace(valid_tool_names=set())
    assert KANBAN_GUIDANCE not in (_tool_guidance_block(agent) or "")
