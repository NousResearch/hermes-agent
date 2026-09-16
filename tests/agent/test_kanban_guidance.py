from types import SimpleNamespace

import pytest

from agent import prompt_builder as pb
from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
from agent.system_prompt import _tool_guidance_block


@pytest.mark.parametrize("task", [None, "", "   "])
def test_guidance_requires_nonempty_task_id(monkeypatch, task):
    if task is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    assert pb.resolve_kanban_worker_guidance({"kanban_show"}) == ""


@pytest.mark.parametrize("context", ["delegate", "cron"])
def test_guidance_excludes_non_dispatcher_contexts(monkeypatch, context):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    scope = delegated_child_context() if context == "delegate" else non_dispatcher_owned_context()
    with scope:
        assert pb.resolve_kanban_worker_guidance({"kanban_show"}) == ""


def test_guidance_requires_tool_and_fallback_choice_is_cached(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    assert pb.resolve_kanban_worker_guidance(set()) == ""
    guidance = pb.resolve_kanban_worker_guidance({"kanban_show"})
    assert guidance == pb.KANBAN_GUIDANCE

    agent = SimpleNamespace(valid_tool_names={"kanban_show"}, _kanban_worker_guidance=guidance)
    first = _tool_guidance_block(agent)
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert _tool_guidance_block(agent) == first
