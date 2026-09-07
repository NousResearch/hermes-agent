"""Worker instructions follow task ownership, not opted-in board access."""

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from agent.delegation_context import (
    delegated_child_context,
    non_dispatcher_owned_context,
)
from agent.prompt_builder import KANBAN_GUIDANCE
from agent.system_prompt import build_system_prompt_parts


@pytest.fixture
def board_profile(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "toolsets: [kanban]\nmodel:\n  context_length: 128000\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    import model_tools
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()
    # Use real board schemas. Retaining this snapshot in delegated children
    # also proves ownership is checked even if a caller supplies cached tools.
    schemas = model_tools.get_tool_definitions(enabled_toolsets=["kanban"], quiet_mode=True)
    assert {"kanban_show", "kanban_list"} <= {t["function"]["name"] for t in schemas}
    yield schemas
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()


def _initialized_agent(monkeypatch, schemas=None) -> Any:
    import model_tools
    from run_agent import AIAgent

    if schemas is not None:
        monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: schemas)
    return AIAgent(
        model="test-model",
        provider="custom",
        api_key="test-key",
        base_url="http://127.0.0.1:1/v1",
        enabled_toolsets=["kanban"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )


_CASES = [
    pytest.param(None, nullcontext, True, False, id="interactive-board"),
    pytest.param("", nullcontext, True, False, id="empty-task"),
    pytest.param("t_worker", nullcontext, True, True, id="dispatcher-worker"),
    pytest.param("t_worker", delegated_child_context, True, False, id="delegated-child"),
    pytest.param("t_worker", non_dispatcher_owned_context, True, False, id="in-process-cron"),
    pytest.param("t_worker", nullcontext, False, False, id="worker-without-tools"),
]


def _set_task(monkeypatch, task):
    if task is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", task)


@pytest.mark.parametrize("task,scope,has_tools,expected", _CASES)
def test_initialized_prompt_freezes_worker_ownership(
    monkeypatch, board_profile, task, scope, has_tools, expected
):
    _set_task(monkeypatch, task)
    with scope():
        schemas = [] if not has_tools else board_profile if scope is delegated_child_context else None
        agent = _initialized_agent(monkeypatch, schemas)
        first = build_system_prompt_parts(agent)["stable"]
        assert (KANBAN_GUIDANCE in first) is expected
        assert agent._kanban_worker_guidance == (KANBAN_GUIDANCE if expected else "")
        if has_tools:
            assert "kanban_show" in agent.valid_tool_names
            if not task:
                assert "kanban_list" in agent.valid_tool_names

    # Rebuilds (including compression) retain BOTH positive and negative init
    # snapshots after ambient task identity / ContextVars change.
    _set_task(monkeypatch, None if task else "t_later")
    with non_dispatcher_owned_context():
        assert build_system_prompt_parts(agent)["stable"] == first
    assert build_system_prompt_parts(agent)["stable"] == first


@pytest.mark.parametrize("task,scope,has_tools,expected", _CASES)
def test_legacy_prompt_fallback_obeys_worker_ownership(
    monkeypatch, board_profile, task, scope, has_tools, expected
):
    _set_task(monkeypatch, task)
    with scope():
        schemas = [] if not has_tools else board_profile if scope is delegated_child_context else None
        agent = _initialized_agent(monkeypatch, schemas)
        # Callers bypassing agent_init have no guidance snapshot. Exercise the
        # real prompt assembly, not a copy of the eligibility expression.
        del agent._kanban_worker_guidance
        assert (KANBAN_GUIDANCE in agent._build_system_prompt()) is expected
