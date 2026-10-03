"""Invariant coverage for assembled Kanban prompt guidance."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    delegated_child_context,
)
from agent.prompt_builder import KANBAN_GUIDANCE, KANBAN_ORCHESTRATOR_GUIDANCE
from agent.system_prompt import build_system_prompt

ORCHESTRATOR_MARKER = KANBAN_ORCHESTRATOR_GUIDANCE.splitlines()[0]
WORKER_MARKER = KANBAN_GUIDANCE.splitlines()[0]


@pytest.fixture(autouse=True)
def _isolated_hermes_home(monkeypatch, tmp_path):
    """Keep prompt construction isolated from the user's real Hermes state."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)


def _make_agent(**overrides):
    """Build the lightweight prompt fixture used by test_system_prompt.py."""
    base = {
        "load_soul_identity": False,
        "skip_context_files": True,
        "valid_tool_names": set(),
        "_task_completion_guidance": False,
        "_parallel_tool_call_guidance": False,
        "_tool_use_enforcement": False,
        "_execution_guidance": False,
        "_environment_probe": False,
        "_bot_mode_protocol": False,
        "_kanban_worker_guidance": "",
        "_kanban_orchestrator_guidance": "",
        "_memory_store": None,
        "_memory_manager": None,
        "model": "",
        "provider": "",
        "platform": "",
        "pass_session_id": False,
        "session_id": "",
        "_emit_status": lambda *_args, **_kwargs: None,
    }
    base.update(overrides)
    base.setdefault("quiet_mode", True)
    return SimpleNamespace(**base)


def _set_kanban_context(monkeypatch, task_id=None):
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
    if task_id is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)


def _load_agent(monkeypatch, *tool_names):
    """Run initialization-time guidance resolution with a real prompt fixture."""
    from agent.agent_init import _load_tools
    import model_tools

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kwargs: [
            {"function": {"name": tool_name}} for tool_name in tool_names
        ],
    )
    agent = _make_agent()

    _load_tools(
        agent,
        enabled_toolsets=["kanban"],
        disabled_toolsets=None,
    )
    return agent


def _assembled_prompt(agent):
    """Return the real assembled prompt with unrelated dynamic inputs removed."""
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch(
            "agent.prompt_builder.build_context_files_prompt",
            return_value="",
        ),
        patch(
            "agent.system_prompt._frozen_plugin_prompt_sections",
            return_value=(),
        ),
    ):
        return build_system_prompt(agent)


@pytest.mark.parametrize(
    ("resolution_path", "task_id", "delegated", "tools", "expected"),
    [
        pytest.param(
            "init", None, False, {"kanban_show"}, "orchestrator",
            id="init-orchestrator",
        ),
        pytest.param(
            "init", "t_worker", False, {"kanban_show"}, "worker",
            id="init-worker",
        ),
        pytest.param(
            "init", "t_parent", True, {"kanban_show"}, "neither",
            id="init-delegated",
        ),
        pytest.param(
            "init", None, False, {"terminal"}, "neither",
            id="init-no-kanban",
        ),
        pytest.param(
            "fallback", None, False, {"kanban_show"}, "orchestrator",
            id="fallback-orchestrator",
        ),
        pytest.param(
            "fallback", "t_worker", False, {"kanban_show"}, "worker",
            id="fallback-worker",
        ),
        pytest.param(
            "fallback", "t_parent", True, {"kanban_show"}, "neither",
            id="fallback-delegated",
        ),
        pytest.param(
            "fallback", None, False, {"terminal"}, "neither",
            id="fallback-no-kanban",
        ),
    ],
)
def test_kanban_guidance_follows_dispatch_ownership(
    monkeypatch,
    resolution_path,
    task_id,
    delegated,
    tools,
    expected,
):
    _set_kanban_context(monkeypatch, task_id=task_id)
    context = delegated_child_context() if delegated else nullcontext()

    with context:
        if resolution_path == "init":
            agent = _load_agent(monkeypatch, *tools)
        else:
            agent = _make_agent(valid_tool_names=set(tools))
            delattr(agent, "_kanban_worker_guidance")
            delattr(agent, "_kanban_orchestrator_guidance")
        prompt = _assembled_prompt(agent)

    expected_markers = {
        "orchestrator": (True, False),
        "worker": (False, True),
        "neither": (False, False),
    }
    assert (
        ORCHESTRATOR_MARKER in prompt,
        WORKER_MARKER in prompt,
    ) == expected_markers[expected]


def test_kanban_guidance_is_frozen_after_first_build(monkeypatch):
    _set_kanban_context(monkeypatch)
    agent = _make_agent(valid_tool_names={"kanban_show"})
    delattr(agent, "_kanban_worker_guidance")
    delattr(agent, "_kanban_orchestrator_guidance")

    first = _assembled_prompt(agent)
    assert ORCHESTRATOR_MARKER in first
    assert WORKER_MARKER not in first

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_changed")
    second = _assembled_prompt(agent)

    assert first == second
    assert agent._kanban_worker_guidance == ""
    assert agent._kanban_orchestrator_guidance == KANBAN_ORCHESTRATOR_GUIDANCE

    disabled_agent = _make_agent(
        valid_tool_names={"kanban_show"},
        _kanban_worker_guidance="",
        _kanban_orchestrator_guidance="",
    )
    disabled_prompt = _assembled_prompt(disabled_agent)

    assert ORCHESTRATOR_MARKER not in disabled_prompt
    assert WORKER_MARKER not in disabled_prompt
