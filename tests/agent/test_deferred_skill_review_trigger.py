"""Background skill review stays eligible when ``skill_manage`` is deferred behind tool search.

The skill trigger must answer "can this session reach skill_manage?" from the session's own
assembly (directly exposed, or ``tool_call`` + the session-scoped bridge catalog), never from the
model-visible schema alone. Agents are built for real under the test ``HERMES_HOME``; only the
provider client, the codex app-server turn and the review-thread spawn are controlled.
"""
from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import get_hermes_home
from run_agent import AIAgent

_DEFER_SKILL_MANAGE = "tools:\n  tool_search:\n    defer: [skill_manage]\n"


@pytest.fixture(autouse=True)
def _no_background_titles(monkeypatch):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)


def _response(content="", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    finish = "tool_calls" if tool_calls else "stop"
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                           model="test/model", usage=None)


def _build_agent(config_yaml: str, provider_client, **toolsets) -> AIAgent:
    (get_hermes_home() / "config.yaml").write_text(config_yaml, encoding="utf-8")
    with patch("agent.process_bootstrap.OpenAI", return_value=provider_client):
        agent = AIAgent(
            model="test/model", api_key="test-key", base_url="http://127.0.0.1:9/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli", **toolsets,
        )
    agent._persist_session = MagicMock()
    agent._save_trajectory = MagicMock()
    return agent


@pytest.mark.parametrize("runtime", ["chat_completions", "codex_app_server"])
@pytest.mark.parametrize(
    ("config_yaml", "toolsets", "expect_review"),
    [
        pytest.param("", {}, True, id="directly-exposed"),
        pytest.param(_DEFER_SKILL_MANAGE, {}, True, id="deferred-enabled"),
        pytest.param(_DEFER_SKILL_MANAGE, {"disabled_toolsets": ["skills"]}, False, id="skills-disabled"),
        pytest.param(_DEFER_SKILL_MANAGE, {"enabled_toolsets": ["terminal", "file"]}, False,
                     id="bridge-and-skills-absent"),
    ],
)
def test_skill_review_trigger_follows_session_reachability(runtime, config_yaml, toolsets, expect_review):
    client = MagicMock()
    client.chat.completions.create.return_value = _response(content="done")
    agent = _build_agent(config_yaml, client, **toolsets)
    agent._spawn_background_review = MagicMock()
    tools_before = copy.deepcopy(agent.tools)
    exposed_before = set(agent.valid_tool_names)
    agent._skill_nudge_interval = 1

    if runtime == "chat_completions":
        agent._iters_since_skill = 1
        agent.run_conversation("hello")
    else:
        from agent.codex_runtime import _finish_codex_turn

        agent._iters_since_skill = 0
        turn = SimpleNamespace(tool_iterations=1, interrupted=False, error=None, final_text="done")
        with patch("agent.codex_runtime._record_codex_app_server_compaction"), \
                patch("agent.codex_runtime._record_codex_app_server_usage", return_value={}):
            _finish_codex_turn(agent, turn, [{"role": "assistant", "content": "done"}],
                               original_user_message="hello", should_review_memory=False)

    if expect_review:
        agent._spawn_background_review.assert_called_once()
        assert agent._spawn_background_review.call_args.kwargs["review_skills"] is True
    else:
        agent._spawn_background_review.assert_not_called()
    # Eligibility is evaluated without widening what the model sees.
    assert agent.tools == tools_before
    assert set(agent.valid_tool_names) == exposed_before


def test_deferred_skill_review_writes_through_bridge():
    skill_md = ("---\nname: deferred-bridge-proof\ndescription: Proof that review writes reach the bridge.\n"
                "---\n\n# Deferred bridge proof\n\nWritten by the background review via tool_call.\n")
    bridge_call = SimpleNamespace(id="call_bridge", type="function", function=SimpleNamespace(
        name="tool_call", arguments=json.dumps({"calls": [{"name": "skill_manage", "arguments": {"operations": [
            {"action": "create", "name": "deferred-bridge-proof", "content": skill_md}]}}]})))
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(content="done"),  # parent turn
        _response(tool_calls=[bridge_call]),  # review fork: create via the bridge
        _response(content="Created the skill."),
    ]
    agent = _build_agent(_DEFER_SKILL_MANAGE, client)
    assert "skill_manage" not in agent.valid_tool_names
    agent._skill_nudge_interval = 1
    agent._iters_since_skill = 1

    # The review fork builds its own client: keep the provider boundary controlled for it too.
    with patch("agent.process_bootstrap.OpenAI", return_value=client):
        agent.run_conversation("hello")
        for thread in threading.enumerate():
            if thread.name == "bg-review":
                thread.join(timeout=30)

    written = get_hermes_home() / "skills" / "deferred-bridge-proof" / "SKILL.md"
    assert written.read_text(encoding="utf-8") == skill_md
