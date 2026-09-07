"""Two senders sharing one cached agent: each tool hook sees ITS sender's identity.

The sequential path fires the hooks on the turn thread; the concurrent path fires them in
pool workers (``propagate_context_to_thread`` snapshots the turn's ContextVars). Both must
report the per-message session identity, never the stale construction identity of the
shared agent, and never the other sender's.
"""

from __future__ import annotations

import contextvars
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_executor import execute_tool_calls_concurrent, execute_tool_calls_sequential
from gateway.session_context import reset_session_vars, set_session_vars
from hermes_cli.plugins import PluginManager
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _unbound_session_vars():
    reset_session_vars()
    yield
    reset_session_vars()


def _make_agent(tmp_path: Path) -> AIAgent:
    with (
        patch(
            "model_tools.get_tool_definitions",
            return_value=[{
                "type": "function",
                "function": {"name": "web_extract", "description": "test tool",
                             "parameters": {"type": "object", "properties": {}}},
            }],
        ),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", tmp_path),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True, platform="discord", user_id="000",
            chat_id="chat-1", chat_type="channel", gateway_session_key="agent:main:discord:channel:chat-1",
        )
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._append_guardrail_observation = MagicMock(side_effect=lambda _name, _args, result, **_kwargs: result)
    agent._record_file_mutation_result = MagicMock()
    agent._subdirectory_hints.check_tool_call = MagicMock(return_value="")
    agent._tool_result_content_for_active_model = MagicMock(side_effect=lambda _name, result: result)
    return agent


def _tool_call(call_id: str):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name="web_extract", arguments="{}"))


def _as_sender(user_id: str, run):
    """Run *run* in a fresh context bound to one sender (what one gateway message does)."""
    def _bound():
        set_session_vars(platform="discord", chat_id="chat-1", chat_type="channel", user_id=user_id,
                         session_key="agent:main:discord:channel:chat-1")
        return run()
    return contextvars.copy_context().run(_bound)


def test_each_sender_sees_own_user_id_on_sequential_and_concurrent_paths(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    pre_seen: dict[str, str] = {}
    post_seen: dict[str, str] = {}

    manager = PluginManager()
    manager._discovered = True
    manager._hooks = {
        "pre_tool_call": [lambda tool_name, args, tool_call_id="", user_id="", **kw: pre_seen.__setitem__(tool_call_id, user_id)],
        "post_tool_call": [lambda tool_name, args, result, tool_call_id="", user_id="", **kw: post_seen.__setitem__(tool_call_id, user_id)],
    }
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    # Registry dispatch is stubbed (not handle_function_call): the concurrent path relies on
    # handle_function_call's own post_tool_call emission, the sequential path on the executor's.
    with patch("model_tools.registry.dispatch", return_value=json.dumps({"ok": True})):
        _as_sender("111", lambda: execute_tool_calls_sequential(
            agent, SimpleNamespace(tool_calls=[_tool_call("seq-1")]), [], "task"))
        _as_sender("222", lambda: execute_tool_calls_concurrent(
            agent, SimpleNamespace(tool_calls=[_tool_call("conc-1")]), [], "task"))

    # One call per concurrent batch on purpose: the hook dispatcher skips a callback that is
    # still running for a sibling call, so two pool workers sharing one lambda can drop an id
    # under load. Identity propagation into the pool is what this pins, not batch fan-out.
    assert pre_seen == {"seq-1": "111", "conc-1": "222"}
    assert post_seen == {"seq-1": "111", "conc-1": "222"}
    # The shared agent's construction identity ("000") never leaked into a hook.
    assert "000" not in set(pre_seen.values()) | set(post_seen.values())
