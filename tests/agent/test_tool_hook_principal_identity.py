"""Principal identity (who is acting) reaches the tool hooks.

A cached gateway agent is shared by every sender of a chat, so ``agent._user_id`` is the
identity of whoever *created* it, not of the current message. The per-message session
ContextVars are the truth; the agent's construction identity is the fallback (cron,
subagents); outside the gateway everything is ``""``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.inline_tool_executors import tool_hook_ids
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from model_tools import handle_function_call

_PRINCIPAL_KEYS = ("user_id", "user_name", "platform", "chat_id", "chat_type", "thread_id", "gateway_session_key")


@pytest.fixture(autouse=True)
def _unbound_session_vars():
    reset_session_vars()
    yield
    reset_session_vars()


def _stale_agent():
    return SimpleNamespace(
        session_id="sess", _current_turn_id="turn", _current_api_request_id="req",
        _user_id="111", _user_name="owner", platform="discord", _chat_id="chat-1",
        _chat_type="channel", _thread_id="", _gateway_session_key="agent:main:discord:channel:chat-1",
    )


def test_session_context_beats_agent_construction_identity():
    agent = _stale_agent()
    set_session_vars(platform="discord", chat_id="chat-1", chat_type="channel", user_id="222",
                     user_name="spouse", session_key="agent:main:discord:channel:chat-1")
    ids = tool_hook_ids(agent, "task-1", "call-1")

    assert ids["user_id"] == "222"
    assert ids["user_name"] == "spouse"
    assert ids["platform"] == "discord"
    assert ids["gateway_session_key"] == "agent:main:discord:channel:chat-1"
    # Existing ids keep flowing next to the principal fields.
    assert ids["task_id"] == "task-1" and ids["tool_call_id"] == "call-1" and ids["session_id"] == "sess"


def test_agent_identity_is_the_fallback_when_session_context_is_cleared():
    agent = _stale_agent()
    tokens = set_session_vars(platform="discord", chat_id="chat-1", user_id="222")
    clear_session_vars(tokens)

    ids = tool_hook_ids(agent, "task-1", "call-1")

    assert ids["user_id"] == "111"
    assert ids["platform"] == "discord"
    assert ids["chat_id"] == "chat-1"
    assert ids["gateway_session_key"] == "agent:main:discord:channel:chat-1"
    # No agent at all and nothing bound: every key is present and empty (CLI contract).
    assert all(tool_hook_ids(None, "", None)[k] == "" for k in _PRINCIPAL_KEYS)


def test_handle_function_call_delivers_session_identity_to_pre_and_post_hooks():
    set_session_vars(platform="discord", chat_id="chat-1", chat_type="channel", user_id="222",
                     session_key="agent:main:discord:channel:chat-1")
    hook_calls: list[tuple[str, dict]] = []
    with (
        patch("model_tools.registry.dispatch", return_value=json.dumps({"ok": True})),
        patch("hermes_cli.plugins.has_hook", return_value=True),
        patch("hermes_cli.plugins.invoke_hook", side_effect=lambda name, **kw: hook_calls.append((name, kw)) or []),
    ):
        handle_function_call("web_search", {"q": "kita"}, task_id="task-1", tool_call_id="call-1")

    by_hook = {name: kw for name, kw in hook_calls}
    for hook in ("pre_tool_call", "post_tool_call"):
        assert by_hook[hook]["user_id"] == "222", hook
        assert by_hook[hook]["platform"] == "discord", hook
        assert by_hook[hook]["chat_id"] == "chat-1", hook
        assert by_hook[hook]["gateway_session_key"] == "agent:main:discord:channel:chat-1", hook
        assert set(_PRINCIPAL_KEYS) <= set(by_hook[hook]), hook
