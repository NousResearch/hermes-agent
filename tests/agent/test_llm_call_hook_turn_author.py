"""``pre_llm_call`` and ``post_llm_call`` carry the turn author, through real plugin dispatch."""

import logging
from types import SimpleNamespace

import pytest

import hermes_cli.plugins as plugins_mod
from agent.turn_context import _collect_pre_llm_call_context
from agent.turn_finalizer import _apply_output_hooks
from hermes_cli.plugins import PluginManager

AUTHOR = {"id": "bot:helper", "name": "Helper", "is_bot": True}
HOOKS = ("pre_llm_call", "post_llm_call")


def _agent(author):
    return SimpleNamespace(
        _parent_session_id=None, _persist_disabled=False, _turn_author=author, _user_id="",
        model="test/model", platform="telegram", session_id="session-1",
    )


@pytest.fixture
def manager(monkeypatch):
    mgr = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
    return mgr


def _run_turn_hooks(agent):
    messages = [{"role": "user", "content": "hello"}]
    _collect_pre_llm_call_context(
        agent, effective_task_id="task-1", turn_id="turn-1", original_user_message="hello",
        messages=messages, conversation_history=None,
    )
    _apply_output_hooks(
        agent, "response", logging.getLogger(__name__), platform="telegram",
        effective_task_id="task-1", turn_id="turn-1", original_user_message="hello", messages=messages,
    )


@pytest.mark.parametrize("author", [AUTHOR, None], ids=["author", "no-author"])
def test_llm_call_hooks_receive_the_turn_author(manager, author):
    seen = {}
    for hook in HOOKS:
        manager._hooks[hook] = [lambda hook=hook, **kwargs: seen.__setitem__(hook, kwargs.get("turn_author", "absent"))]

    _run_turn_hooks(_agent(dict(author) if author else None))

    assert seen == {hook: author for hook in HOOKS}


def test_a_hook_cannot_rewrite_the_turn_author_memory_providers_receive(manager):
    received = []

    def rewrite(**kwargs):
        author = kwargs.get("turn_author")
        received.append(dict(author) if author else author)
        if author:
            author["name"] = "Someone else"

    for hook in HOOKS:
        manager._hooks[hook] = [rewrite]
    agent = _agent(dict(AUTHOR))

    _run_turn_hooks(agent)

    assert received == [AUTHOR, AUTHOR]
    assert agent._turn_author == AUTHOR
