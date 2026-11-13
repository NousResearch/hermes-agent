"""``pre_llm_call`` must tell plugins which PROVIDER is acting.

Under the MoA virtual provider ``agent.model`` holds a PRESET name (see
``agent/agent_init.py``: "AI Agent initialized with MoA preset"), not a real
model identity. A plugin that classifies the host model by name therefore sees
something like ``default``, which matches no model family.

The name alone cannot disambiguate -- a provider may legitimately ship a model
called ``default`` -- so the dispatch must also forward ``provider``. Without
it a cross-family review gate cannot tell "MoA preset" from "ordinary model"
and silently mis-gates every MoA turn.

These tests exercise the real ``_collect_pre_llm_call_context`` and assert on
the kwargs the hook actually receives.
"""

from __future__ import annotations

import types
from unittest.mock import patch

from agent.turn_context import _collect_pre_llm_call_context


def _agent(model="claude-opus-5", provider="anthropic"):
    return types.SimpleNamespace(
        session_id="sess-1",
        model=model,
        provider=provider,
        platform="cli",
        _parent_session_id=None,
        _user_id=None,
    )


def _collect(agent):
    """Run the real collector, returning the kwargs the hook received."""
    seen = {}

    def fake_invoke_hook(name, **kwargs):
        seen["name"] = name
        seen["kwargs"] = kwargs
        return []

    lifecycle = types.ModuleType("hermes_cli.lifecycle")
    lifecycle.invoke_hook = fake_invoke_hook  # type: ignore[attr-defined]

    with patch.dict("sys.modules", {"hermes_cli.lifecycle": lifecycle}):
        _collect_pre_llm_call_context(
            agent,
            effective_task_id="task-1",
            turn_id="turn-1",
            original_user_message="hello",
            messages=[{"role": "user", "content": "hello"}],
            conversation_history=None,
        )
    return seen


def test_pre_llm_call_receives_the_host_provider():
    seen = _collect(_agent())
    assert seen["name"] == "pre_llm_call"
    assert seen["kwargs"]["provider"] == "anthropic"
    assert seen["kwargs"]["model"] == "claude-opus-5"


def test_moa_turn_forwards_the_virtual_provider_with_the_preset_name():
    """The pairing is what makes a MoA turn identifiable.

    ``model`` is the preset and ``provider`` is ``moa``; a plugin needs BOTH
    to resolve the preset to its acting aggregator.
    """
    seen = _collect(_agent(model="default", provider="moa"))
    assert seen["kwargs"]["provider"] == "moa"
    assert seen["kwargs"]["model"] == "default"


def test_absent_provider_is_forwarded_as_an_empty_string():
    """A provider-less agent must not raise or omit the key.

    Hooks read ``kwargs["provider"]``; omitting it on some paths would make
    the contract conditional and reintroduce the ambiguity for those callers.
    """
    agent = _agent()
    del agent.provider
    seen = _collect(agent)
    assert seen["kwargs"]["provider"] == ""


def test_provider_is_forwarded_alongside_the_preexisting_kwargs():
    """Forwarding must be additive -- no existing hook input may be dropped."""
    seen = _collect(_agent())
    for key in (
        "session_id",
        "task_id",
        "turn_id",
        "user_message",
        "conversation_history",
        "is_first_turn",
        "model",
        "platform",
        "parent_session_id",
        "sender_id",
        "provider",
    ):
        assert key in seen["kwargs"], f"pre_llm_call lost the {key!r} kwarg"
