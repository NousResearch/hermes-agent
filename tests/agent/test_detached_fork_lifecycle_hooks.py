"""Lifecycle-hook isolation for persistence-disabled internal forks."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.conversation_loop import _restore_or_build_system_prompt
from agent.turn_context import _collect_pre_llm_call_context
from agent.turn_finalizer import _apply_output_hooks


def _agent(*, persist_disabled: bool):
    return SimpleNamespace(
        _cached_system_prompt=None,
        _parent_session_id=None,
        _persist_disabled=persist_disabled,
        _session_db=None,
        model="test/model",
        platform="cli",
        session_id="shared-session",
        _build_system_prompt=Mock(return_value="system prompt"),
    )


def _apply_hooks(agent):
    return _apply_output_hooks(
        agent,
        "response",
        logging.getLogger(__name__),
        platform="cli",
        effective_task_id="task-1",
        turn_id="turn-1",
        original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}],
    )


def test_persist_disabled_fork_skips_session_and_turn_lifecycle_hooks():
    agent = _agent(persist_disabled=True)

    with (
        patch("hermes_cli.lifecycle.invoke_hook") as lifecycle_hook,
        patch("agent.credits_tracker.seed_credits_at_session_start"),
    ):
        _restore_or_build_system_prompt(agent, None, [])
        context = _collect_pre_llm_call_context(
            agent,
            effective_task_id="task-1",
            turn_id="turn-1",
            original_user_message="hello",
            messages=[{"role": "user", "content": "hello"}],
            conversation_history=None,
        )

    output_calls = []

    def invoke_output_hook(name, _logger, **_kwargs):
        output_calls.append(name)
        return ["transformed"] if name == "transform_llm_output" else []

    with patch("agent.turn_finalizer._invoke_hook_safely", side_effect=invoke_output_hook):
        response, transformed, _original = _apply_hooks(agent)

    lifecycle_hook.assert_not_called()
    assert context == ""
    assert output_calls == ["transform_llm_output"]
    assert response == "transformed"
    assert transformed is True


def test_persisted_agent_still_fires_session_and_turn_lifecycle_hooks():
    agent = _agent(persist_disabled=False)

    with (
        patch("hermes_cli.lifecycle.invoke_hook") as lifecycle_hook,
        patch("agent.credits_tracker.seed_credits_at_session_start"),
    ):
        lifecycle_hook.side_effect = lambda name, **_kwargs: (
            [{"context": "plugin context"}] if name == "pre_llm_call" else []
        )
        _restore_or_build_system_prompt(agent, None, [])
        context = _collect_pre_llm_call_context(
            agent,
            effective_task_id="task-1",
            turn_id="turn-1",
            original_user_message="hello",
            messages=[{"role": "user", "content": "hello"}],
            conversation_history=None,
        )

    output_calls = []

    def invoke_output_hook(name, _logger, **_kwargs):
        output_calls.append(name)
        return []

    with patch("agent.turn_finalizer._invoke_hook_safely", side_effect=invoke_output_hook):
        _apply_hooks(agent)

    assert [call.args[0] for call in lifecycle_hook.call_args_list] == [
        "on_session_start",
        "pre_llm_call",
    ]
    assert context == "plugin context"
    assert output_calls == ["transform_llm_output", "post_llm_call"]
