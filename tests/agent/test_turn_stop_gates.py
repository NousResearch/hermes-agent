"""pre_finish stop-gate: fires after verify gates, including turns with no file edits."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from agent.turn_stop_gates import apply_stop_gates


def _agent(**extra):
    ns = SimpleNamespace(
        session_id="s",
        platform="cli",
        model="m",
        _turn_file_mutation_paths=set(),
        _pre_finish_nudges=0,
        _pre_verify_nudges=0,
        _verification_stop_nudges=0,
        _kanban_stop_nudges=0,
        _session_messages=None,
        _resolved_is_coding=False,
        _emit_interim_assistant_message=lambda *_a, **_k: None,
        _flush_messages_to_session_db=lambda *_a, **_k: None,
        _interim_content_was_streamed=lambda *_a, **_k: False,
        _emit_status=lambda *_a, **_k: None,
    )
    for key, value in extra.items():
        setattr(ns, key, value)
    return ns


def _apply(agent, messages=None, final_response="done"):
    messages = [] if messages is None else messages
    return apply_stop_gates(
        agent,
        {"role": "assistant", "content": final_response},
        final_response=final_response,
        messages=messages,
        conversation_history=[],
        pending_verification_response=None,
        pending_verification_response_previewed=False,
    ), messages


def _gate_stack(*, finish=None, verify=None, has=None, finish_bound=3, verify_bound=3):
    names = set(has) if has is not None else {"pre_finish"}
    stack = ExitStack()
    stack.enter_context(patch("agent.verification_stop.verify_on_stop_enabled", return_value=False))
    stack.enter_context(patch("agent.kanban_stop.build_kanban_stop_nudge", return_value=None))
    stack.enter_context(patch("hermes_cli.lifecycle.has_hook", side_effect=lambda name: name in names))
    finish_mock = stack.enter_context(
        patch("hermes_cli.plugins.get_pre_finish_continue_message", return_value=finish)
    )
    stack.enter_context(
        patch("hermes_cli.plugins.get_pre_verify_continue_message", return_value=verify)
    )
    stack.enter_context(patch("agent.verify_hooks.max_finish_nudges", return_value=finish_bound))
    stack.enter_context(patch("agent.verify_hooks.max_verify_nudges", return_value=verify_bound))
    return stack, finish_mock


def test_pre_finish_continues_when_nothing_was_edited():
    agent = _agent()
    stack, _finish = _gate_stack(finish="confirm the remote write")
    with stack:
        verdict, messages = _apply(agent)

    assert verdict.continue_turn is True
    assert agent._pre_finish_nudges == 1
    assert messages[-1]["content"] == "confirm the remote write"
    assert messages[-1]["_pre_finish_synthetic"] is True


def test_pre_finish_runs_after_pre_verify_declines_on_an_edited_turn():
    agent = _agent(_turn_file_mutation_paths={"config.yaml"})
    stack, _finish = _gate_stack(
        finish="check the deploy", verify=None, has={"pre_verify", "pre_finish"}
    )
    with stack:
        verdict, messages = _apply(agent)

    assert verdict.continue_turn is True
    assert messages[-1]["_pre_finish_synthetic"] is True
    assert agent._pre_verify_nudges == 0


def test_pre_verify_continue_skips_pre_finish():
    agent = _agent(_turn_file_mutation_paths={"app.py"})
    stack, finish_mock = _gate_stack(
        finish="should not run", verify="run tests", has={"pre_verify", "pre_finish"}
    )
    with stack:
        verdict, messages = _apply(agent)

    assert verdict.continue_turn is True
    assert messages[-1]["_pre_verify_synthetic"] is True
    finish_mock.assert_not_called()
    assert agent._pre_finish_nudges == 0


def test_pre_finish_bound_stops_further_continues():
    agent = _agent(_pre_finish_nudges=2)
    stack, _finish = _gate_stack(finish="again", finish_bound=2)
    with stack:
        verdict, _messages = _apply(agent)

    assert verdict.continue_turn is False
    assert agent._pre_finish_nudges == 2
