"""A task stopped by a system producer is not booked as the user's cancel.

Delegated children are stopped by their parent ("parent delegation ended", "Async delegation
cancelled", ...) and turns by watchdogs or a gateway shutdown. The agent records the issuer in the
turn exit reason (#112647); the shared-metrics task terminal must keep that distinction instead of
counting every interrupted turn as ``user_cancelled``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from agent.interrupt_control import interrupted_during_api_call_reason
from agent.turn_iteration_prep import begin_iteration
from hermes_cli.observability.shared_metrics_contract import task_terminal_state

_SYSTEM_STOP = ("cancelled", "system_aborted", "system_aborted")
_USER_STOP = ("cancelled", "user_cancelled", "user_cancelled")

# tool_reason categories as the producers pass them: delegate_tool_child_run / delegate_tool_dispatch,
# gateway.run, the turn liveness watchdog. ``None`` and the interrupt() categories are human stops.
_CASES = [
    ("parent delegation ended", _SYSTEM_STOP),
    ("parent agent interrupted", _SYSTEM_STOP),
    ("gateway shutdown", _SYSTEM_STOP),
    ("turn liveness watchdog", _SYSTEM_STOP),
    ("explicit stop requested", _USER_STOP),
    ("user sent a new message", _USER_STOP),
    ("user interrupt", _USER_STOP),
    (None, _USER_STOP),
]


def _agent(tool_interrupt_reason):
    return SimpleNamespace(
        _interrupt_requested=True, _tool_interrupt_reason=tool_interrupt_reason, quiet_mode=True,
        _drain_pending_redirect=lambda: None, _checkpoint_mgr=SimpleNamespace(new_turn=lambda: None),
    )


@pytest.mark.parametrize("tool_interrupt_reason, expected", _CASES)
def test_stop_between_iterations_keeps_its_issuer(tool_interrupt_reason, expected):
    verdict = begin_iteration(
        _agent(tool_interrupt_reason), messages=[], conversation_history=[], original_user_message="hi",
        api_call_count=0, interrupted=False, _turn_exit_reason="unknown",
    )
    assert verdict.interrupted is True
    event = {"interrupted": True, "turn_exit_reason": verdict._turn_exit_reason}
    assert task_terminal_state(event) == expected


@pytest.mark.parametrize("tool_interrupt_reason, expected", _CASES)
def test_stop_during_api_call_keeps_its_issuer(tool_interrupt_reason, expected):
    reason = interrupted_during_api_call_reason(_agent(tool_interrupt_reason))
    assert task_terminal_state({"interrupted": True, "turn_exit_reason": reason}) == expected


def test_interrupt_flag_without_a_named_issuer_stays_a_user_stop():
    # The CLI shutdown path fires on_session_end with interrupted=True and no exit reason.
    assert task_terminal_state({"interrupted": True}) == _USER_STOP


def _run_stopped_turn(monkeypatch, stop):
    """Run one real AIAgent turn against a hanging loopback model, stop it with ``stop(agent)``
    once the request is in flight, and return the ``on_session_end`` payload the turn fired."""
    from agent import turn_finalizer
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import FakeLLMServer, Hang

    fired: list[dict] = []
    real_invoke = turn_finalizer._invoke_hook_safely

    def _record(name, logger, **kwargs):
        if name == "on_session_end":
            fired.append(kwargs)
        return real_invoke(name, logger, **kwargs)

    monkeypatch.setattr(turn_finalizer, "_invoke_hook_safely", _record)
    with FakeLLMServer([Hang(seconds=30)]) as server:
        agent = AIAgent(
            base_url=server.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["todo"],
        )
        worker = threading.Thread(target=agent.run_conversation, args=("work on it",), daemon=True)
        try:
            worker.start()
            server.wait_for_requests(1, timeout=20)
            stop(agent)
            worker.join(timeout=30)
            assert not worker.is_alive()
        finally:
            agent.close()
    [payload] = fired
    return payload


def test_a_child_stopped_by_its_parent_is_not_a_user_cancel(monkeypatch):
    from tools.delegate_tool_child_run import _signal_child_stop

    payload = _run_stopped_turn(monkeypatch, lambda agent: _signal_child_stop(agent, "Async delegation cancelled"))
    assert payload["interrupted"] is True
    assert task_terminal_state(payload) == _SYSTEM_STOP


def test_a_user_stop_is_still_a_user_cancel(monkeypatch):
    payload = _run_stopped_turn(monkeypatch, lambda agent: agent.hard_interrupt())
    assert payload["interrupted"] is True
    assert task_terminal_state(payload) == _USER_STOP
