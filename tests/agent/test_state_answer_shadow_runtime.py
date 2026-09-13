from typing import Any

from agent.state_answer_shadow import ReceiptStatus, TerminalStatus
from agent.state_answer_shadow_runtime import (
    finish_runtime_shadow,
    settle_runtime_shadow,
    start_runtime_shadow,
)


class Collector:
    def __init__(self):
        self.initial = []
        self.terminal = []

    def enqueue_initial(self, event):
        self.initial.append(event)

    def enqueue_terminal(self, event_id, update):
        self.terminal.append((event_id, update))


class Agent:
    _state_answer_shadow_collector: Any = None


def _wait_for(predicate):
    import time

    for _ in range(100):
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_runtime_shadow_is_noop_without_explicit_collector():
    agent = Agent()
    assert start_runtime_shadow(agent) is None


def test_runtime_shadow_uses_explicit_unavailable_provider():
    agent = Agent()
    collector = Collector()
    agent._state_answer_shadow_collector = collector

    event_id = start_runtime_shadow(agent)
    assert isinstance(event_id, str)
    _wait_for(lambda: len(collector.initial) == 1)
    event = collector.initial[0]
    assert event["input_status"] == "input_unavailable"
    assert event["current_state_status"] == "unavailable"
    assert event["evidence_status"] == "unavailable"
    assert event["shadow_decision"] == "not_evaluated"


def test_runtime_shadow_terminal_uses_positive_receipt_only_after_completed_persisted_turn():
    agent = Agent()
    collector = Collector()
    agent._state_answer_shadow_collector = collector
    event_id = start_runtime_shadow(agent)
    _wait_for(lambda: len(collector.initial) == 1)

    finish_runtime_shadow(
        agent,
        event_id,
        {"api_calls": 1, "failed": False, "interrupted": False, "persistence_confirmed": True},
    )
    _wait_for(lambda: len(collector.terminal) == 1)
    assert collector.terminal[0][1]["terminal_status"] == TerminalStatus.COMPLETED.value
    assert collector.terminal[0][1]["persistence_receipt_status"] == ReceiptStatus.TRUE.value


def test_runtime_shadow_failed_and_unconfirmed_turn_is_not_positive():
    agent = Agent()
    collector = Collector()
    agent._state_answer_shadow_collector = collector
    event_id = start_runtime_shadow(agent)
    _wait_for(lambda: len(collector.initial) == 1)

    finish_runtime_shadow(
        agent,
        event_id,
        {"api_calls": 1, "failed": True, "interrupted": False, "persistence_confirmed": False},
    )
    _wait_for(lambda: len(collector.terminal) == 1)
    update = collector.terminal[0][1]
    assert update["terminal_status"] == TerminalStatus.EXCEPTION.value
    assert update["persistence_receipt_status"] == ReceiptStatus.FALSE.value


def test_runtime_shadow_interrupt_without_persistence_is_unobserved_receipt():
    agent = Agent()
    collector = Collector()
    agent._state_answer_shadow_collector = collector
    event_id = start_runtime_shadow(agent)
    _wait_for(lambda: len(collector.initial) == 1)

    finish_runtime_shadow(
        agent,
        event_id,
        {"api_calls": 0, "failed": False, "interrupted": True, "persistence_confirmed": None},
    )
    _wait_for(lambda: len(collector.terminal) == 1)
    update = collector.terminal[0][1]
    assert update["terminal_status"] == TerminalStatus.INTERRUPTED.value
    assert update["model_call_status"] == "not_called"
    assert update["persistence_receipt_status"] == ReceiptStatus.UNOBSERVED.value


def test_settlement_classifies_early_results_and_preserves_identity():
    for reason, expected in (
        ("preflight_return", TerminalStatus.PREFLIGHT_RETURN.value),
        ("retry_exhausted", TerminalStatus.RETRY_EXHAUSTED.value),
        ("provider_error", TerminalStatus.PROVIDER_ERROR.value),
    ):
        agent = Agent()
        collector = Collector()
        agent._state_answer_shadow_collector = collector
        event_id = start_runtime_shadow(agent)
        _wait_for(lambda: len(collector.initial) == 1)
        result = {"turn_exit_reason": reason, "api_calls": 0, "failed": True}
        assert settle_runtime_shadow(agent, event_id, result) is result
        _wait_for(lambda: len(collector.terminal) == 1)
        assert collector.terminal[0][1]["terminal_status"] == expected
        assert collector.terminal[0][1]["persistence_receipt_status"] == ReceiptStatus.UNOBSERVED.value
