from dataclasses import dataclass
import time

from agent.state_answer_shadow import (
    InputStatus,
    ReceiptStatus,
    ShadowInput,
    TerminalStatus,
    observe_state_answer_shadow,
    observe_state_answer_shadow_terminal,
)
from agent.state_candidate_evaluator import CandidateStatus, DeltaType, StateCandidateResult


@dataclass
class Provider:
    value: object

    def get_input(self):
        return self.value


class Collector:
    def __init__(self, fail=False):
        self.initial = []
        self.terminal = []
        self.fail = fail

    def enqueue_initial(self, event):
        if self.fail:
            raise RuntimeError("collector failure")
        self.initial.append(event)

    def enqueue_terminal(self, event_id, update):
        if self.fail:
            raise RuntimeError("collector failure")
        self.terminal.append((event_id, update))


def pending(key="plan", scope="scope-a"):
    return StateCandidateResult(
        candidate_id="candidate-opaque",
        delta_type=DeltaType.REPLACE,
        status=CandidateStatus.PENDING,
        state_key=key,
        scope=scope,
    )


def test_unavailable_is_not_evaluated_and_does_not_call_gate():
    collector = Collector()
    calls = []
    event_id = observe_state_answer_shadow(
        Provider(ShadowInput(InputStatus.INPUT_UNAVAILABLE)),
        collector,
        gate_evaluator=lambda *args, **kwargs: calls.append(1),
    )
    assert calls == []
    assert event_id == collector.initial[0]["shadow_event_id"]
    assert collector.initial[0]["shadow_decision"] == "not_evaluated"
    assert collector.initial[0]["gate_decision"] == "unobserved"


def test_ready_evaluates_all_candidates_and_uses_strongest_policy():
    collector = Collector()
    seen = []
    event_id = observe_state_answer_shadow(
        Provider(ShadowInput(
            InputStatus.READY,
            active_scope="scope-a",
            answer_scope="scope-a",
            candidates=(pending(), pending("other")),
            requested_state_keys=("plan", "other"),
        )),
        collector,
        gate_evaluator=lambda candidate, **kwargs: (
            seen.append((candidate.state_key, kwargs["requested_state_keys"]))
            or type("Gate", (), {"decision": type("Decision", (), {"response_policy": "hold"})()})()
        ),
    )
    assert event_id
    assert [item[0] for item in seen] == ["plan", "other"]
    assert seen[0][1] == ["plan", "other"]
    assert collector.initial[0]["gate_decision"] == "hold"
    assert collector.initial[0]["candidate_count"] == 2


def test_malformed_provider_result_is_not_evaluated():
    collector = Collector()
    observe_state_answer_shadow(Provider(object()), collector)
    assert collector.initial[0]["input_status"] == "provider_error"
    assert collector.initial[0]["shadow_decision"] == "not_evaluated"


def test_provider_exception_is_isolated():
    class Broken:
        def get_input(self):
            raise ValueError("raw secret-like error")

    collector = Collector()
    observe_state_answer_shadow(Broken(), collector)
    assert collector.initial[0]["input_status"] == "provider_error"
    assert "raw secret" not in str(collector.initial[0])


def test_collector_failure_does_not_escape():
    event_id = observe_state_answer_shadow(
        Provider(ShadowInput(InputStatus.DISABLED)), Collector(fail=True)
    )
    assert isinstance(event_id, str)


def test_terminal_receipt_is_three_valued():
    collector = Collector()
    event_id = observe_state_answer_shadow(Provider(ShadowInput(InputStatus.DISABLED)), collector)
    observe_state_answer_shadow_terminal(
        collector, event_id, terminal_status=TerminalStatus.COMPLETED,
        model_call_status="observed", persistence_confirmed=None,
    )
    assert collector.terminal[0][1]["persistence_receipt_status"] == ReceiptStatus.UNOBSERVED.value
    second_event_id = observe_state_answer_shadow(Provider(ShadowInput(InputStatus.DISABLED)), collector)
    observe_state_answer_shadow_terminal(
        collector, second_event_id, terminal_status=TerminalStatus.COMPLETED,
        model_call_status="observed", persistence_confirmed=True,
    )
    assert collector.terminal[1][1]["persistence_receipt_status"] == ReceiptStatus.TRUE.value


def test_invalid_terminal_metadata_is_dropped():
    collector = Collector()
    observe_state_answer_shadow_terminal(
        collector, "0123456789abcdef0123456789abcdef", terminal_status="not-a-terminal",
        model_call_status="observed", persistence_confirmed=True,
    )
    assert collector.terminal == []


def test_terminal_enqueue_failure_can_be_retried():
    initial_collector = Collector()
    event_id = observe_state_answer_shadow(
        Provider(ShadowInput(InputStatus.DISABLED)), initial_collector
    )
    failing = Collector(fail=True)
    observe_state_answer_shadow_terminal(
        failing, event_id, terminal_status=TerminalStatus.COMPLETED,
        model_call_status="observed", persistence_confirmed=True,
    )
    time.sleep(0.02)
    succeeding = Collector()
    observe_state_answer_shadow_terminal(
        succeeding, event_id, terminal_status=TerminalStatus.COMPLETED,
        model_call_status="observed", persistence_confirmed=True,
    )
    assert succeeding.terminal


def test_malformed_candidate_and_projection_are_explicit():
    collector = Collector()
    observe_state_answer_shadow(
        Provider(ShadowInput(
            InputStatus.READY, active_scope="scope-a", answer_scope="scope-a",
            candidates=(object(),), requested_state_keys=("plan",),
        )), collector,
    )
    assert collector.initial[0]["reason_code"] == "malformed_input"

    collector = Collector()
    observe_state_answer_shadow(
        Provider(ShadowInput(
            InputStatus.READY, active_scope="scope-a", answer_scope="scope-a",
            candidates=tuple(pending() for _ in range(33)), requested_state_keys=("plan",),
        )), collector,
    )
    assert collector.initial[0]["reason_code"] == "malformed_input"


def test_terminal_collector_failure_isolated():
    collector = Collector(fail=True)
    event_id = observe_state_answer_shadow(Provider(ShadowInput(InputStatus.DISABLED)), Collector())
    observe_state_answer_shadow_terminal(
        collector, event_id, terminal_status=TerminalStatus.EXCEPTION,
        model_call_status="failed", persistence_confirmed=False,
    )


def test_terminal_observer_rejects_non_opaque_id():
    collector = Collector()
    observe_state_answer_shadow_terminal(
        collector, "user-email=alice@example.test", terminal_status=TerminalStatus.COMPLETED,
        model_call_status="observed", persistence_confirmed=True,
    )
    assert collector.terminal == []


def test_scope_mismatch_is_not_evaluated():
    collector = Collector()
    calls = []
    observe_state_answer_shadow(
        Provider(ShadowInput(
            InputStatus.READY, active_scope="active-scope", answer_scope="answer-scope",
            candidates=(pending("unrelated", "answer-scope"),), requested_state_keys=("other",),
        )),
        collector, gate_evaluator=lambda *args, **kwargs: calls.append(1),
    )
    assert calls == []
    assert collector.initial[0]["shadow_decision"] == "not_evaluated"
    assert collector.initial[0]["reason_code"] == "scope_mismatch"


def test_wire_format_ready_status_is_evaluated():
    collector = Collector()
    calls = []
    observe_state_answer_shadow(
        Provider(ShadowInput(
            "ready", active_scope="scope-a", answer_scope="scope-a",
            candidates=(), requested_state_keys=(),
        )),
        collector, gate_evaluator=lambda *args, **kwargs: calls.append(1),
    )
    assert calls == []
    assert collector.initial[0]["shadow_decision"] == "evaluated"


def test_event_contains_only_allowlisted_metadata():
    collector = Collector()
    observe_state_answer_shadow(Provider(ShadowInput(InputStatus.DISABLED)), collector)
    event = collector.initial[0]
    assert set(event) <= {
        "schema_version", "shadow_event_id", "origin", "mode", "input_status",
        "shadow_decision", "gate_decision", "model_call_status", "terminal_status",
        "persistence_receipt_status", "reason_code", "candidate_count", "run_id", "turn_id",
        "created_at", "candidate_outcomes", "active_scope_status", "current_state_status",
        "evidence_status", "candidate_status",
    }


def test_true_receipt_requires_completed_observed():
    collector = Collector()
    event_id = observe_state_answer_shadow(Provider(ShadowInput(InputStatus.DISABLED)), collector)
    observe_state_answer_shadow_terminal(
        collector, event_id, terminal_status=TerminalStatus.PREFLIGHT_RETURN,
        model_call_status="not_called", persistence_confirmed=True,
    )
    assert collector.terminal[0][1]["persistence_receipt_status"] == ReceiptStatus.UNOBSERVED.value


def test_unvalidated_state_and_evidence_are_not_available():
    collector = Collector()
    observe_state_answer_shadow(
        Provider(ShadowInput(InputStatus.DISABLED, current_state=object(), evidence=object())), collector
    )
    event = collector.initial[0]
    assert event["current_state_status"] == "unavailable"
    assert event["evidence_status"] == "unavailable"
