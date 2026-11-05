from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gateway.evaluator_policy import SubprocessEvaluatorPolicy
from gateway.pre_delivery import PreDeliveryGate


EVALUATOR_ADAPTER = Path(__file__).resolve().parents[1] / "fixtures" / "evaluator_policy.py"
EVALUATOR_ADAPTER_CRASH_AFTER_OUTPUT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "evaluator_policy_crash_after_output.py"
)

# NOTE: this file intentionally covers the evaluator-policy bridge and the
# gate in isolation only. It does not test GatewayRunner injection or a real
# GatewayStreamConsumer, because this PR does not wire pre_delivery_gate into
# GatewayRunner or add quarantine/release methods to the stream consumer.
# Those seams, and the integration tests that exercise them, are a follow-up
# PR (tracked from the discussion in #104038).


def _policy():
    return {
        "required_patterns": ["evidence"],
        "forbidden_patterns": [],
    }


def test_subprocess_policy_connects_to_evaluator_pass():
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )
    gate = PreDeliveryGate(mode="strict", policy=policy)

    decision = gate.evaluate_sync(
        final_text="Answer includes evidence.",
        metadata={"turn_id": "bridge-pass-001", "platform": "test"},
    )

    assert decision.allowed is True
    assert decision.status == "passed"
    assert decision.final_text == "Answer includes evidence."


def test_subprocess_policy_connects_to_evaluator_block():
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )
    gate = PreDeliveryGate(mode="strict", policy=policy)

    decision = gate.evaluate_sync(
        final_text="Answer without the marker.",
        metadata={"turn_id": "bridge-block-001", "platform": "test"},
    )

    assert decision.allowed is False
    assert decision.status == "blocked"
    assert decision.final_text is None


def test_subprocess_policy_does_not_use_shell_commands():
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )
    assert policy._command[0] == sys.executable


def test_subprocess_policy_rejects_mismatched_request_id(monkeypatch):
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )

    class Completed:
        returncode = 0
        stdout = '{"schema_version": 1, "request_id": "other-turn", "status": "passed", "allowed": true, "final_text": "ok"}'

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Completed())
    with pytest.raises(RuntimeError, match="mismatched request_id"):
        policy(final_text="Answer includes evidence.", metadata={"turn_id": "turn-1"})


def test_subprocess_policy_rejects_unknown_schema(monkeypatch):
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )

    class Completed:
        returncode = 0
        stdout = '{"schema_version": 999, "request_id": "turn-1", "status": "passed", "allowed": true, "final_text": "ok"}'

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Completed())
    with pytest.raises(RuntimeError, match="unsupported schema_version"):
        policy(final_text="Answer includes evidence.", metadata={"turn_id": "turn-1"})


def test_subprocess_policy_rejects_boolean_schema_version(monkeypatch):
    # In Python, True == 1 and isinstance(True, int) is True, so a naive
    # `!= 1` or `isinstance(x, int)` check silently accepts a JSON `true`
    # schema_version. Reject anything that is not exactly the int 1.
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )

    class Completed:
        returncode = 0
        stdout = '{"schema_version": true, "request_id": "turn-1", "status": "passed", "allowed": true, "final_text": "ok"}'

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Completed())
    with pytest.raises(RuntimeError, match="unsupported schema_version"):
        policy(final_text="Answer includes evidence.", metadata={"turn_id": "turn-1"})


def test_subprocess_policy_requires_turn_identity():
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )
    with pytest.raises(RuntimeError, match="requires a non-empty turn_id"):
        policy(final_text="Answer includes evidence.", metadata={})


def test_subprocess_policy_rejects_nonzero_exit_even_with_pass_shaped_json(monkeypatch):
    # Reviewer-found gap: a crashed process's stdout must never be trusted,
    # even when that stdout happens to be a well-formed "allowed": true
    # payload written before the crash.
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )

    class Completed:
        returncode = 17
        stdout = (
            '{"schema_version": 1, "request_id": "turn-1", "status": "passed", '
            '"allowed": true, "final_text": "unverified"}'
        )

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Completed())
    with pytest.raises(RuntimeError, match="exited with code 17"):
        policy(final_text="Answer includes evidence.", metadata={"turn_id": "turn-1"})


def test_subprocess_policy_real_child_crash_after_output_is_inconclusive():
    # Real-child regression (not mocked): the crash fixture prints a
    # pass-shaped decision and then exits 17. The gate must classify this
    # as inconclusive/blocked, never as an authoritative passed delivery.
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER_CRASH_AFTER_OUTPUT)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )
    gate = PreDeliveryGate(mode="strict", policy=policy)

    decision = gate.evaluate_sync(
        final_text="Answer includes evidence.",
        metadata={"turn_id": "crash-after-output-001", "platform": "test"},
    )

    assert decision.allowed is False
    assert decision.final_text is None
    assert decision.status == "inconclusive"


def test_subprocess_policy_real_child_blocked_result_exits_zero():
    # A "blocked" business decision is a successful, authoritative
    # evaluation and must exit 0 — only process failure is nonzero.
    completed = subprocess.run(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        input=json.dumps({"request_id": "turn-1", "final_text": "no marker here"}),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0

    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )
    decision = PreDeliveryGate(mode="strict", policy=policy).evaluate_sync(
        final_text="no marker here",
        metadata={"turn_id": "turn-1", "platform": "test"},
    )
    assert decision.status == "blocked"
    assert decision.allowed is False
