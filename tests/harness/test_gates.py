"""Gate contracts: budgets, verification hierarchy, recovery decisions."""

import time

from harness.budget import BudgetGovernor
from harness.knowledge import extract, is_durable, resolve_conflict, KnowledgeCandidate
from harness.recovery import (
    FailureClass,
    Strategy,
    classify_failure,
    decide,
    progress_made,
)
from harness.state import ExecutionBudget, KnowledgeItem, VerificationCheck
from harness.verify import (
    CheckStrength,
    completion_allowed,
    file_contains_check,
    verify,
)


def test_budget_hard_stop():
    gov = BudgetGovernor(ExecutionBudget(max_tool_calls=2, max_retries=1))
    assert gov.consume_tool_call() and gov.consume_tool_call()
    assert not gov.consume_tool_call()
    assert gov.exhausted() == ["tool calls"]
    assert gov.consume_retry() and not gov.consume_retry()


def test_budget_elapsed_limit(tmp_path):
    gov = BudgetGovernor(ExecutionBudget(max_elapsed_seconds=1000))
    assert gov.exhausted() == []
    gov.usage.started_at -= 2000
    assert gov.exhausted() == ["elapsed time"]


def test_verification_hierarchy_prefers_strongest():
    weak = VerificationCheck(name="claim", passed=True, strength=CheckStrength.CLAIM)
    strong = VerificationCheck(name="pytest", passed=True, strength=CheckStrength.TESTS)
    assert (
        verify([weak], ["done"]).confidence
        < verify([weak, strong], ["done"]).confidence
    )


def test_model_claim_alone_never_verifies():
    assert not verify([], ["done"]).passed
    failed = verify([VerificationCheck(name="t", passed=False)], ["done"])
    assert not failed.passed and failed.failures


def test_completion_gate_needs_everything():
    assert completion_allowed(True, True, True, True)
    assert not completion_allowed(True, True, False, True)


def test_file_check_is_real_inspection(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello world")
    assert file_contains_check("has-hello", target, "hello").passed
    assert not file_contains_check("has-bye", target, "bye").passed
    assert not file_contains_check("missing", tmp_path / "nope.txt", "x").passed


def test_classify_transient_vs_deterministic():
    assert classify_failure("connection timeout after 30s") == FailureClass.TRANSIENT
    assert classify_failure("file does not exist") == FailureClass.DETERMINISTIC
    assert classify_failure("", transient=True) == FailureClass.TRANSIENT
    assert classify_failure("   ") == FailureClass.UNKNOWN


def test_identical_failure_forces_new_strategy():
    seen = {"a\x00h\x00x": 1}
    assert decide("a", "h", "x", FailureClass.TRANSIENT, seen) == Strategy.RETRY
    seen["a\x00h\x00x"] = 2
    assert (
        decide("a", "h", "x", FailureClass.TRANSIENT, seen)
        == Strategy.RETRIEVE_MORE_CONTEXT
    )
    seen["a\x00h\x00x"] = 9
    assert decide("a", "h", "x", FailureClass.DETERMINISTIC, seen) == Strategy.STOP


def test_progress_invariant():
    assert not progress_made()
    assert progress_made(new_evidence=True)


def test_knowledge_gate():
    good = KnowledgeCandidate(type="SOLUTION", content="restart the worker to clear it")
    assert extract([good], [], has_evidence=False) == []
    assert extract([good], [], has_evidence=True)[0].content == good.content
    assert (
        extract(
            [good],
            [KnowledgeItem(id="k", type="SOLUTION", content=good.content)],
            has_evidence=True,
        )
        == []
    )
    assert not is_durable(KnowledgeCandidate(type="SOLUTION", content="tmp debug"))
    assert resolve_conflict(["model_inference", "verified_source"]) == "verified_source"
