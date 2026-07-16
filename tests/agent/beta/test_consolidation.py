from agent.beta.consolidation import consolidate_results
from agent.beta.delegation import SpecialistResult
from agent.beta.router import route_request


def _result(specialist_id, **overrides):
    data = {
        "task_id": f"task:{specialist_id}",
        "specialist_id": specialist_id,
        "correlation_id": "corr",
        "status": "completed",
        "summary": f"{specialist_id} summary.",
        "confidence": 0.8,
    }
    data.update(overrides)
    return SpecialistResult(**data)


def test_consensus_deduplicates_and_keeps_hypotheses_separate():
    decision = route_request("Diagnose PostgreSQL lento")
    results = [
        _result("dba", facts=("Disk writes are high",), evidence=("metric:iops=95",), hypotheses=("Long query causes I/O",)),
        _result("monitoring", facts=("Disk writes are high",), evidence=("metric:iops=95",), hypotheses=("Long query causes I/O",)),
    ]
    consolidated = consolidate_results("Diagnose PostgreSQL lento", decision, results)
    assert consolidated.facts == ("Disk writes are high",)
    assert consolidated.evidence == ("metric:iops=95",)
    assert consolidated.probable_cause == "Long query causes I/O"
    assert "Long query causes I/O" not in consolidated.facts
    assert consolidated.contradictions == ()


def test_conflict_triggers_qa_and_reduces_confidence():
    decision = route_request("Diagnose PostgreSQL lento")
    results = [
        _result("dba", facts=("CPU is high",), confidence=0.9),
        _result("infra", facts=("CPU is normal",), confidence=0.9),
    ]
    calls = []

    def qa_validator(raw_results, contradictions):
        calls.append((raw_results, contradictions))
        return _result("qa-auditor", summary="QA found conflicting CPU evidence.", confidence=0.7)

    consolidated = consolidate_results(
        "Diagnose PostgreSQL lento", decision, results, qa_validator=qa_validator
    )
    assert calls and consolidated.contradictions
    assert consolidated.qa_required is True
    assert consolidated.qa_performed is True
    assert "qa-auditor" in consolidated.agents_activated
    assert consolidated.confidence < 0.9


def test_partial_failure_keeps_valid_sibling_evidence():
    decision = route_request("Diagnose PostgreSQL lento")
    results = [
        _result("dba", evidence=("pg_stat_activity row 42",), facts=("Query 42 holds a lock",)),
        _result("infra", status="timeout", errors=("deadline",), confidence=0),
    ]
    consolidated = consolidate_results("Diagnose PostgreSQL lento", decision, results)
    assert consolidated.evidence == ("pg_stat_activity row 42",)
    assert consolidated.partial_failures == ("infra: deadline",)


def test_missing_evidence_prevents_categorical_cause():
    decision = route_request("Diagnose PostgreSQL lento")
    consolidated = consolidate_results(
        "Diagnose PostgreSQL lento",
        decision,
        [_result("dba", hypotheses=("Maybe a lock",), confidence=0.9)],
    )
    assert consolidated.probable_cause is None
    assert consolidated.confidence <= 0.49
    assert "insufficient" in consolidated.result.lower()


def test_high_risk_recommendation_requires_approval_and_qa():
    decision = route_request("Diagnose PostgreSQL lento")
    consolidated = consolidate_results(
        "Diagnose PostgreSQL lento",
        decision,
        [_result("dba", recommended_actions=("Terminate database session 42",))],
    )
    assert consolidated.risk == "high"
    assert consolidated.authorization_required is True
    assert consolidated.qa_required is True
    assert "approval" in consolidated.next_step.lower() or "qa" in consolidated.next_step.lower()

