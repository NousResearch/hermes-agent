"""Cron completion authority composed with the durable trusted settlement.

A trusted ``runtime_stop`` success deliberately writes no closing assistant row,
so the transcript tail alone cannot tell it apart from a run that died mid-turn.
Completion therefore consults the durable receipt — and a proof that is absent,
unsettled, malformed, or bound to another policy/fire/session must never promote
an empty response that would otherwise be the current soft failure.
"""

from __future__ import annotations

import pytest

from cron.scheduler_settlement import classify_completion

EMPTY_RESPONSE_ERROR = (
    "Agent completed but produced empty response "
    "(model error, timeout, or misconfiguration)"
)
JOB = {"id": "job-1", "name": "bounded", "runtime_policy": "fleet-runtime"}
FINAL_SESSION = "cron_job-1_child"


def _settlement(execution_id, *, status="success", job_id="job-1",
                session_id=FINAL_SESSION, policy="fleet-runtime"):
    """The shape the real producer writes: outcome + receipt, both run-bound."""
    run_id = f"cron:{job_id}:{execution_id}"
    return {
        "settled": True,
        "outcome": {"reason": "max_items", "status": status,
                    "policy": policy, "run_id": run_id},
        "receipt": {"status": "finalized", "policy": policy,
                    "run_id": run_id, "session_id": session_id},
    }


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def _fire(ledger, build=None, *, job_id="job-1", session_id=FINAL_SESSION):
    """Book a real fire, optionally recording the settlement ``build`` returns."""
    execution_id = ledger.create_execution(job_id, source="builtin")["id"]
    if build is not None:
        assert ledger.record_execution_settlement(
            execution_id, session_id=session_id, settlement=build(execution_id)) is True
    return execution_id


def test_valid_trusted_success_completes_without_assistant_prose(ledger):
    execution_id = _fire(ledger, _settlement)
    assert classify_completion(JOB, execution_id, FINAL_SESSION, True, None, "") == (True, None)


def test_trusted_failure_stays_a_failure_even_with_prose(ledger):
    execution_id = _fire(ledger, lambda e: _settlement(e, status="failure"))
    assert classify_completion(JOB, execution_id, FINAL_SESSION, True, None, "all done") == (
        False, "Authoritative runtime policy reported failure")


def test_absent_proof_never_promotes_an_empty_run(ledger):
    assert classify_completion(JOB, _fire(ledger), FINAL_SESSION, True, None, "") == (
        False, EMPTY_RESPONSE_ERROR)


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda s: s.__setitem__("settled", False), id="unsettled"),
    pytest.param(lambda s: s.pop("receipt"), id="no-receipt"),
    pytest.param(lambda s: s.__setitem__("outcome", "max_items"), id="outcome-not-a-mapping"),
    pytest.param(lambda s: s["outcome"].__setitem__("reason", ""), id="no-stop-reason"),
    pytest.param(lambda s: s["outcome"].__setitem__("status", "partial"), id="unknown-status"),
    pytest.param(lambda s: s["outcome"].__setitem__("policy", "other-policy"), id="wrong-policy"),
    pytest.param(lambda s: s["outcome"].__setitem__("run_id", "cron:job-1:other"), id="wrong-run"),
    pytest.param(lambda s: s["receipt"].__setitem__("status", "pending"), id="unfinalized-receipt"),
    pytest.param(lambda s: s["receipt"].__setitem__("session_id", "cron_job-1"), id="wrong-receipt-session"),
])
def test_unsound_proof_never_promotes_an_empty_run(ledger, mutate):
    def build(execution_id):
        settlement = _settlement(execution_id)
        mutate(settlement)
        return settlement

    assert classify_completion(JOB, _fire(ledger, build), FINAL_SESSION, True, None, "") == (
        False, EMPTY_RESPONSE_ERROR)


def test_proof_from_another_fire_never_promotes_this_one(ledger):
    """Execution identity is immutable: another fire's receipt proves nothing here."""
    mine = _fire(ledger)
    _fire(ledger, _settlement)  # a later fire of the same job, settled on the same session
    assert classify_completion(JOB, mine, FINAL_SESSION, True, None, "") == (
        False, EMPTY_RESPONSE_ERROR)


def test_proof_binds_to_the_final_session_after_compression_rotation(ledger):
    execution_id = _fire(ledger, _settlement)
    # The session the fire started on carries no proof...
    assert classify_completion(JOB, execution_id, "cron_job-1", True, None, "") == (
        False, EMPTY_RESPONSE_ERROR)
    # ...the rotated session the run ended on does, under the same fire id.
    assert classify_completion(JOB, execution_id, FINAL_SESSION, True, None, "") == (True, None)


def test_a_run_without_its_fire_id_can_never_be_promoted(ledger):
    _fire(ledger, _settlement)
    assert classify_completion(JOB, None, FINAL_SESSION, True, None, "") == (
        False, EMPTY_RESPONSE_ERROR)


@pytest.mark.parametrize("policy", [None, "", "observer"])
def test_untrusted_empty_response_remains_the_soft_failure(ledger, policy):
    """No authority in force: the durable row is observer data, not a verdict."""
    execution_id = _fire(ledger, _settlement)
    assert classify_completion(
        {**JOB, "runtime_policy": policy}, execution_id, FINAL_SESSION, True, None, "",
    ) == (False, EMPTY_RESPONSE_ERROR)


def test_a_valid_receipt_never_rescues_a_failed_run(ledger):
    execution_id = _fire(ledger, _settlement)
    assert classify_completion(JOB, execution_id, FINAL_SESSION, False, "provider timeout", "") == (
        False, "provider timeout")


def test_a_non_empty_response_is_untouched_without_authority(ledger):
    assert classify_completion(JOB, _fire(ledger), FINAL_SESSION, True, None, "the report") == (
        True, None)
