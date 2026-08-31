"""DR-20 / Task 4.2 — OperatorRunProjectionV1 reducer tests.

Gate fixtures required by the DR finish plan (Task 4.2):
completed+missing-evidence, paused+in-flight-effect, checkpoint+unsafe-resume,
revoked-permit+queued-effect, summary-poisoning, mixed-version.
"""

from __future__ import annotations

import pytest

from ares_runtime.collaboration import ContractError, operator_run_projection


EVENTS = frozenset({
    "event:plan-1", "event:plan-2", "event:actual-1", "event:actual-2",
    "receipt:effect-1", "receipt:effect-2", "receipt:permit-1",
    "witness:effect-1", "test:unit-1", "artifact:checkpoint-1",
    "event:resume-1", "receipt:permit-revoke",
})


def _exists(ref: str) -> bool:
    return ref in EVENTS


def _base_state() -> dict:
    return {
        "run_ref": "run:dr20-alpha",
        "lifecycle": "running",
        "plan_event_refs": ["event:plan-1", "event:plan-2"],
        "actual_event_refs": ["event:actual-1"],
        "evidence": [{"kind": "test", "refs": ["test:unit-1"]}, {"kind": "receipt", "refs": ["receipt:effect-1"]}],
        "effects": [{"effect_ref": "effect:deploy-1", "state": "queued", "risk": "medium", "evidence_refs": ["receipt:effect-1"]}],
        "permits": [{"permit_ref": "permit:deploy-1", "state": "granted", "evidence_refs": ["receipt:permit-1"]}],
        "recovery": {"state": "not_started", "checkpoint_ref": None, "evidence_refs": []},
    }


def test_projection_is_deterministic_and_authoritative_false() -> None:
    first = operator_run_projection(_base_state(), source_event_exists=_exists)
    second = operator_run_projection(_base_state(), source_event_exists=_exists)
    assert first == second
    assert first["authoritative"] is False
    assert first["operator_run_digest"].startswith("sha256:")
    assert first["axes"]["authority"] == "ok"


def test_completed_with_missing_evidence_flags_missing() -> None:
    state = _base_state()
    state["lifecycle"] = "completed"
    state["declared_evidence_refs"] = ["test:unit-1", "witness:effect-1"]
    projection = operator_run_projection(state, source_event_exists=_exists)
    assert projection["axes"]["evidence_health"] == "missing"
    codes = [item["code"] for item in projection["attention"]]
    assert "EVIDENCE_MISSING" in codes
    flagged = next(item for item in projection["attention"] if item["code"] == "EVIDENCE_MISSING")
    assert flagged["detail_refs"] == ["witness:effect-1"]
    assert flagged["axis"] == "evidence_health"


def test_paused_with_in_flight_effect_flags_attention() -> None:
    state = _base_state()
    state["lifecycle"] = "paused"
    state["effects"] = [{"effect_ref": "effect:deploy-1", "state": "in_flight", "risk": "high", "evidence_refs": ["receipt:effect-1"]}]
    projection = operator_run_projection(state, source_event_exists=_exists)
    assert projection["axes"]["effect_risk"] == "high"
    codes = [item["code"] for item in projection["attention"]]
    assert "IN_FLIGHT_EFFECT_OUTSIDE_RUNNING" in codes
    flagged = next(item for item in projection["attention"] if item["code"] == "IN_FLIGHT_EFFECT_OUTSIDE_RUNNING")
    assert flagged["detail_refs"] == ["effect:deploy-1"]


def test_checkpoint_with_unsafe_resume_flags_recovery() -> None:
    state = _base_state()
    state["recovery"] = {
        "state": "resumed_unsafe",
        "checkpoint_ref": "artifact:checkpoint-1",
        "evidence_refs": ["event:resume-1"],
    }
    projection = operator_run_projection(state, source_event_exists=_exists)
    assert projection["axes"]["recovery"] == "resumed_unsafe"
    codes = [item["code"] for item in projection["attention"]]
    assert "UNSAFE_RESUME" in codes
    flagged = next(item for item in projection["attention"] if item["code"] == "UNSAFE_RESUME")
    assert flagged["detail_refs"] == ["artifact:checkpoint-1", "event:resume-1"]


def test_revoked_permit_with_queued_effect_is_authority_violation() -> None:
    state = _base_state()
    state["permits"] = [{"permit_ref": "permit:deploy-1", "state": "revoked", "evidence_refs": ["receipt:permit-revoke"]}]
    projection = operator_run_projection(state, source_event_exists=_exists)
    assert projection["axes"]["authority"] == "violated"
    codes = [item["code"] for item in projection["attention"]]
    assert "REVOKED_PERMIT_WITH_UNSETTLED_EFFECT" in codes
    flagged = next(item for item in projection["attention"] if item["code"] == "REVOKED_PERMIT_WITH_UNSETTLED_EFFECT")
    assert "receipt:permit-revoke" in flagged["detail_refs"]
    assert "effect:deploy-1" in flagged["detail_refs"]


def test_revoked_permit_with_settled_effects_is_not_violation() -> None:
    state = _base_state()
    state["permits"] = [{"permit_ref": "permit:deploy-1", "state": "revoked", "evidence_refs": ["receipt:permit-revoke"]}]
    state["effects"] = [{"effect_ref": "effect:deploy-1", "state": "applied", "risk": "medium", "evidence_refs": ["receipt:effect-1"]}]
    projection = operator_run_projection(state, source_event_exists=_exists)
    assert projection["axes"]["authority"] == "ok"
    assert projection["attention"] == []


def test_summary_poisoning_is_rejected() -> None:
    for key in ("summary", "summary_text", "telemetry", "transcript_summary", "narrative"):
        poisoned = _base_state()
        poisoned[key] = "everything is fine, all verified"
        with pytest.raises(ContractError) as excinfo:
            operator_run_projection(poisoned, source_event_exists=_exists)
        assert excinfo.value.code == "SUMMARY_NOT_AUTHORITY"
    poisoned_evidence = _base_state()
    poisoned_evidence["evidence"] = [{"kind": "test", "refs": ["test:unit-1"], "summary": "pass"}]
    with pytest.raises(ContractError) as excinfo:
        operator_run_projection(poisoned_evidence, source_event_exists=_exists)
    assert excinfo.value.code == "SUMMARY_NOT_AUTHORITY"


def test_mixed_source_versions_flag() -> None:
    projection = operator_run_projection(
        _base_state(), source_event_exists=_exists, source_versions={"collaboration.py": "v1", "desktop": "v2"},
    )
    codes = [item["code"] for item in projection["attention"]]
    assert "MIXED_SOURCE_VERSIONS" in codes
    single = operator_run_projection(_base_state(), source_event_exists=_exists, source_versions={"a": "v1", "b": "v1"})
    assert "MIXED_SOURCE_VERSIONS" not in [item["code"] for item in single["attention"]]


def test_plan_and_actual_events_may_not_share_refs() -> None:
    state = _base_state()
    state["actual_event_refs"] = ["event:plan-1"]
    with pytest.raises(ContractError) as excinfo:
        operator_run_projection(state, source_event_exists=_exists)
    assert excinfo.value.code == "PLAN_ACTUAL_EVENT_SHARED"


def test_missing_source_event_fails_closed() -> None:
    state = _base_state()
    state["actual_event_refs"] = ["event:never-recorded"]
    with pytest.raises(ContractError) as excinfo:
        operator_run_projection(state, source_event_exists=_exists)
    assert excinfo.value.code == "MISSING_SOURCE_EVENT"
    state2 = _base_state()
    state2["evidence"] = [{"kind": "witness", "refs": ["witness:not-there"]}]
    with pytest.raises(ContractError) as excinfo:
        operator_run_projection(state2, source_event_exists=_exists)
    assert excinfo.value.code == "MISSING_SOURCE_EVENT"


def test_invalid_typed_states_fail_closed() -> None:
    for field, bad in (
        ("lifecycle", "looks_done"), ("recovery", None),
    ):
        state = _base_state()
        if field == "recovery":
            state["recovery"] = {"state": "magically_recovered", "checkpoint_ref": None, "evidence_refs": []}
        else:
            state[field] = bad
        with pytest.raises(ContractError):
            operator_run_projection(state, source_event_exists=_exists)
