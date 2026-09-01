"""Public-review regression cases for OperatorRunProjectionV1."""

from __future__ import annotations

import pytest

from ares_runtime.collaboration import ContractError, operator_run_projection


REFS = {
    "event:plan", "event:actual", "receipt:effect", "receipt:permit",
    "artifact:checkpoint", "event:recovery",
}


def exists(ref: str) -> bool:
    return ref in REFS


def state() -> dict:
    return {
        "run_ref": "run:review",
        "lifecycle": "running",
        "plan_event_refs": ["event:plan"],
        "actual_event_refs": ["event:actual"],
        "evidence": [{"kind": "event", "refs": ["event:actual"]}],
        "effects": [{"effect_ref": "effect:one", "state": "queued", "risk": "high", "evidence_refs": ["receipt:effect"]}],
        "permits": [{"permit_ref": "permit:one", "state": "granted", "evidence_refs": ["receipt:permit"]}],
        "recovery": {"state": "not_started", "checkpoint_ref": None, "evidence_refs": []},
    }


def test_checkpoint_is_canonical_and_required_for_recovery_states() -> None:
    s = state()
    s["recovery"] = {"state": "checkpointed", "checkpoint_ref": None, "evidence_refs": []}
    with pytest.raises(ContractError) as exc:
        operator_run_projection(s, source_event_exists=exists)
    assert exc.value.code == "CHECKPOINT_REQUIRED"
    s["recovery"] = {"state": "resumed_safe", "checkpoint_ref": "artifact:missing", "evidence_refs": []}
    with pytest.raises(ContractError) as exc:
        operator_run_projection(s, source_event_exists=exists)
    assert exc.value.code == "MISSING_SOURCE_EVENT"


def test_summary_and_telemetry_are_rejected_in_effects_and_permits() -> None:
    for field in ("summary", "telemetry", "narrative"):
        s = state()
        s["effects"][0][field] = "green"
        with pytest.raises(ContractError) as exc:
            operator_run_projection(s, source_event_exists=exists)
        assert exc.value.code == "SUMMARY_NOT_AUTHORITY"
        s = state()
        s["permits"][0][field] = "green"
        with pytest.raises(ContractError) as exc:
            operator_run_projection(s, source_event_exists=exists)
        assert exc.value.code == "SUMMARY_NOT_AUTHORITY"


def test_terminal_queued_effect_requires_attention() -> None:
    s = state()
    s["lifecycle"] = "completed"
    projection = operator_run_projection(s, source_event_exists=exists)
    item = next(item for item in projection["attention"] if item["code"] == "UNSETTLED_EFFECT_OUTSIDE_RUNNING")
    assert item["detail_refs"] == ["effect:one"]


def test_source_versions_are_strictly_typed() -> None:
    for bad in ({"runtime": 3}, {3: "v1"}, {"runtime": ""}):
        with pytest.raises(ContractError) as exc:
            operator_run_projection(state(), source_event_exists=exists, source_versions=bad)
        assert exc.value.code == "INVALID_SOURCE_VERSIONS"
