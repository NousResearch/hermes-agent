from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "role_authority_gate", ROOT / "scripts/role_authority_gate.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


@pytest.fixture
def gate_module():
    return module


def test_public_publication_ready_is_rejected_with_claim_or_evidence_blockers(gate_module) -> None:
    for claim_blockers, evidence_blockers in ((["claim-gap"], []), ([], ["evidence-gap"])):
        result = gate_module.evaluate(
            gate_module.AuthorityRequest(
                role=gate_module.Role.PUBLIC,
                action=gate_module.Action.PUBLICATION_READY,
                payload={
                    "claim_blockers": claim_blockers,
                    "evidence_blockers": evidence_blockers,
                },
            )
        )
        assert not result.allowed
        assert result.code == "blocking_evidence"


def test_public_publication_ready_accepts_unblocked_claim(gate_module) -> None:
    result = gate_module.evaluate(
        gate_module.AuthorityRequest(
            role=gate_module.Role.PUBLIC,
            action=gate_module.Action.PUBLICATION_READY,
            payload={"claim_blockers": [], "evidence_blockers": []},
        )
    )
    assert result == gate_module.GateResult.allow()


def test_explorer_reconciliation_requires_preserved_dissent_reference(gate_module) -> None:
    for payload in (
        {"preservation": "dropped", "preserved_artifact_ref": "artifact://dissent"},
        {"preservation": "summary_only", "preserved_artifact_ref": None},
    ):
        result = gate_module.evaluate(
            gate_module.AuthorityRequest(
                role=gate_module.Role.EXPLORER,
                action=gate_module.Action.RECONCILIATION,
                payload=payload,
            )
        )
        assert not result.allowed
        assert result.code == "dissent_not_preserved"


def test_explorer_reconciliation_accepts_preserved_dissent(gate_module) -> None:
    result = gate_module.evaluate(
        gate_module.AuthorityRequest(
            role=gate_module.Role.EXPLORER,
            action=gate_module.Action.RECONCILIATION,
            payload={
                "preservation": "preserved_artifact",
                "preserved_artifact_ref": "artifact://dissent",
            },
        )
    )
    assert result == gate_module.GateResult.allow()


def test_data_evidence_cannot_promote_outside_its_lane(gate_module) -> None:
    result = gate_module.evaluate(
        gate_module.AuthorityRequest(
            role=gate_module.Role.DATA_EVIDENCE,
            action=gate_module.Action.PROMOTION,
            payload={"promotion": "publication_readiness"},
        )
    )
    assert not result.allowed
    assert result.code == "promotion_forbidden"


def test_public_evaluation_path_does_not_accept_caller_registry(gate_module) -> None:
    request = gate_module.AuthorityRequest(
        role=gate_module.Role.DATA_EVIDENCE,
        action=gate_module.Action.PROMOTION,
        payload={"promotion": "publication_readiness"},
    )
    with pytest.raises(TypeError):
        gate_module.evaluate(request, registry={"roles": []})


def test_public_evaluation_path_validates_canonical_registry(gate_module, tmp_path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"roles": []}), encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gate_module, "REGISTRY_PATH", path)
    try:
        with pytest.raises(gate_module.GateInputError, match="canonical role registry invalid"):
            gate_module.evaluate(
                gate_module.AuthorityRequest(
                    role=gate_module.Role.PUBLIC,
                    action=gate_module.Action.PUBLICATION_READY,
                    payload={"claim_blockers": [], "evidence_blockers": []},
                )
            )
    finally:
        monkeypatch.undo()


def test_data_evidence_cannot_use_runtime_authority_action(gate_module) -> None:
    with pytest.raises(gate_module.GateInputError, match="runtime_authority requires"):
        gate_module.evaluate(
            gate_module.AuthorityRequest(
                role=gate_module.Role.DATA_EVIDENCE,
                action=gate_module.Action.RUNTIME_AUTHORITY,
                payload={},
            )
        )


def test_data_evidence_accepts_allowed_promotions(gate_module) -> None:
    for promotion in ("observed_evidence", "verified_evidence", "derived_projection"):
        result = gate_module.evaluate(
            gate_module.AuthorityRequest(
                role=gate_module.Role.DATA_EVIDENCE,
                action=gate_module.Action.PROMOTION,
                payload={"promotion": promotion},
            )
        )
        assert result == gate_module.GateResult.allow()


def test_fiv_requires_all_stages_and_verifiable_states(gate_module) -> None:
    valid = [
        {"kind": "finding", "evidence_state": "verified"},
        {"kind": "implementation", "evidence_state": "derived"},
        {"kind": "verification", "evidence_state": "verified"},
    ]
    assert gate_module.evaluate(
        gate_module.AuthorityRequest(
            role=gate_module.Role.SUPERVISOR,
            action=gate_module.Action.FIV_PROMOTION,
            payload={"stages": valid},
        )
    ) == gate_module.GateResult.allow()

    for stages in (valid[:2], *(
        [
            {**stage, "evidence_state": state}
            if stage["kind"] == "verification" else stage
            for stage in valid
        ]
        for state in ("blocked", "unknown", "superseded")
    )):
        result = gate_module.evaluate(
            gate_module.AuthorityRequest(
                role=gate_module.Role.SUPERVISOR,
                action=gate_module.Action.FIV_PROMOTION,
                payload={"stages": stages},
            )
        )
        assert not result.allowed
        assert result.code == "fiv_not_verifiable"


def test_api_rejects_wrong_role_and_malformed_payload(gate_module) -> None:
    with pytest.raises(gate_module.GateInputError):
        gate_module.evaluate(
            gate_module.AuthorityRequest(
                role=gate_module.Role.EXPLORER,
                action=gate_module.Action.PUBLICATION_READY,
                payload={},
            )
        )
    with pytest.raises(gate_module.GateInputError):
        gate_module.evaluate(
            gate_module.AuthorityRequest(
                role=gate_module.Role.PUBLIC,
                action=gate_module.Action.PUBLICATION_READY,
                payload={"claim_blockers": "not-a-list", "evidence_blockers": []},
            )
        )


def test_cli_accepts_and_rejects_json_requests(gate_module, tmp_path, capsys) -> None:
    request = {
        "role": "role.public_evidence_editor",
        "action": "publication_ready",
        "payload": {"claim_blockers": [], "evidence_blockers": []},
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    assert gate_module.main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["allowed"] is True

    request["payload"]["claim_blockers"] = ["gap-1"]
    path.write_text(json.dumps(request), encoding="utf-8")
    assert gate_module.main([str(path)]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["allowed"] is False
    assert "not connected" in output["consumer_note"]
