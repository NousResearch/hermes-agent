from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.ship_crew_contracts import (
    ContractValidationError,
    canonical_sha256,
    contract_metadata,
    validate_completion_metadata,
    validate_contract,
)
from hermes_cli.ship_crew_planning import (
    GraphCompilationError,
    classify_risk,
    compile_graph,
    compile_graph_plan,
    sail_before_execution,
)


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    return home


def _envelope(contract_type: str, producer: str, consumer: str, payload: dict) -> dict:
    return {
        "schema_version": "crew-handoff/v1",
        "mission_id": "Q-S01-TEST",
        "task_id": "t_child",
        "source_task_id": "t_parent",
        "producer": producer,
        "consumer": consumer,
        "contract_type": contract_type,
        "governance_class": "standard",
        "governance_version": "0.5.1",
        "idempotency_key": f"Q-S01-TEST:{producer}:r1",
        "data_class": "ordinary",
        "authority": {"write_scope": "worktree", "external_effect": False},
        "payload_ref": "/tmp/ship-crew-payload.json",
        "payload_sha256": canonical_sha256(payload),
    }


def test_each_role_contract_accepts_positive_payload():
    fixtures = [
        ("navigator-evidence/v1", "navigator", "captain", {"evidence": [], "findings": []}),
        ("engineer-delivery/v1", "engineer", "pirate", {"status": "completed", "changed_paths": [], "verification": [{"id": "V1", "status": "passed"}]}),
        ("pirate-review/v1", "pirate", "captain", {"decision": "approve", "findings": [], "required_action": "none"}),
        ("captain-disposition/v1", "captain", "user", {"decision": "sail", "rationale": "bounded", "sail_required": True}),
        ("crew-block/v1", "engineer", "captain", {"kind": "quota", "retryable": True, "blocked_until": 123, "quota_domain": "provider-a", "required_action": "wait", "required_evidence": []}),
    ]
    for contract_type, producer, consumer, payload in fixtures:
        envelope = _envelope(contract_type, producer, consumer, payload)
        validate_contract(envelope, payload)
        assert contract_metadata(envelope)["contract_type"] == contract_type


def test_contract_rejects_unknown_safety_fields_role_pairs_and_bad_digest():
    payload = {"decision": "approve", "findings": [], "required_action": "none"}
    envelope = _envelope("pirate-review/v1", "pirate", "captain", payload)
    envelope["unsafe"] = True
    with pytest.raises(ContractValidationError):
        validate_contract(envelope, payload)

    wrong_role = _envelope("pirate-review/v1", "engineer", "captain", payload)
    with pytest.raises(ContractValidationError, match="role pair"):
        validate_contract(wrong_role, payload)

    wrong_digest = _envelope("pirate-review/v1", "pirate", "captain", payload)
    wrong_digest["payload_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(ContractValidationError, match="payload_sha256"):
        validate_contract(wrong_digest, payload)


def test_completion_metadata_output_class_contract():
    validate_completion_metadata({"status": "completed", "output_class": "O1", "summary": "done", "evidence_refs": []})
    with pytest.raises(ContractValidationError):
        validate_completion_metadata({"status": "completed", "output_class": "O1", "summary": "done", "evidence_refs": [], "extra": True})


def _capsule(**overrides):
    capsule = {
        "schema_version": "quest-capsule/v1",
        "quest_id": "Q-S02-TEST",
        "user_intent": "implement bounded work",
        "objective": "produce verified result",
        "risk_inputs": {
            "blast_radius": 1,
            "reversibility": 1,
            "authority": 1,
            "data_security": 0,
            "uncertainty": 1,
            "operational_cost": 0,
            "hard_triggers": [],
        },
        "scope": {"external_effect": False, "write_scope": "read-only", "data_class": "ordinary"},
        "acceptance_evidence": ["pytest"],
    }
    for key, value in overrides.items():
        capsule[key] = value
    return capsule


def test_risk_boundaries_and_deterministic_reviewer():
    lite = classify_risk(_capsule())
    assert lite.governance_class == "lite"

    standard_capsule = _capsule()
    standard_capsule["risk_inputs"]["data_security"] = 2
    standard_capsule["risk_inputs"]["blast_radius"] = 2
    standard = classify_risk(standard_capsule)
    assert standard.governance_class == "standard"
    assert standard.reviewer == "pirate"

    constitutional_capsule = _capsule()
    constitutional_capsule["risk_inputs"]["hard_triggers"] = ["external-side-effect"]
    constitutional = classify_risk(constitutional_capsule)
    assert constitutional.governance_class == "constitutional"
    assert constitutional.reviewer is None


def test_graph_plan_cannot_lower_hard_trigger_class():
    capsule = _capsule()
    capsule["risk_inputs"]["hard_triggers"] = ["irreversible-write"]
    with pytest.raises(GraphCompilationError, match="lower"):
        compile_graph_plan(capsule, captain_classification="standard")


def test_graph_compile_is_atomic_idempotent_and_sail_gated(kanban_home: Path):
    capsule = _capsule()
    capsule["risk_inputs"]["hard_triggers"] = ["external-side-effect"]
    with kb.connect_closing() as conn:
        graph = compile_graph(conn, capsule)
        assert graph.task_ids is not None
        assert len(graph.task_ids) == 7
        assert sail_before_execution(conn, graph) is True
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 7
        execution = kb.get_task(conn, graph.task_ids["execution"])
        gate = kb.get_task(conn, graph.task_ids["user_sail"])
        assert execution is not None and execution.status == "todo"
        assert gate is not None and gate.status == "blocked" and gate.block_kind == "needs_input"

        again = compile_graph(conn, capsule)
        assert again.task_ids == graph.task_ids
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 7


def test_partial_idempotency_refuses_silent_repair(kanban_home: Path):
    capsule = _capsule()
    with kb.connect_closing() as conn:
        plan = compile_graph_plan(capsule)
        key = f"ship-crew:{plan.quest_id}:{plan.capsule_sha256[:16]}:{plan.graph_template}:{plan.nodes[0].key}"
        kb.create_task(conn, title="partial", idempotency_key=key)
        with pytest.raises(GraphCompilationError, match="partial graph"):
            compile_graph(conn, capsule)
