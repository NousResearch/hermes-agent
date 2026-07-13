"""Tests for the read-only autonomous proposal engine."""

import pytest

from hermes_cli.autonomous_proposals import (
    AutonomousProposalSafetyError,
    assert_autonomous_proposal_record_safe,
    build_autonomous_proposal_records,
)


def test_build_autonomous_proposal_records_from_observable_signal_with_chief_gate():
    records = build_autonomous_proposal_records(
        [
            {
                "id": "registry-passive-static-seeds",
                "title": "Team vivo: proposal engine autonomo con challenge interna",
                "kind": "evolution",
                "source_agent": "strategy-chief",
                "signal": "Registro attivo con seed statici e review pending osservati.",
                "interpretation": "La pagina governa approvazioni ma non produce iniziativa collettiva.",
                "supporter": "Systems Reliability Steward",
                "supporter_view": "Serve trasformare frizioni ricorrenti in proposte prima del prompt umano.",
                "critic": "Strategy Chief of Staff",
                "critic_view": "Rischio rumore: massimo pochi segnali con dedupe stabile e gate umano.",
                "chief_synthesis": "Preparare solo una proposta in review, senza dispatch automatico.",
                "confidence": "high",
                "evidence_refs": ["team_proposals:active_count", "team_proposals:pending_review_count"],
            }
        ],
        generated_at="2026-06-24T10:00:00Z",
    )

    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == "autonomous_proposal.v2"
    assert record["signal"]["summary"] == "Registro attivo con seed statici e review pending osservati."
    assert record["signal"]["observed_at"] == "2026-06-24T10:00:00Z"
    assert record["interpretation"]["hypothesis"]
    assert record["supporter"]["actor"] == "Systems Reliability Steward"
    assert "proposte" in record["supporter"]["rationale"]
    assert record["critic"]["actor"] == "Strategy Chief of Staff"
    assert "Rischio rumore" in record["critic"]["rationale"]
    assert record["chief_synthesis"]["synthesis"] == "Preparare solo una proposta in review, senza dispatch automatico."
    assert record["gate_state"] == "review_required"
    assert record["gate"]["requires_daniele"] is True
    assert "create ready executable tasks" in record["gate"]["forbidden_without_approval"]
    assert record["source_agent"] == "strategy-chief"
    assert record["confidence"] == "high"
    assert record["evidence_refs"] == ["team_proposals:active_count", "team_proposals:pending_review_count"]
    assert record["evidence_contract"]["refs"] == record["evidence_refs"]
    assert record["engine"]["method"] == "rules_based_v2"
    assert record["no_auto_dispatch"] is True
    assert record["auto_spawned"] is False
    assert record["cron_created"] is False
    assert record["external_send"] is False
    assert "task_id" not in record
    assert "plan_task_id" not in record


def test_autonomous_proposal_safety_guard_rejects_dispatch_intent_fields():
    record = build_autonomous_proposal_records(
        [
            {
                "title": "Unsafe dispatch",
                "source_agent": "reliability",
                "signal": "A signal exists.",
                "supporter_view": "Do it.",
                "critic_view": "Only with a gate.",
                "chief_synthesis": "Review first.",
                "evidence_refs": ["unit-test"],
            }
        ]
    )[0]

    for forbidden_patch in (
        {"auto_spawned": True},
        {"cron_created": True},
        {"external_send": True},
        {"task_id": "t_should_not_exist"},
        {"gate_state": "approved_for_dispatch"},
        {"gate": {"requires_daniele": False}},
        {"gate": {"requires_daniele": True, "forbidden_without_approval": ["send external message"]}},
    ):
        unsafe = {**record, **forbidden_patch}
        with pytest.raises(AutonomousProposalSafetyError):
            assert_autonomous_proposal_record_safe(unsafe)
