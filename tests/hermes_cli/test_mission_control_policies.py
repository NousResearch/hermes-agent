"""Mission Control pure-policy contracts.

These tests intentionally exercise standalone, side-effect-free policy modules.
The dashboard/API can later import these modules without changing their safety
contracts.
"""

from hermes_cli.mission_control.cockpit_selection import select_cockpit_lanes
from hermes_cli.mission_control.dispatch_gate import (
    build_standard_subagent_output_contract,
    evaluate_dispatch_requests,
    get_role_contract,
)


def test_cockpit_selector_limits_visible_decisions_and_collapses_audit_noise():
    signals = [
        {
            "id": "MC-001",
            "kind": "release_gate",
            "title": "Brochure CO2Farm contiene claim carbon neutral",
            "status": "approval_required",
            "risk": "stop",
            "benefit": "high",
            "effort": "small",
            "evidence": "verified",
            "decision_required": True,
            "external_effect": True,
        },
        {
            "id": "MC-002",
            "kind": "proposal",
            "title": "Nuovo widget decorativo",
            "status": "proposal",
            "risk": "low",
            "benefit": "low",
            "effort": "medium",
            "evidence": "weak",
        },
        {
            "id": "MC-003",
            "kind": "failure",
            "title": "Batch transcript ha fallito dopo retry",
            "status": "failed_after_retry",
            "risk": "high",
            "benefit": "high",
            "effort": "small",
            "evidence": "verified",
        },
        {
            "id": "MC-004",
            "kind": "regression",
            "title": "Dashboard Team Operativo mostra più di 3 decisioni",
            "status": "approval_required",
            "risk": "high",
            "benefit": "high",
            "effort": "small",
            "evidence": "verified",
            "decision_required": True,
        },
        {
            "id": "MC-005",
            "kind": "handoff",
            "title": "DAN-13 validation report pronto",
            "status": "done",
            "risk": "low",
            "benefit": "medium",
            "effort": "small",
            "evidence": "verified",
        },
        {
            "id": "MC-006",
            "kind": "log",
            "title": "Log grezzo duplicato",
            "status": "done",
            "risk": "low",
            "benefit": "low",
            "effort": "small",
            "evidence": "weak",
        },
        {
            "id": "MC-007",
            "kind": "proposal",
            "title": "Duplicate decorative widget",
            "status": "duplicate",
            "risk": "low",
            "benefit": "low",
            "effort": "small",
            "evidence": "weak",
        },
    ]

    cockpit = select_cockpit_lanes(signals)

    assert cockpit["ok"] is True
    assert cockpit["guardrails"] == {
        "external_send": False,
        "auto_dispatch": False,
        "cron_created": False,
        "kanban_launch": False,
    }
    assert [item["id"] for item in cockpit["decide_now"]] == ["MC-001", "MC-003", "MC-004"]
    assert len(cockpit["decide_now"]) == 3
    assert [item["id"] for item in cockpit["handoff_collapsed"]] == ["MC-005"]
    assert {item["id"] for item in cockpit["deep_audit_collapsed"]} == {"MC-006", "MC-007"}
    assert all(item["ui_copy"].startswith("Preview-only") for item in cockpit["decide_now"])


def test_dispatch_gate_recommends_ready_only_after_explicit_safe_launch_gate():
    requests = [
        {
            "request_id": "DG-001",
            "title": "HermesPM sviluppo cockpit max-3",
            "role": "hermespm",
            "action_type": "development_task",
            "board": "sviluppo-hermes",
            "evidence": "DAN-12 validation report",
            "explicit_confirmation": True,
            "input_complete": True,
        },
        {
            "request_id": "DG-002",
            "title": "Reliability indaga failure transcript",
            "role": "reliability",
            "action_type": "failure_triage",
            "board": "sviluppo-hermes",
            "evidence": "batch failure log",
            "explicit_confirmation": False,
            "input_complete": True,
        },
        {
            "request_id": "DG-003",
            "title": "Research corpus YouTube senza fonte",
            "role": "research",
            "action_type": "source_extraction",
            "board": "team-operativo",
            "evidence": "",
            "explicit_confirmation": True,
            "input_complete": False,
        },
        {
            "request_id": "DG-004",
            "title": "CO2Farm pubblica claim certificazione",
            "role": "co2farm-mrv",
            "action_type": "co2farm_external_claim",
            "board": "team-operativo",
            "evidence": "claim draft",
            "explicit_confirmation": True,
            "input_complete": True,
            "external_effect": True,
        },
        {
            "request_id": "DG-005",
            "title": "Ruolo sconosciuto",
            "role": "random-agent",
            "action_type": "research_task",
            "board": "team-operativo",
            "evidence": "unit test",
            "explicit_confirmation": True,
            "input_complete": True,
        },
    ]

    result = evaluate_dispatch_requests(requests)

    assert result["ok"] is True
    assert result["counts"] == {
        "ready_to_launch": 1,
        "preview_only": 1,
        "blocked_for_input": 1,
        "denied": 2,
    }
    assert result["would_launch_count"] == 1
    assert result["side_effects_performed"] == {
        "kanban_task_created": False,
        "worker_launched": False,
        "cron_created": False,
        "external_send": False,
    }
    by_id = {item["request_id"]: item for item in result["results"]}
    assert by_id["DG-001"]["decision"] == "ready_to_launch"
    assert by_id["DG-001"]["recommended_kanban_status"] == "ready"
    assert by_id["DG-001"]["role_contract"]["known"] is True
    assert by_id["DG-001"]["role_contract"]["required_output_sections"] == [
        "status",
        "result",
        "evidence",
        "limits_or_uncertainties",
        "risks",
        "recommended_next_action",
    ]
    assert by_id["DG-002"]["decision"] == "preview_only"
    assert by_id["DG-002"]["recommended_kanban_status"] == "none"
    assert by_id["DG-003"]["decision"] == "blocked_for_input"
    assert by_id["DG-003"]["recommended_kanban_status"] == "blocked"
    assert by_id["DG-004"]["decision"] == "denied"
    assert "external effect requires separate external confirmation" in by_id["DG-004"]["reasons"]
    assert by_id["DG-005"]["decision"] == "denied"
    assert by_id["DG-005"]["role_contract"]["known"] is False


def test_specialist_role_contracts_are_lean_evidence_first_and_side_effect_safe():
    research = get_role_contract("research")
    assert research["known"] is True
    assert research["external_allowed"] is False
    assert research["evidence_required"] is True
    assert "transcript_analysis" in research["allowed_actions"]
    assert "evidence" in research["required_output_sections"]
    assert any("Non inventa fonti" in item for item in research["does_not"])

    unknown = get_role_contract("unknown-specialist")
    assert unknown["known"] is False
    assert unknown["evidence_required"] is True
    assert unknown["required_output_sections"] == research["required_output_sections"]

    handoff = build_standard_subagent_output_contract()
    assert "Status`: completed | partial | blocked | uncertain" in handoff
    assert "Evidence`" in handoff
    assert "over-engineering concerns" in handoff
    assert "not `completed`" in handoff
