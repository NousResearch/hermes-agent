#!/usr/bin/env python3
"""Multi-recipe packaging runner for Neon Genie.

Usage:
  python scripts/recipe_run.py --list
  python scripts/recipe_run.py --name product-audit
  python scripts/recipe_run.py --name zero-option --out out/neon-genie/zero-option
  python scripts/recipe_run.py --name fragmentation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import recipe_common as rc

SKILL_ROOT = rc.SKILL_ROOT

RecipeFn = Callable[[Path], int]


def recipe_product_audit(out: Path) -> int:
    brief = SKILL_ROOT / "examples" / "product-audit.brief.yaml"
    route = rc.route_request(brief)
    rc.write_json(out / "profile-route.json", route)
    profiles = route.get("selected") or ["core"]

    handoff = {
        "packet": "WayfinderExecutionPacket",
        "status": "PROPOSED",
        "product_intent_changes_require_neon_genie_review": True,
        "profiles_loaded": profiles,
        "objective": "Audit product coherence and produce Wayfinder-ready handoff",
        "non_goals": [
            "Do not rewrite product intent in execution planning",
            "Do not grant execution or spending authority",
        ],
        "authority": "advisory_only",
        "grants_execution": False,
        "brief": rc.rel(brief),
        "note": "Stub handoff from packaging recipe. Full product packet is prose-runtime.",
    }
    handoff_path = out / "wayfinder-handoff.stub.json"
    rc.write_json(handoff_path, handoff)

    product_stub = {
        "target_user": "operator building a governed multi-skill product",
        "job_to_be_done": "clarify product boundary and handoff without inventing feasibility",
        "product_boundary": {
            "in": ["product intent", "scorecard", "wayfinder handoff stub"],
            "out": ["implementation status", "spend", "publish"],
        },
        "system_inventory": ["profiles", "schemas", "packaging CLI"],
        "core_loops": ["OPEN→SEAL advisory loop"],
        "conflict_scan": {"status": "PROPOSED", "conflicts": []},
        "experience_architecture": "operator CLI + Hermes skill prose",
        "validation_path": ["do check", "do eval", "human review"],
        "completion_proof": "Operator accepts product packet + handoff stub in review without intent rewrite",
        "proof_path": [
            "human reviews product-packet.stub.json",
            "confirm open DataRequests listed",
            "optional: satisfy access request then re-score integration",
            "append learning ledger on real outcome",
        ],
        "promotion_state": "SPEC_COMPLETE",
        "authority": "advisory_only",
        "post_seal_checklist": "references/post-seal-verification.md",
    }
    product_path = out / "product-packet.stub.json"
    rc.write_json(product_path, product_stub)
    rc.validate_packet(product_path, "product")

    # Example brief has empty canonical_sources → open private access DataRequest
    data_req = {
        "request_id": "dr-product-access",
        "field": "critical_integration_access",
        "why_decision_critical": (
            "Integration feasibility cannot be OBSERVED without declared access"
        ),
        "sensitivity": "private",
        "suggested_source": "Operator environment credentials or access inventory",
        "blocks_promotion": True,
        "status": "open",
    }
    data_requests = [data_req]
    rc.write_json(out / "data-requests.json", data_requests)

    receipt_path = out / "run-receipt.json"
    rc.build_receipt(
        receipt_path,
        profiles,
        status="PROPOSED",
        promotion_state="SPEC_COMPLETE",
        packets=[product_path, handoff_path],
        data_requests=data_requests,
        brief=brief,
    )
    return rc.finish(
        recipe="product-audit",
        brief=brief,
        out=out,
        route=route,
        artifacts=[
            out / "profile-route.json",
            product_path,
            handoff_path,
            out / "data-requests.json",
            receipt_path,
            out / "recipe-summary.json",
        ],
    )


def recipe_zero_option(out: Path) -> int:
    brief = SKILL_ROOT / "examples" / "zero-option.brief.yaml"
    route = rc.route_request(brief)
    rc.write_json(out / "profile-route.json", route)
    profiles = route.get("selected") or ["core", "zero_option"]

    # Example brief declares no skills/access → honest NOT_COMPUTABLE packet
    packet = {
        "constraints": ["no capital", "no fictional resources"],
        "capabilities": [],
        "opportunities": [],
        "shadow_flags": ["NON_EXECUTION", "FICTIONAL_RESOURCE_RISK_IF_INVENTED"],
        "system_state": "NOT_COMPUTABLE",
        "status": "NOT_COMPUTABLE",
        "reason": "No executable capabilities or access supplied",
        "completion_proof": "NOT_COMPUTABLE: no skills/access to prove first cash",
        "proof_path": [
            "declare skills and access",
            "re-run zero-option-executable recipe or Hermes zero_option profile",
        ],
        "authority": "advisory_only",
        "grants_execution": False,
    }
    packet_path = out / "zero-option-packet.json"
    rc.write_json(packet_path, packet)
    rc.validate_packet(packet_path, "zero_option")

    receipt_path = out / "run-receipt.json"
    rc.build_receipt(
        receipt_path,
        profiles,
        status="NOT_COMPUTABLE",
        promotion_state="NOT_COMPUTABLE",
        not_computable="capabilities,access,opportunities",
        packets=[packet_path],
        brief=brief,
    )
    return rc.finish(
        recipe="zero-option",
        brief=brief,
        out=out,
        route=route,
        artifacts=[
            out / "profile-route.json",
            packet_path,
            receipt_path,
            out / "recipe-summary.json",
        ],
        extra={"outcome": "NOT_COMPUTABLE", "gate": "G"},
    )


def recipe_zero_option_executable(out: Path) -> int:
    """Constrained but declared assets → micro-loop stub (still advisory)."""
    brief = SKILL_ROOT / "examples" / "zero-option-with-skills.brief.yaml"
    route = rc.route_request(brief)
    rc.write_json(out / "profile-route.json", route)
    profiles = route.get("selected") or ["core", "zero_option"]

    packet = {
        "constraints": ["no capital", "7 day window"],
        "capabilities": ["writing", "existing audience channel", "public portfolio"],
        "opportunities": [
            {
                "name": "paid micro-audit from existing network",
                "proof": "first paid engagement or honest no",
                "window_days": 7,
            }
        ],
        "shadow_flags": [],
        "system_state": "OPTIONALITY",
        "micro_loop": {
            "narrative": "Use only declared skills/access",
            "action": "Offer one bounded paid diagnostic to known warm contacts",
            "result": "cash or documented refusal",
            "reinforcement": "log outcome; no fictional pipeline",
        },
        "completion_proof": "Paid engagement booked or written refusal within 7 days",
        "proof_path": [
            "list warm contacts from declared access",
            "send one bounded paid diagnostic offer",
            "record cash or refusal",
            "append learning ledger (proof_obtained or proof_failed)",
        ],
        "status": "PROPOSED",
        "authority": "advisory_only",
        "grants_execution": False,
    }
    packet_path = out / "zero-option-packet.json"
    rc.write_json(packet_path, packet)
    rc.validate_packet(packet_path, "zero_option")

    receipt_path = out / "run-receipt.json"
    rc.build_receipt(
        receipt_path,
        profiles,
        status="PROPOSED",
        promotion_state="TESTABLE",
        packets=[packet_path],
        brief=brief,
    )
    return rc.finish(
        recipe="zero-option-executable",
        brief=brief,
        out=out,
        route=route,
        artifacts=[
            out / "profile-route.json",
            packet_path,
            receipt_path,
            out / "recipe-summary.json",
        ],
        extra={"outcome": "PROPOSED_MICRO_LOOP"},
    )


def recipe_fragmentation(out: Path) -> int:
    brief = SKILL_ROOT / "examples" / "fragmentation.brief.yaml"
    route = rc.route_request(brief)
    rc.write_json(out / "profile-route.json", route)
    profiles = route.get("selected") or ["core", "fragmentation"]

    packet = {
        "target_outcome": "Reduce multi-portal status reconciliation tax",
        "scale": "team-to-org",
        "fragmentation_types": ["workflow", "identity", "payment", "data"],
        "systems_involved": ["portal_a", "portal_b", "spreadsheet_status"],
        "friction_events": [
            {
                "site": "multi-portal status lookup",
                "frequency": "daily",
                "severity": "high",
                "who_pays_tax": "operator / end user",
                "authority_to_fix": "unknown",
            }
        ],
        "coordination_gap": "Systems work alone but force repeated handoffs",
        "proposed_defragmentation_layer": {
            "archetype": "status unifier",
            "integration_burden": "medium",
            "capturable_value": "time saved on status reconciliation",
            "reject_if": "integration burden exceeds capturable value",
        },
        "scorecard": {
            "integration_feasibility": "NOT_COMPUTABLE_without_access"
        },
        "completion_proof": "Measured friction reduction after access inventory OR explicit reject (burden > value)",
        "proof_path": [
            "satisfy access DataRequest",
            "measure handoff time tax",
            "compare integration burden vs capturable value",
            "promote only if net positive",
        ],
        "status": "PROPOSED",
        "promotion_state": "MAPPED",
        "authority": "advisory_only",
        "grants_execution": False,
    }
    packet_path = out / "fragmentation-packet.json"
    rc.write_json(packet_path, packet)
    rc.validate_packet(packet_path, "fragmentation")

    receipt_path = out / "run-receipt.json"
    rc.build_receipt(
        receipt_path,
        profiles,
        status="PROPOSED",
        promotion_state="MAPPED",
        not_computable="critical_integration_access",
        packets=[packet_path],
        brief=brief,
    )
    return rc.finish(
        recipe="fragmentation",
        brief=brief,
        out=out,
        route=route,
        artifacts=[
            out / "profile-route.json",
            packet_path,
            receipt_path,
            out / "recipe-summary.json",
        ],
    )


def recipe_commercial(out: Path) -> int:
    brief = SKILL_ROOT / "examples" / "commercial.brief.yaml"
    route = rc.route_request(brief)
    rc.write_json(out / "profile-route.json", route)
    profiles = route.get("selected") or ["core", "commercial"]

    data_req = {
        "request_id": "dr-buyer-budget",
        "field": "buyer_budget_authority",
        "why_decision_critical": "Cannot separate buyer vs beneficiary or set firm price without budget authority",
        "sensitivity": "private",
        "suggested_source": "Operator CRM or stakeholder interview notes",
        "blocks_promotion": True,
        "status": "open",
    }
    data_requests = [data_req]
    rc.write_json(out / "data-requests.json", data_requests)

    packet = {
        "buyer_map": {
            "user": "NOT_COMPUTABLE_until_declared",
            "beneficiary": "NOT_COMPUTABLE_until_declared",
            "buyer": "NOT_COMPUTABLE_until_declared",
            "payer": "NOT_COMPUTABLE_until_declared",
            "authorizer": "NOT_COMPUTABLE_until_declared",
            "risk_bearer": "NOT_COMPUTABLE_until_declared",
            "note": "Gate C: do not collapse roles without evidence",
        },
        "pricing_model": {
            "status": "PROPOSED_SCAFFOLD",
            "shape": "fixed diagnostic fee + optional follow-on",
            "firm_price": "NOT_COMPUTABLE",
            "label": "SPECULATIVE without buyer + cite",
        },
        "cost_scenarios": {
            "conservative": "NOT_COMPUTABLE",
            "balanced": "NOT_COMPUTABLE",
            "aggressive": "NOT_COMPUTABLE",
        },
        "revenue_scenarios": {
            "conservative": "NOT_COMPUTABLE",
            "balanced": "NOT_COMPUTABLE",
            "aggressive": "NOT_COMPUTABLE",
        },
        "distribution": {
            "primary": "warm network / existing relationships",
            "label": "SPECULATIVE until operator confirms access",
        },
        "risks": [
            "buyer/beneficiary conflation",
            "fabricated market size",
            "pricing without completion proof",
        ],
        "completion_proof": "Named buyer + accepted price band OR honest NOT_COMPUTABLE after DataRequest",
        "proof_path": [
            "satisfy buyer_budget_authority DataRequest",
            "separate commercial roles with evidence",
            "cite public pricing comps or mark SPECULATIVE",
            "append learning ledger on first sale attempt",
        ],
        "authority": "advisory_only",
        "grants_execution": False,
        "promotion_state": "CONCEPTUAL",
    }
    packet_path = out / "commercial-simulation.stub.json"
    rc.write_json(packet_path, packet)
    rc.validate_packet(packet_path, "commercial")

    receipt_path = out / "run-receipt.json"
    rc.build_receipt(
        receipt_path,
        profiles,
        status="PROPOSED",
        promotion_state="CONCEPTUAL",
        not_computable="buyer_map,firm_price,revenue_scenarios",
        packets=[packet_path],
        data_requests=data_requests,
        brief=brief,
    )
    return rc.finish(
        recipe="commercial",
        brief=brief,
        out=out,
        route=route,
        artifacts=[
            out / "profile-route.json",
            packet_path,
            out / "data-requests.json",
            receipt_path,
            out / "recipe-summary.json",
        ],
        extra={"outcome": "SCAFFOLD_WITH_DATA_REQUEST", "gate": "C"},
    )


def recipe_audit(out: Path) -> int:
    brief = SKILL_ROOT / "examples" / "audit.brief.yaml"
    route = rc.route_request(brief)
    rc.write_json(out / "profile-route.json", route)
    profiles = route.get("selected") or ["core", "audit_delivery"]

    data_req = {
        "request_id": "dr-coi-metrics",
        "field": "measured_cost_of_inaction_inputs",
        "why_decision_critical": "Quantitative COI needs operator metrics; inventing $ is Gate J",
        "sensitivity": "operator",
        "suggested_source": "Operator incident logs, time-loss estimates, revenue impact notes",
        "blocks_promotion": False,
        "status": "open",
    }
    data_requests = [data_req]
    rc.write_json(out / "data-requests.json", data_requests)

    packet = {
        "current_state": {
            "summary": "Fragmented tools; limited operator notes in this exemplar",
            "label": "OBSERVED_from_request_only",
        },
        "observed_gaps": [
            "unclear cost of inaction",
            "no quantified remediation ROI",
            "authority boundaries for implementation not declared",
        ],
        "cost_of_inaction": {
            "qualitative": "Delay preserves coordination tax and decision fog",
            "quantified_usd": "NOT_COMPUTABLE",
            "label": "qualitative only under offline / missing metrics",
        },
        "target_state": {
            "summary": "Legible diagnostic package for human go/no-go",
        },
        "intervention_sequence": [
            "map current state from operator files",
            "list gaps with claim labels",
            "define validation gates",
            "optional implementation offer (non-coercive)",
        ],
        "validation_gates": [
            "no fabricated dollar COI",
            "offline prior not marked OBSERVED",
            "diagnosis does not authorize contact or spend",
        ],
        "evidence_manifest": {
            "operator_supplied": [],
            "live_research": "skipped_offline",
        },
        "authority_boundaries": {
            "authority": "advisory_only",
            "grants_execution": False,
            "may_contact_client": False,
        },
        "implementation_offer": {
            "status": "optional_scaffold",
            "note": "Offer map only; operator decides",
        },
        "completion_proof": "Stakeholder accepts diagnostic package as decision input without unauthorized outreach",
        "proof_path": [
            "human reviews audit packet",
            "optionally supply COI metrics DataRequest",
            "re-run online for external comps if needed",
            "ledger outcome after client decision",
        ],
        "not_computable_fields": ["quantified_cost_of_inaction"],
        "authority": "advisory_only",
        "grants_execution": False,
    }
    packet_path = out / "audit-delivery.stub.json"
    rc.write_json(packet_path, packet)
    rc.validate_packet(packet_path, "audit")

    receipt_path = out / "run-receipt.json"
    rc.build_receipt(
        receipt_path,
        profiles,
        status="PROPOSED",
        promotion_state="TESTABLE",
        not_computable="quantified_cost_of_inaction",
        packets=[packet_path],
        data_requests=data_requests,
        brief=brief,
    )
    return rc.finish(
        recipe="audit",
        brief=brief,
        out=out,
        route=route,
        artifacts=[
            out / "profile-route.json",
            packet_path,
            out / "data-requests.json",
            receipt_path,
            out / "recipe-summary.json",
        ],
        extra={"outcome": "OFFLINE_DIAGNOSTIC_SCAFFOLD", "research": "offline"},
    )


def recipe_agentic(out: Path) -> int:
    brief = SKILL_ROOT / "examples" / "agentic.brief.yaml"
    route = rc.route_request(brief)
    rc.write_json(out / "profile-route.json", route)
    profiles = route.get("selected") or ["core", "agentic_services"]

    packet = {
        "requested_outcome": "Delegated diagnostic workflow without ornamental machine payments",
        "actions": [
            {
                "id": "intake",
                "actor": "agent",
                "autonomy": "AUTO_ALLOWED",
                "description": "Collect operator-supplied brief fields",
            },
            {
                "id": "clarify",
                "actor": "agent",
                "autonomy": "USER_CONFIRMATION_REQUIRED",
                "description": "Clarify scope for subjective consulting work",
            },
            {
                "id": "draft",
                "actor": "agent",
                "autonomy": "USER_CONFIRMATION_REQUIRED",
                "description": "Draft deliverable for human review",
            },
            {
                "id": "deliver",
                "actor": "human",
                "autonomy": "QUALIFIED_HUMAN_REQUIRED",
                "description": "Client-facing delivery / relationship management",
            },
        ],
        "authority_gates": {
            "default": "USER_CONFIRMATION_REQUIRED",
            "spend": "PROHIBITED",
            "machine_settlement": "PROHIBITED_in_this_scaffold",
        },
        "exception_paths": [
            "escalate ambiguous scope to human",
            "reject x402 when persistent relationship billing is superior",
        ],
        "completion_proof": "Payment model decision recorded; no auto-spend paths enabled",
        "proof_path": [
            "confirm service is relationship-heavy",
            "set x402_fit REJECT or redesign for machine-payable units",
            "human confirms action graph authority gates",
        ],
        "x402_fit": "REJECT",
        "x402_reason": "persistent subjective consulting favors account/subscription billing (Gate F)",
        "authority": "advisory_only",
        "grants_execution": False,
        "promotion_state": "CONCEPTUAL",
    }
    packet_path = out / "agentic-service-graph.stub.json"
    rc.write_json(packet_path, packet)
    rc.validate_packet(packet_path, "agentic")

    receipt_path = out / "run-receipt.json"
    rc.build_receipt(
        receipt_path,
        profiles,
        status="PROPOSED",
        promotion_state="CONCEPTUAL",
        packets=[packet_path],
        brief=brief,
    )
    return rc.finish(
        recipe="agentic",
        brief=brief,
        out=out,
        route=route,
        artifacts=[
            out / "profile-route.json",
            packet_path,
            receipt_path,
            out / "recipe-summary.json",
        ],
        extra={"x402_fit": "REJECT", "gate": "F"},
    )


def recipe_memetic(out: Path) -> int:
    brief = SKILL_ROOT / "examples" / "memetic.brief.yaml"
    route = rc.route_request(brief)
    rc.write_json(out / "profile-route.json", route)
    profiles = route.get("selected") or ["core", "memetic"]

    packet = {
        "candidates": [
            {"name": "Quantum Leverage OS", "type": "name", "status": "pressure_test_only"},
            {"name": "Bounded Proof Loop", "type": "name", "status": "safer_claim_surface"},
        ],
        "pressure_scores": {
            "Quantum Leverage OS": {
                "stickiness": 0.9,
                "claim_defensibility": 0.2,
                "hype_drift_risk": 0.85,
            },
            "Bounded Proof Loop": {
                "stickiness": 0.55,
                "claim_defensibility": 0.7,
                "hype_drift_risk": 0.3,
            },
        },
        "claim_defensibility": {
            "note": "High stickiness cannot invent feasibility",
            "evidence_gate": "FAIL",
        },
        "hype_drift_risks": [
            "Name implies capability not evidenced",
            "Public language may overclaim BUILD_READY",
        ],
        "promotion_gate_status": {
            "memetic_may_raise_promotion": False,
            "promotion_state_max": "CONCEPTUAL",
            "gate": "D",
        },
        "level99_route": {
            "status": "optional",
            "note": "Public-language execution remains downstream; Neon drafts only",
        },
        "hooks": ["proof-first", "evidence-bound"],
        "rejected_candidates": [],
        "completion_proof": "Evidence/feasibility gates pass independently of name scores",
        "authority": "advisory_only",
        "grants_execution": False,
    }
    packet_path = out / "memetic-pressure.stub.json"
    rc.write_json(packet_path, packet)
    rc.validate_packet(packet_path, "memetic")

    receipt_path = out / "run-receipt.json"
    rc.build_receipt(
        receipt_path,
        profiles,
        status="PROPOSED",
        promotion_state="CONCEPTUAL",
        packets=[packet_path],
        brief=brief,
    )
    return rc.finish(
        recipe="memetic",
        brief=brief,
        out=out,
        route=route,
        artifacts=[
            out / "profile-route.json",
            packet_path,
            receipt_path,
            out / "recipe-summary.json",
        ],
        extra={"gate": "D", "memetic_may_raise_promotion": False},
    )


def recipe_evidence(out: Path) -> int:
    brief = SKILL_ROOT / "examples" / "evidence.brief.yaml"
    route = rc.route_request(brief)
    rc.write_json(out / "profile-route.json", route)
    profiles = route.get("selected") or ["core", "evidence_intelligence"]

    data_req = {
        "request_id": "dr-crm-conversion",
        "field": "crm_pipeline_conversion",
        "why_decision_critical": "Private conversion rates required for non-speculative commercial scenarios",
        "sensitivity": "private",
        "suggested_source": "Operator CRM export (no private scraping)",
        "blocks_promotion": True,
        "status": "open",
    }
    data_requests = [data_req]
    rc.write_json(out / "data-requests.json", data_requests)

    packet = {
        "decision_questions": [
            "Who are public competitors and what do they publish on pricing?",
            "What private conversion rates apply to our pipeline?",
        ],
        "query_plan": [
            {
                "question": "public competitor positioning",
                "source_class": "open_web",
                "status": "planned_or_tool_dependent",
            },
            {
                "question": "CRM conversion",
                "source_class": "operator_private",
                "status": "data_request",
            },
        ],
        "evidence_items": [],
        "claim_ledger": [
            {
                "claim": "Public comps require host fetch + cite for OBSERVED",
                "label": "SPECULATIVE_until_cited",
            },
            {
                "claim": "Private conversion rate",
                "label": "NOT_COMPUTABLE",
                "data_request": "dr-crm-conversion",
            },
        ],
        "tooling_gaps": [],
        "attribution_boundaries": {
            "person": [],
            "company": [],
            "foundation": [],
            "model_prior": "SPECULATIVE_only",
        },
        "research_log": [
            {
                "query": "public competitor pricing for declared category",
                "outcome": "tool_dependent_packaging_scaffold",
            }
        ],
        "not_computable_fields": ["crm_pipeline_conversion", "uncited_market_size"],
        "competitive_scan": {"status": "scaffold", "items": []},
        "completion_proof": "Cited public evidence items + satisfied CRM DataRequest or waiver",
        "authority": "advisory_only",
        "grants_execution": False,
    }
    packet_path = out / "evidence-intelligence.stub.json"
    rc.write_json(packet_path, packet)
    rc.validate_packet(packet_path, "evidence")

    receipt_path = out / "run-receipt.json"
    rc.build_receipt(
        receipt_path,
        profiles,
        status="PROPOSED",
        promotion_state="TESTABLE",
        not_computable="crm_pipeline_conversion,uncited_market_size",
        packets=[packet_path],
        data_requests=data_requests,
        brief=brief,
    )
    return rc.finish(
        recipe="evidence",
        brief=brief,
        out=out,
        route=route,
        artifacts=[
            out / "profile-route.json",
            packet_path,
            out / "data-requests.json",
            receipt_path,
            out / "recipe-summary.json",
        ],
        extra={"outcome": "EVIDENCE_SCAFFOLD_WITH_REQUEST"},
    )


def recipe_opportunity(out: Path) -> int:
    brief = SKILL_ROOT / "examples" / "opportunity.brief.yaml"
    route = rc.route_request(brief)
    rc.write_json(out / "profile-route.json", route)
    profiles = route.get("selected") or ["core", "opportunity_mining"]

    data_req = {
        "request_id": "dr-skills-buyer",
        "field": "declared_skills_access_and_buyer",
        "why_decision_critical": "Cannot invent capabilities or economic buyer",
        "sensitivity": "operator",
        "suggested_source": "Operator inventory of skills, access, and who pays",
        "blocks_promotion": True,
        "status": "open",
    }
    data_requests = [data_req]
    rc.write_json(out / "data-requests.json", data_requests)

    packet = {
        "blocked_transition": {
            "from": "idea without buyer",
            "to": "testable paid diagnostic booked",
        },
        "outcome_model": {
            "success": "first cash or documented refusal",
            "failure": "silence with invented pipeline",
        },
        "system_topology": {
            "actors": ["operator", "buyer_unknown"],
            "systems": ["outreach", "delivery"],
        },
        "opportunity_thesis": (
            "Warm-network micro-audit may convert if skills/access exist; "
            "do not invent network or capital"
        ),
        "validation_path": [
            "declare skills/access/buyer",
            "offer one bounded diagnostic",
            "track invoice or no",
        ],
        "completion_proof": "first paid diagnostic invoice or signed SOW within 14 days",
        "proof_path": [
            "satisfy DataRequest for skills/access/buyer",
            "send bounded offer to declared warm contacts only",
            "record cash or refusal",
            "ledger proof_obtained or proof_failed",
        ],
        "scorecard": {
            "evidence_density": "low_until_request_satisfied",
            "outcome_clarity": "high_if_proof_defined",
        },
        "promotion_state": "TESTABLE",
        "authority": "advisory_only",
        "grants_execution": False,
    }
    packet_path = out / "opportunity-packet.stub.json"
    rc.write_json(packet_path, packet)
    rc.validate_packet(packet_path, "opportunity")

    receipt_path = out / "run-receipt.json"
    rc.build_receipt(
        receipt_path,
        profiles,
        status="PROPOSED",
        promotion_state="TESTABLE",
        not_computable="buyer,capabilities_until_declared",
        packets=[packet_path],
        data_requests=data_requests,
        brief=brief,
    )
    return rc.finish(
        recipe="opportunity",
        brief=brief,
        out=out,
        route=route,
        artifacts=[
            out / "profile-route.json",
            packet_path,
            out / "data-requests.json",
            receipt_path,
            out / "recipe-summary.json",
        ],
        extra={"outcome": "OPPORTUNITY_WITH_PROOF_AND_REQUEST"},
    )


def recipe_capital_sprint(out: Path) -> int:
    brief = SKILL_ROOT / "examples" / "capital-sprint.brief.yaml"
    route = rc.route_request(brief)
    rc.write_json(out / "profile-route.json", route)
    profiles = route.get("selected") or ["core", "privacy", "capital_sprint"]

    data_req = {
        "request_id": "dr-org-identity-deadline",
        "field": "org_legal_identity_and_hard_deadline",
        "why_decision_critical": "Capital sprint fails closed without legal org identity and deadline",
        "sensitivity": "operator",
        "suggested_source": "Operator org records (EIN, tax status) and board-approved deadline",
        "blocks_promotion": True,
        "status": "open",
    }
    data_requests = [data_req]
    rc.write_json(out / "data-requests.json", data_requests)

    packet = {
        "sprint_window": {"days": [7, 14, 30], "deadline": "NOT_COMPUTABLE_until_operator"},
        "impact_object": {
            "goal": "NOT_COMPUTABLE_until_operator",
            "unit_cost": "NOT_COMPUTABLE_until_operator",
            "note": "Stub packaging packet; full CapitalSprintPacket is prose-runtime",
        },
        "warm_network_classes": ["member", "alumni", "corporate", "creator", "cold"],
        "completion_proof": "externally checkable gift total vs floor by deadline OR documented shortfall",
        "proof_path": [
            "satisfy DataRequest for org identity and deadline",
            "publish advisory campaign card (operator executes)",
            "record gifts against floor",
            "ledger proof_obtained or proof_failed",
        ],
        "promotion_state": "TESTABLE",
        "authority": "advisory_only",
        "grants_execution": False,
        "constraints": [
            "no private org names in shared corpus",
            "dues are not donations",
        ],
    }
    packet_path = out / "capital-sprint-packet.stub.json"
    rc.write_json(packet_path, packet)
    # Typed validate skipped: validate_packet has no capital_sprint type yet

    receipt_path = out / "run-receipt.json"
    rc.build_receipt(
        receipt_path,
        profiles,
        status="PROPOSED",
        promotion_state="TESTABLE",
        not_computable="org_identity,deadline,floor_until_declared",
        packets=[packet_path],
        data_requests=data_requests,
        brief=brief,
    )
    return rc.finish(
        recipe="capital-sprint",
        brief=brief,
        out=out,
        route=route,
        artifacts=[
            out / "profile-route.json",
            packet_path,
            out / "data-requests.json",
            receipt_path,
            out / "recipe-summary.json",
        ],
        extra={"outcome": "CAPITAL_SPRINT_STUB_WITH_REQUEST"},
    )


RECIPES: dict[str, RecipeFn] = {
    "product-audit": recipe_product_audit,
    "zero-option": recipe_zero_option,
    "zero-option-executable": recipe_zero_option_executable,
    "fragmentation": recipe_fragmentation,
    "commercial": recipe_commercial,
    "audit": recipe_audit,
    "agentic": recipe_agentic,
    "memetic": recipe_memetic,
    "evidence": recipe_evidence,
    "opportunity": recipe_opportunity,
    "capital-sprint": recipe_capital_sprint,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Neon Genie packaging recipes")
    parser.add_argument("--list", action="store_true", help="List recipe names")
    parser.add_argument(
        "--name",
        default="product-audit",
        help=f"Recipe name (default product-audit). Known: {', '.join(RECIPES)}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default out/neon-genie/<name>)",
    )
    # Accept leftover so `do recipe product-audit` style can be layered later
    args, _unknown = parser.parse_known_args(argv)

    if args.list:
        for name in sorted(RECIPES):
            print(name)
        return 0

    name = args.name
    # Allow first positional-like unknown as name: handled via --name only for clarity
    if name not in RECIPES:
        print(f"FAIL: unknown recipe {name!r}", file=sys.stderr)
        print(f"  known: {', '.join(sorted(RECIPES))}", file=sys.stderr)
        return 1

    out = args.out or (SKILL_ROOT / "out" / "neon-genie" / name)
    out.mkdir(parents=True, exist_ok=True)
    try:
        return RECIPES[name](out)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
