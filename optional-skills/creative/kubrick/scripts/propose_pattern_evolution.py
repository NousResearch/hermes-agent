#!/usr/bin/env python3
"""Aggregate multi-signal evidence into a human-reviewed evolution proposal.

Never applies structural or confidence changes automatically. Large confidence
deltas and any structural mutation require an explicit human review gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")

LARGE_CONFIDENCE_THRESHOLD = 0.08
MAX_DELTA = 0.2


def load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        return json.loads(text) or {}
    return yaml.safe_load(text) or {}


def clamp(value: float, lo: float = -MAX_DELTA, hi: float = MAX_DELTA) -> float:
    return max(lo, min(hi, value))


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def receipt_matches(receipt: dict[str, Any], pattern_id: str) -> bool:
    selected = receipt.get("selected_patterns") or {}
    if selected.get("primary") == pattern_id:
        return True
    if pattern_id in (selected.get("supporting") or []):
        return True
    return False


def forge_matches(bundle: dict[str, Any], pattern_id: str) -> dict[str, Any] | None:
    for item in bundle.get("pattern_evidence") or []:
        if item.get("pattern_id") == pattern_id:
            return item
    return None


def build(
    pattern_id: str,
    receipts: list[dict[str, Any]],
    forge_bundles: list[dict[str, Any]],
) -> dict[str, Any]:
    matching = [r for r in receipts if receipt_matches(r, pattern_id)]
    forge_hits = [(b, forge_matches(b, pattern_id)) for b in forge_bundles]
    forge_hits = [(b, hit) for b, hit in forge_hits if hit is not None]

    if not matching and not forge_hits:
        raise SystemExit("no supplied receipt or forge bundle references the requested pattern")

    accepted = sum(r.get("outcome", {}).get("status") == "ACCEPTED" for r in matching)
    rejected = sum(r.get("outcome", {}).get("status") == "REJECTED" for r in matching)
    partial = sum(r.get("outcome", {}).get("status") == "PARTIAL" for r in matching)

    fidelity = [
        r.get("signals", {}).get("visual_fidelity")
        for r in matching
        if isinstance(r.get("signals", {}).get("visual_fidelity"), (int, float))
    ]
    feasibility = [
        r.get("signals", {}).get("production_feasibility")
        for r in matching
        if isinstance(r.get("signals", {}).get("production_feasibility"), (int, float))
    ]
    anti_slop_vals = [
        1.0 if r.get("signals", {}).get("anti_slop_pass") else 0.0
        for r in matching
        if "anti_slop_pass" in (r.get("signals") or {})
    ]
    cultural_vals = [
        1.0 if r.get("signals", {}).get("cultural_boundary_respected") else 0.0
        for r in matching
        if r.get("signals", {}).get("cultural_boundary_respected") is not None
    ]
    payoff_vals = [
        1.0 if r.get("signals", {}).get("payoff_realized") else 0.0
        for r in matching
        if r.get("signals", {}).get("payoff_realized") is not None
    ]
    graph_vals = [
        1.0 if r.get("signals", {}).get("graph_valid") else 0.0
        for r in matching
        if "graph_valid" in (r.get("signals") or {})
    ]
    conf_vals = [
        float(r.get("signals", {}).get("retrieval_confidence"))
        for r in matching
        if isinstance(r.get("signals", {}).get("retrieval_confidence"), (int, float))
    ]

    mutation_events = 0
    mutation_successes = 0
    collision_count = 0
    debt_total = 0.0
    failed_payoffs = 0
    forge_conf: list[float] = []
    source_projects: list[str] = []

    for receipt in matching:
        if receipt.get("project_id"):
            source_projects.append(str(receipt["project_id"]))
    for bundle, hit in forge_hits:
        source_projects.append(str(bundle.get("project_id") or "unknown"))
        forge_conf.append(float(hit.get("outcome_confidence", 0.5)))
        collision_count += int(hit.get("collision_count", 0) or 0)
        debt_total += float(hit.get("debt_contribution", 0) or 0)
        if hit.get("payoff_status") == "failed":
            failed_payoffs += 1
        evidence = hit.get("evidence_of_use") or []
        if any("mutation" in str(e) for e in evidence):
            mutation_events += 1
            if hit.get("outcome_confidence", 0) >= 0.6:
                mutation_successes += 1
        delta = (bundle.get("signals") or {}).get("ledger_delta") or {}
        if pattern_id in (delta.get("motifs_mutated") or []):
            mutation_events += 1
            mutation_successes += 1

    avg_fidelity = mean([float(x) for x in fidelity])
    avg_feasibility = mean([float(x) for x in feasibility])
    avg_anti_slop = mean(anti_slop_vals)
    avg_cultural = mean(cultural_vals)
    avg_payoff = mean(payoff_vals)
    avg_graph = mean(graph_vals)
    avg_conf = mean(conf_vals + forge_conf) or 0.5
    mutation_success = (
        round(mutation_successes / mutation_events, 4) if mutation_events else None
    )
    collision_pressure = min(1.0, collision_count * 0.25) if (matching or forge_hits) else None
    debt_pressure = min(1.0, debt_total) if (matching or forge_hits) else None

    delta = 0.0
    rationale: list[str] = []
    lifecycle = "NONE"
    misuse: list[str] = []
    mutations: list[str] = []

    if accepted >= 2 and rejected == 0 and (avg_fidelity is None or avg_fidelity >= 0.9):
        delta += 0.03
        rationale.append("Repeated accepted outcomes with no observed rejection.")
    if rejected >= 2:
        delta -= 0.05
        rationale.append("Repeated rejected outcomes require confidence reduction and human review.")
    if rejected >= 3 or failed_payoffs >= 2:
        lifecycle = "DEPRECATE"
        rationale.append("Repeated debt, collision, or failed payoff evidence justifies deprecation proposal.")
        misuse.append("pattern repeatedly fails payoff realization or operator acceptance")
    if collision_count >= 3 or (debt_pressure is not None and debt_pressure >= 0.45):
        if lifecycle == "NONE":
            lifecycle = "DEPRECATE"
        rationale.append("Repeated collisions or symbolic debt pressure justify lifecycle review.")
        misuse.append("pattern contributes to ledger collisions or symbolic debt")
    if collision_count >= 5 or (debt_pressure is not None and debt_pressure >= 0.75 and rejected >= 2):
        lifecycle = "RETIRE"
        rationale.append("Severe repeated debt and failure evidence justify retirement proposal.")
    if avg_fidelity is not None and avg_fidelity < 0.7:
        delta = min(delta, -0.03)
        rationale.append("Observed visual fidelity remains below 0.70.")
    if avg_feasibility is not None and avg_feasibility < 0.5:
        delta = min(delta, -0.03)
        rationale.append("Production feasibility remains below 0.50.")
        misuse.append("production cost exceeds practical affordance for repeated use")
    if avg_anti_slop is not None and avg_anti_slop < 1.0:
        delta = min(delta, -0.02)
        rationale.append("Anti-slop compliance failed on at least one observed use.")
        misuse.append("anti-slop gate failure observed in production use")
    if avg_cultural is not None and avg_cultural < 1.0:
        delta = min(delta, -0.04)
        rationale.append("Cultural boundary was not respected in at least one observed use.")
        misuse.append("cultural boundary violation risk")
    if avg_payoff is not None and avg_payoff < 0.5:
        delta = min(delta, -0.03)
        rationale.append("Payoff realization rate is below 0.50.")
    if avg_payoff is not None and avg_payoff >= 0.8 and accepted >= 1:
        delta += 0.02
        rationale.append("Payoff realization is consistently observed.")
    if mutation_success is not None and mutation_success >= 0.75 and mutation_events >= 2:
        delta += 0.02
        rationale.append("Mutation success is consistently observed under production pressure.")
        mutations.append("production-pressure mutation path")
    if mutation_success is not None and mutation_success < 0.4 and mutation_events >= 2:
        delta = min(delta, -0.03)
        rationale.append("Mutation attempts repeatedly fail under production pressure.")
        mutations.append("failed mutation under production pressure")
    if partial and not accepted and not rejected:
        rationale.append("Only partial outcomes observed; retain confidence pending stronger evidence.")
    if not rationale:
        rationale.append("Evidence is mixed or insufficient; retain current confidence pending review.")

    delta = round(clamp(delta), 4)
    structural = bool(misuse or mutations or lifecycle != "NONE")
    large_conf = abs(delta) >= LARGE_CONFIDENCE_THRESHOLD
    review_required = structural or large_conf or rejected >= 2
    review_reasons: list[str] = []
    if large_conf:
        review_reasons.append(f"confidence_delta {delta} exceeds large-change threshold {LARGE_CONFIDENCE_THRESHOLD}")
    if structural:
        review_reasons.append("structural mutation or lifecycle change proposed")
    if rejected >= 2:
        review_reasons.append("multiple rejected outcomes")
    if not review_reasons:
        review_reasons.append("proposal retained for human acknowledgment")

    multi_signal = {
        "confidence_evidence": round(avg_conf, 4),
        "mutation_success": mutation_success,
        "production_feasibility": avg_feasibility,
        "anti_slop_compliance": avg_anti_slop,
        "cultural_boundary_respect": avg_cultural,
        "payoff_realization": avg_payoff,
        "collision_pressure": collision_pressure,
        "debt_pressure": debt_pressure,
        "visual_fidelity": avg_fidelity,
        "graph_validity": avg_graph,
    }

    evolution_receipt = {
        "schema_version": "1.0.0",
        "pattern_id": pattern_id,
        "evidence_sources": {
            "use_receipts": [r.get("receipt_id", "unknown") for r in matching],
            "forge_bundles": [b.get("bundle_id", "unknown") for b, _ in forge_hits],
            "source_projects": sorted(set(source_projects)),
        },
        "multi_signal_scores": {
            "confidence_evidence": multi_signal["confidence_evidence"],
            "mutation_success": multi_signal["mutation_success"],
            "production_feasibility": multi_signal["production_feasibility"],
            "anti_slop_compliance": multi_signal["anti_slop_compliance"],
            "cultural_boundary_respect": multi_signal["cultural_boundary_respect"],
            "payoff_realization": multi_signal["payoff_realization"],
            "collision_pressure": multi_signal["collision_pressure"],
            "debt_pressure": multi_signal["debt_pressure"],
        },
        "proposed_changes": {
            "confidence_delta": delta,
            "lifecycle_action": lifecycle,
            "structural": {
                "add_misuse_risks": sorted(set(misuse)),
                "add_mutation_variables": sorted(set(mutations)),
                "requires_human_approval": True,
            },
        },
        "human_review_gate": {
            "required": review_required,
            "reasons": review_reasons,
            "automatic_application_allowed": False,
            "large_confidence_change": large_conf,
            "structural_mutation": structural,
        },
        "rationale": rationale,
        "authority": {
            "state": "PROPOSAL",
            "forge_canonical": True,
            "automatic_corpus_change_allowed": False,
        },
    }
    evolution_receipt["receipt_id"] = hashlib.sha256(
        json.dumps(evolution_receipt, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    # Backward-compatible proposal artifact (existing schema)
    proposal = {
        "schema_version": "1.0.0",
        "pattern_id": pattern_id,
        "evidence_receipts": evolution_receipt["evidence_sources"]["use_receipts"]
        or evolution_receipt["evidence_sources"]["forge_bundles"],
        "proposed_changes": {
            "confidence_delta": delta,
            "add_misuse_risks": sorted(set(misuse)),
            "add_mutation_variables": sorted(set(mutations)),
            "lifecycle_action": lifecycle,
        },
        "rationale": rationale,
        "review": {
            "status": "PROPOSED",
            "reviewer": None,
            "automatic_application_allowed": False,
        },
        "multi_signal_receipt_id": evolution_receipt["receipt_id"],
        "human_review_gate": evolution_receipt["human_review_gate"],
        "pattern_history": {
            "evidence_of_use": sorted(
                {
                    *(
                        f"use_receipt:{r.get('receipt_id', 'unknown')}"
                        for r in matching
                    ),
                    *(
                        e
                        for _, hit in forge_hits
                        for e in (hit.get("evidence_of_use") or [])
                    ),
                }
            ),
            "source_projects": sorted(set(source_projects)),
            "outcome_confidence": multi_signal["confidence_evidence"],
        },
    }
    proposal["proposal_id"] = hashlib.sha256(
        json.dumps({k: v for k, v in proposal.items() if k != "proposal_id"}, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return {"proposal": proposal, "multi_signal_receipt": evolution_receipt}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern-id", required=True)
    parser.add_argument("--receipt", action="append", default=[], help="pattern-use receipt path")
    parser.add_argument("--forge-bundle", action="append", default=[], help="forge signal bundle path")
    parser.add_argument("--output", required=True, help="proposal YAML path")
    parser.add_argument("--receipt-output", help="optional multi-signal evolution receipt path")
    args = parser.parse_args()
    if not args.receipt and not args.forge_bundle:
        raise SystemExit("provide at least one --receipt or --forge-bundle")
    receipts = [load(path) for path in args.receipt]
    bundles = [load(path) for path in args.forge_bundle]
    result = build(args.pattern_id, receipts, bundles)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(result["proposal"], sort_keys=False), encoding="utf-8")
    receipt_path = Path(args.receipt_output) if args.receipt_output else out.with_name(out.stem + ".multi-signal.yaml")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(yaml.safe_dump(result["multi_signal_receipt"], sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(result["proposal"], sort_keys=False))
    print(yaml.safe_dump({"multi_signal_receipt_path": str(receipt_path)}, sort_keys=False))


if __name__ == "__main__":
    main()
