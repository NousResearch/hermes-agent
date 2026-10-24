#!/usr/bin/env python3
"""Structured anti-slop audit for Kubrick symbolic packets and graph-derived data."""
from __future__ import annotations
import argparse, json
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")


def load(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def norm(values):
    return {str(v).strip().lower() for v in values if str(v).strip()}


def violation(gate, code, evidence, repair):
    return {"gate": gate, "code": code, "evidence": evidence, "repair": repair}


def audit(packet: dict) -> dict:
    violations = []
    motifs = packet.get("motifs", [])
    channels = packet.get("channels", {})
    sites = packet.get("convergence_sites", [])
    cultural = packet.get("cultural_sources", [])
    actions = packet.get("causal_actions", [])
    claims = packet.get("interpretation_claims", [])

    if not actions:
        violations.append(violation("U", "NO_CAUSAL_ACTION", [], "Add at least one observable action whose consequence changes later behavior."))

    for motif in motifs:
        if not str(motif.get("dramatic_function", "")).strip():
            violations.append(violation("U", "MOTIF_WITHOUT_DRAMATIC_FUNCTION", [motif.get("motif_id")], "Assign a causal dramatic function or remove the motif."))
        recurrences = motif.get("recurrences", [])
        for index, recurrence in enumerate(recurrences[1:], start=2):
            mutation = str(recurrence.get("mutation", "")).strip().lower()
            if not mutation or mutation in {"none", "unchanged", "same"}:
                violations.append(violation("Q", "RECURRENCE_WITHOUT_MUTATION", [motif.get("motif_id"), index], "Mutate ownership, scale, rhythm, state, function, or consequence."))

    nonempty_channels = {name: norm(values) for name, values in channels.items() if values}
    channel_names = sorted(nonempty_channels)
    for i, left in enumerate(channel_names):
        for right in channel_names[i + 1:]:
            shared = nonempty_channels[left] & nonempty_channels[right]
            if shared:
                violations.append(violation("O", "CHANNEL_REDUNDANCY", [left, right, sorted(shared)], "Move one channel into counterpoint, mutation, or silence."))

    if len(sites) > 2:
        violations.append(violation("U", "CONVERGENCE_OVERLOAD", [len(sites)], "Reduce to one primary convergence site and at most one distinct secondary site."))
    for site in sites:
        if len(site.get("functions", [])) < 2:
            violations.append(violation("O", "WEAK_CONVERGENCE", [site.get("site_id")], "Require at least two independently legible functions at the site."))

    for source in cultural:
        if not str(source.get("boundary", "")).strip():
            violations.append(violation("S", "CULTURAL_SOURCE_WITHOUT_BOUNDARY", [source.get("source")], "Record what may transfer and what surface or equivalence is prohibited."))

    closure_terms = ("true meaning", "correct interpretation", "definitely symbolizes", "proves that")
    for claim in claims:
        lowered = str(claim).lower()
        if any(term in lowered for term in closure_terms):
            violations.append(violation("W", "PREMATURE_CLOSURE", [claim], "Rewrite as observable evidence and preserve interpretive openness."))

    status = "PASS" if not violations else "FAIL"
    return {
        "audit_version": "1.0.0",
        "status": status,
        "violation_count": len(violations),
        "violations": violations,
        "dimensions": {
            "causal_actions": len(actions),
            "motifs": len(motifs),
            "active_channels": len(nonempty_channels),
            "convergence_sites": len(sites),
            "cultural_sources": len(cultural),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(load(args.input))
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
