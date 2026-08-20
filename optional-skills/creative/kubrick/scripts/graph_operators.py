#!/usr/bin/env python3
"""CLI graph/ledger operators with auditable receipts. Fail closed on weak evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")

ESOTERICA = {
    "nigredo",
    "albedo",
    "citrinitas",
    "rubedo",
    "syzygy",
    "ouroboros",
    "choronzon",
    "kabbalah",
    "qabalah",
    "sephirot",
    "ain soph",
    "magnum opus",
    "solve et coagula",
    "thelema",
    "egregore",
}


def load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text) or {}
    return yaml.safe_load(text) or {}


def dump(data: dict[str, Any], path: Path | None) -> None:
    text = yaml.safe_dump(data, sort_keys=False)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)


def receipt_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def base_receipt(operator: str, status: str, errors: list[str], data: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": "1.0.0",
        "operator": operator,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "result": data,
        "authority": {
            "state": "OBSERVATION" if status != "NOT_COMPUTABLE" else "NOT_COMPUTABLE",
            "automatic_application_allowed": False,
            "fail_closed": True,
        },
    }
    body["receipt_id"] = receipt_id(body)
    return body


def score_saturation(ledger: dict[str, Any]) -> dict[str, Any]:
    motifs = ledger.get("active_motifs") or []
    if not isinstance(motifs, list):
        return base_receipt("saturation-score", "NOT_COMPUTABLE", ["active_motifs missing"], {})
    recurrences = sum(int(m.get("recurrence_count", 0) or 0) for m in motifs if isinstance(m, dict))
    collisions = len(ledger.get("collisions") or [])
    unresolved = len(ledger.get("unresolved_payoffs") or [])
    debt = float(ledger.get("symbolic_debt", 0) or 0)
    score = min(
        1.0,
        (len(motifs) * 0.12) + (recurrences * 0.04) + (collisions * 0.08) + (unresolved * 0.05) + (debt * 0.1),
    )
    band = "low" if score < 0.35 else "moderate" if score < 0.65 else "high"
    return base_receipt(
        "saturation-score",
        "OK",
        [],
        {
            "saturation_score": round(score, 4),
            "band": band,
            "active_motifs": len(motifs),
            "recurrence_total": recurrences,
            "collision_count": collisions,
            "unresolved_payoffs": unresolved,
            "symbolic_debt": debt,
            "recommendation": "reduce recurrence density or retire low-payoff motifs" if band == "high" else "within operable range",
        },
    )


def counterpoint(packet: dict[str, Any]) -> dict[str, Any]:
    """Score whether diegetic/dramaturgical/cinematic channels diversify rather than repeat."""
    channels = packet.get("channels") or packet.get("symbolic_channels") or {}
    diegetic = [str(x).lower() for x in (channels.get("diegetic") or [])]
    dramaturgical = [str(x).lower() for x in (channels.get("dramaturgical") or [])]
    cinematic = [str(x).lower() for x in (channels.get("cinematic") or [])]
    if not (diegetic or dramaturgical or cinematic):
        return base_receipt("counterpoint", "NOT_COMPUTABLE", ["no channel evidence"], {})

    def tokens(items: list[str]) -> set[str]:
        out: set[str] = set()
        for item in items:
            out.update(t for t in item.replace(",", " ").split() if len(t) > 3)
        return out

    a, b, c = tokens(diegetic), tokens(dramaturgical), tokens(cinematic)
    overlap_ab = len(a & b)
    overlap_ac = len(a & c)
    overlap_bc = len(b & c)
    total = max(1, len(a | b | c))
    redundancy = (overlap_ab + overlap_ac + overlap_bc) / total
    diversity = max(0.0, 1.0 - min(1.0, redundancy))
    status = "OK" if diversity >= 0.45 else "WARN"
    return base_receipt(
        "counterpoint",
        status,
        [],
        {
            "counterpoint_score": round(diversity, 4),
            "redundancy_ratio": round(min(1.0, redundancy), 4),
            "channel_sizes": {"diegetic": len(diegetic), "dramaturgical": len(dramaturgical), "cinematic": len(cinematic)},
            "recommendation": "introduce channel conflict or withheld confirmation" if diversity < 0.45 else "channels diversify adequately",
        },
    )


def lock_convergence(graph: dict[str, Any], site_id: str | None) -> dict[str, Any]:
    sites = graph.get("convergence_sites") or []
    if not sites:
        return base_receipt("convergence-lock", "NOT_COMPUTABLE", ["no convergence sites"], {})
    if site_id:
        site = next((s for s in sites if s.get("site_id") == site_id), None)
        if not site:
            return base_receipt("convergence-lock", "NOT_COMPUTABLE", [f"unknown site_id {site_id}"], {})
    else:
        site = max(sites, key=lambda s: float(s.get("mask_priority") or 0))
        site_id = site.get("site_id")
    node_ids = site.get("node_ids") or []
    edge_ids = site.get("edge_ids") or []
    effect = site.get("observable_effect") or ""
    errors = []
    if len(node_ids) < 2:
        errors.append("site requires at least two nodes")
    if not edge_ids and not site.get("functions"):
        errors.append("site lacks edge or function evidence")
    if not str(effect).strip():
        errors.append("site lacks observable effect")
    if errors:
        return base_receipt("convergence-lock", "NOT_COMPUTABLE", errors, {"site_id": site_id})
    return base_receipt(
        "convergence-lock",
        "OK",
        [],
        {
            "locked_site_id": site_id,
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "observable_effect": effect,
            "mask_priority": site.get("mask_priority"),
            "lock_policy": "preserve site identity across frames; mutate only declared participants",
        },
    )


def surface_occult_audit(text_or_packet: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(text_or_packet, dict):
        blob = json.dumps(text_or_packet, sort_keys=True).lower()
        surface = text_or_packet.get("surface_output") or text_or_packet.get("audience_prompt") or text_or_packet
        if isinstance(surface, dict):
            blob = json.dumps(surface, sort_keys=True).lower()
        elif isinstance(surface, str):
            blob = surface.lower()
    else:
        blob = str(text_or_packet).lower()
    hits = sorted({term for term in ESOTERICA if term in blob})
    status = "FAIL" if hits else "PASS"
    return base_receipt(
        "surface-occult-audit",
        status if status == "PASS" else "FAIL",
        [f"named esoterica in surface: {', '.join(hits)}"] if hits else [],
        {
            "named_esoterica_found": hits,
            "audience_safe": not hits,
            "repair": "replace named esoterica with observable geometry, material, rhythm, or relation" if hits else None,
        },
    )


def export_symbolic_architecture(graph: dict[str, Any], ledger: dict[str, Any] | None, brief: dict[str, Any] | None) -> dict[str, Any]:
    if graph.get("validation", {}).get("status") not in {None, "VALID"} and graph.get("nodes") is None:
        # allow graphs without validation block if nodes exist
        if not graph.get("nodes"):
            return base_receipt("symbolic-architecture-export", "NOT_COMPUTABLE", ["graph invalid or empty"], {})
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    sites = graph.get("convergence_sites") or []
    motifs = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        motifs.append(
            {
                "motif_id": node.get("id") or node.get("motif_id"),
                "observed_form": node.get("observed_form"),
                "initial_state": node.get("initial_state"),
                "target_state": node.get("target_state"),
                "pattern_links": node.get("pattern_links") or [],
                "provenance": node.get("provenance_label") or "OBSERVED",
            }
        )
    if ledger:
        for m in ledger.get("active_motifs") or []:
            if not any(x.get("motif_id") == m.get("motif_id") for x in motifs):
                motifs.append(
                    {
                        "motif_id": m.get("motif_id"),
                        "observed_form": m.get("observed_form"),
                        "current_state": m.get("current_state"),
                        "pattern_links": m.get("pattern_links") or [],
                        "provenance": "LEDGER",
                    }
                )
    surface = graph.get("surface_output") or {}
    architecture = {
        "governing_tension": (brief or {}).get("dramatic_problem") or graph.get("symbolic_intent", {}).get("dramatic_function"),
        "symbolic_intent": graph.get("symbolic_intent") or {},
        "motif_registry": motifs,
        "relations": edges,
        "convergence_plan": sites,
        "residue_plan": graph.get("residue") or surface.get("residue") or [],
        "cinematic_encoding": {
            "geometry": surface.get("geometry") or [],
            "light": surface.get("light") or [],
            "material": surface.get("material") or [],
            "state_differentials": surface.get("state_differentials") or [],
        },
        "tradition_boundaries": (ledger or {}).get("cultural_boundaries") or (brief or {}).get("cultural_boundaries") or [],
    }
    occult = surface_occult_audit(architecture.get("cinematic_encoding", {}))
    if occult["result"].get("named_esoterica_found"):
        return base_receipt(
            "symbolic-architecture-export",
            "NOT_COMPUTABLE",
            occult["errors"],
            {"partial_architecture": architecture},
        )
    export = {
        "symbolic_architecture": architecture,
        "forge_mappings": {
            "ledger_motifs": [m.get("motif_id") for m in motifs if m.get("motif_id")],
            "shot_contracts": {
                "geometry": architecture["cinematic_encoding"]["geometry"],
                "recurrence": [m.get("motif_id") for m in motifs if m.get("motif_id")],
            },
        },
        "provenance_summary": {
            "graph_id": graph.get("graph_id"),
            "project_id": (ledger or {}).get("project_id") or (brief or {}).get("project_id"),
            "motif_count": len(motifs),
        },
        "mutation_history": [
            {
                "motif_id": m.get("motif_id"),
                "last_mutation": m.get("last_mutation"),
                "recurrence_count": m.get("recurrence_count"),
            }
            for m in (ledger or {}).get("active_motifs") or []
            if isinstance(m, dict)
        ],
        "authority": {"state": "PROPOSED", "forge_canonical": False, "automatic_application_allowed": False},
    }
    return base_receipt("symbolic-architecture-export", "OK", [], export)


def mutate_motif(ledger: dict[str, Any], motif_id: str, observed_form: str, state: str, mutation: str) -> dict[str, Any]:
    motifs = ledger.setdefault("active_motifs", [])
    existing = next((m for m in motifs if m.get("motif_id") == motif_id), None)
    if existing:
        before = dict(existing)
        existing["current_state"] = state
        existing["last_mutation"] = mutation
        existing["recurrence_count"] = int(existing.get("recurrence_count", 0) or 0) + 1
        after = dict(existing)
    else:
        if not observed_form:
            return base_receipt("motif-mutation", "NOT_COMPUTABLE", ["observed_form required for new motif"], {})
        before = None
        after = {
            "motif_id": motif_id,
            "observed_form": observed_form,
            "current_state": state,
            "recurrence_count": 1,
            "last_mutation": mutation,
            "ownership": None,
            "pattern_links": [],
            "convergence_sites": [],
        }
        motifs.append(after)
    ledger["revision"] = int(ledger.get("revision", 0)) + 1
    return base_receipt(
        "motif-mutation",
        "OK",
        [],
        {
            "motif_id": motif_id,
            "before": before,
            "after": after,
            "revision": ledger.get("revision"),
            "note": "local ledger mutation is PROPOSED until Forge or human promotion",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kubrick graph/ledger operators")
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("saturation-score")
    a.add_argument("--ledger", required=True)
    a.add_argument("--output")

    a = sub.add_parser("counterpoint")
    a.add_argument("--packet", required=True)
    a.add_argument("--output")

    a = sub.add_parser("convergence-lock")
    a.add_argument("--graph", required=True)
    a.add_argument("--site-id")
    a.add_argument("--output")

    a = sub.add_parser("surface-occult-audit")
    a.add_argument("--input", required=True)
    a.add_argument("--output")

    a = sub.add_parser("symbolic-architecture-export")
    a.add_argument("--graph", required=True)
    a.add_argument("--ledger")
    a.add_argument("--brief")
    a.add_argument("--output")

    a = sub.add_parser("motif-mutation")
    a.add_argument("--ledger", required=True)
    a.add_argument("--motif-id", required=True)
    a.add_argument("--observed-form", default="")
    a.add_argument("--state", required=True)
    a.add_argument("--mutation", required=True)
    a.add_argument("--write-ledger")
    a.add_argument("--output")

    args = parser.parse_args()
    out = Path(args.output) if getattr(args, "output", None) else None

    if args.cmd == "saturation-score":
        receipt = score_saturation(load(Path(args.ledger)))
    elif args.cmd == "counterpoint":
        receipt = counterpoint(load(Path(args.packet)))
    elif args.cmd == "convergence-lock":
        receipt = lock_convergence(load(Path(args.graph)), args.site_id)
    elif args.cmd == "surface-occult-audit":
        path = Path(args.input)
        if path.suffix in {".yaml", ".yml", ".json"}:
            receipt = surface_occult_audit(load(path))
        else:
            receipt = surface_occult_audit(path.read_text(encoding="utf-8"))
    elif args.cmd == "symbolic-architecture-export":
        receipt = export_symbolic_architecture(
            load(Path(args.graph)),
            load(Path(args.ledger)) if args.ledger else None,
            load(Path(args.brief)) if args.brief else None,
        )
    elif args.cmd == "motif-mutation":
        ledger = load(Path(args.ledger))
        receipt = mutate_motif(ledger, args.motif_id, args.observed_form, args.state, args.mutation)
        if args.write_ledger and receipt["status"] == "OK":
            dump(ledger, Path(args.write_ledger))
    else:
        raise SystemExit(f"unknown command {args.cmd}")

    dump(receipt, out)
    raise SystemExit(0 if receipt["status"] in {"OK", "PASS", "WARN"} else 1)


if __name__ == "__main__":
    main()
