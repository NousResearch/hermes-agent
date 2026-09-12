#!/usr/bin/env python3
"""Initialize, audit, mutate, rehydrate, and export Kubrick project symbolic ledgers.

Ledgers are first-class persistent artifacts consumed by retrieval and evolution.
Structural authority still requires human or Forge promotion.
"""
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


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text) or {}
    return yaml.safe_load(text) or {}


def dump(data: dict[str, Any], path: Path | None = None) -> None:
    text = yaml.safe_dump(data, sort_keys=False)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init(project_id: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "project_id": project_id,
        "governing_grammar": None,
        "supporting_grammars": [],
        "active_motifs": [],
        "retired_motifs": [],
        "prohibited_motifs": [],
        "unresolved_payoffs": [],
        "completed_payoffs": [],
        "collisions": [],
        "cultural_boundaries": [],
        "symbolic_debt": 0.0,
        "saturation_score": 0.0,
        "revision": 0,
        "updated_at": now(),
        "pattern_history": [],
        "evidence_log": [],
        "authority": {"state": "PROPOSED", "forge_canonical": False},
    }


def audit(d: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if d.get("schema_version") != "1.0.0":
        errors.append("unsupported schema_version")
    if not d.get("project_id"):
        errors.append("project_id required")
    if len(d.get("supporting_grammars", [])) > 2:
        errors.append("supporting_grammars exceeds 2")
    ids = [m.get("motif_id") for m in d.get("active_motifs", []) if isinstance(m, dict)]
    if len(ids) != len(set(ids)):
        errors.append("duplicate active motif ids")
    try:
        sat = float(d.get("saturation_score", 0))
        if not 0 <= sat <= 1:
            errors.append("saturation_score outside 0..1")
    except (TypeError, ValueError):
        errors.append("saturation_score not numeric")
    try:
        if float(d.get("symbolic_debt", 0)) < 0:
            errors.append("symbolic_debt below 0")
    except (TypeError, ValueError):
        errors.append("symbolic_debt not numeric")
    return {
        "status": "VALID" if not errors else "INVALID",
        "errors": errors,
        "active_motif_count": len(ids),
        "revision": d.get("revision", 0),
        "saturation_score": d.get("saturation_score", 0),
        "symbolic_debt": d.get("symbolic_debt", 0),
        "pattern_history_count": len(d.get("pattern_history") or []),
    }


def recompute_saturation(d: dict[str, Any]) -> None:
    motifs = d.get("active_motifs") or []
    if not motifs:
        d["saturation_score"] = 0.0
        return
    recurrences = sum(int(m.get("recurrence_count", 0) or 0) for m in motifs if isinstance(m, dict))
    collisions = len(d.get("collisions") or [])
    unresolved = len(d.get("unresolved_payoffs") or [])
    density = min(1.0, (len(motifs) * 0.12) + (recurrences * 0.04) + (collisions * 0.08) + (unresolved * 0.05))
    d["saturation_score"] = round(density, 4)
    d["symbolic_debt"] = round(
        max(0.0, float(d.get("symbolic_debt", 0) or 0) + collisions * 0.05 + unresolved * 0.03),
        4,
    )


def append_evidence(d: dict[str, Any], kind: str, detail: str, refs: list[str] | None = None) -> None:
    log = d.setdefault("evidence_log", [])
    log.append(
        {
            "at": now(),
            "kind": kind,
            "detail": detail,
            "refs": refs or [],
            "revision": d.get("revision", 0),
        }
    )


def record_pattern_use(
    d: dict[str, Any],
    pattern_id: str,
    source_project: str | None,
    outcome_confidence: float,
    evidence: list[str],
) -> None:
    history = d.setdefault("pattern_history", [])
    existing = next((h for h in history if h.get("pattern_id") == pattern_id), None)
    project = source_project or d.get("project_id") or "unknown"
    if existing is None:
        history.append(
            {
                "pattern_id": pattern_id,
                "evidence_of_use": list(evidence),
                "source_projects": [project],
                "outcome_confidence": outcome_confidence,
                "use_count": 1,
            }
        )
    else:
        existing["use_count"] = int(existing.get("use_count", 0) or 0) + 1
        existing["evidence_of_use"] = sorted(set((existing.get("evidence_of_use") or []) + list(evidence)))
        projects = set(existing.get("source_projects") or [])
        projects.add(project)
        existing["source_projects"] = sorted(projects)
        # running mean toward new observation
        prev = float(existing.get("outcome_confidence", 0.5) or 0.5)
        existing["outcome_confidence"] = round((prev * 0.7) + (outcome_confidence * 0.3), 4)


def mutate(
    d: dict[str, Any],
    motif_id: str,
    observed_form: str,
    state: str,
    mutation: str,
    ownership: str | None = None,
    pattern_links: list[str] | None = None,
    convergence_sites: list[str] | None = None,
) -> dict[str, Any]:
    motifs = d.setdefault("active_motifs", [])
    existing = next((m for m in motifs if m.get("motif_id") == motif_id), None)
    if existing:
        existing["current_state"] = state
        existing["last_mutation"] = mutation
        existing["recurrence_count"] = int(existing.get("recurrence_count", 0)) + 1
        if ownership is not None:
            existing["ownership"] = ownership
        if pattern_links:
            existing["pattern_links"] = sorted(set((existing.get("pattern_links") or []) + pattern_links))
        if convergence_sites:
            existing["convergence_sites"] = sorted(set((existing.get("convergence_sites") or []) + convergence_sites))
        detail = f"mutated {motif_id} via {mutation}"
    else:
        motifs.append(
            {
                "motif_id": motif_id,
                "observed_form": observed_form,
                "current_state": state,
                "recurrence_count": 1,
                "last_mutation": mutation,
                "ownership": ownership,
                "pattern_links": pattern_links or [],
                "convergence_sites": convergence_sites or [],
            }
        )
        detail = f"introduced {motif_id}"
    d["revision"] = int(d.get("revision", 0)) + 1
    d["updated_at"] = now()
    recompute_saturation(d)
    append_evidence(d, "mutation", detail, pattern_links or [])
    for link in pattern_links or []:
        record_pattern_use(d, link, d.get("project_id"), 0.6, [f"ledger_mutation:{motif_id}"])
    return d


def rehydrate_from_forge(export_data: dict[str, Any], project_id: str | None = None) -> dict[str, Any]:
    """Rehydrate a project ledger from a Forge symbolic export or ledger snapshot."""
    if export_data.get("schema_version") == "1.0.0" and "active_motifs" in export_data:
        ledger = dict(export_data)
        ledger.setdefault("pattern_history", [])
        ledger.setdefault("evidence_log", [])
        ledger.setdefault("authority", {"state": "PROPOSED", "forge_canonical": True})
        ledger["authority"]["forge_canonical"] = True
        ledger["updated_at"] = now()
        return ledger

    arch = export_data.get("symbolic_architecture") or export_data
    motifs_src = (
        arch.get("motif_registry")
        or export_data.get("forge_mappings", {}).get("ledger_motifs")
        or export_data.get("motifs")
        or []
    )
    pid = project_id or export_data.get("project_id") or arch.get("project_id") or "forge-project"
    ledger = init(pid)
    ledger["authority"] = {"state": "PROPOSED", "forge_canonical": True}
    for item in motifs_src:
        if isinstance(item, str):
            mutate(ledger, item, item, "present", "rehydrated_from_forge")
            continue
        if not isinstance(item, dict):
            continue
        mid = str(item.get("motif_id") or item.get("id") or "motif")
        mutate(
            ledger,
            mid,
            str(item.get("observed_form") or mid),
            str(item.get("current_state") or item.get("state") or "present"),
            str(item.get("last_mutation") or "rehydrated_from_forge"),
            ownership=item.get("ownership"),
            pattern_links=[str(x) for x in (item.get("pattern_links") or [])],
            convergence_sites=[str(x) for x in (item.get("convergence_sites") or [])],
        )
    for payoff in arch.get("completed_payoffs") or export_data.get("completed_payoffs") or []:
        ledger["completed_payoffs"].append(str(payoff))
    for payoff in arch.get("unresolved_payoffs") or export_data.get("unresolved_payoffs") or []:
        ledger["unresolved_payoffs"].append(str(payoff))
    append_evidence(ledger, "rehydrate", "rehydrated from forge export", [str(export_data.get("bundle_id") or "forge-export")])
    recompute_saturation(ledger)
    return ledger


def apply_forge_bundle(ledger: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    if bundle.get("validation", {}).get("status") not in {None, "VALID"}:
        raise SystemExit("forge bundle is not VALID")
    signals = bundle.get("signals") or {}
    delta = signals.get("ledger_delta") or {}
    for mid in delta.get("motifs_removed") or []:
        active = ledger.get("active_motifs") or []
        ledger["active_motifs"] = [m for m in active if m.get("motif_id") != mid]
        retired = set(ledger.get("retired_motifs") or [])
        retired.add(mid)
        ledger["retired_motifs"] = sorted(retired)
    for mid in delta.get("motifs_added") or []:
        if not any(m.get("motif_id") == mid for m in ledger.get("active_motifs") or []):
            mutate(ledger, mid, mid, "present", "forge_added")
    for mid in delta.get("motifs_mutated") or []:
        existing = next((m for m in ledger.get("active_motifs") or [] if m.get("motif_id") == mid), None)
        if existing:
            mutate(
                ledger,
                mid,
                existing.get("observed_form", mid),
                existing.get("current_state", "mutated"),
                "forge_mutated",
                ownership=existing.get("ownership"),
                pattern_links=existing.get("pattern_links") or [],
            )
    sat = signals.get("saturation") or {}
    if sat.get("score") is not None:
        ledger["saturation_score"] = float(sat["score"])
    if sat.get("debt") is not None:
        ledger["symbolic_debt"] = float(sat["debt"])
    collisions = signals.get("collisions") or {}
    if collisions.get("count"):
        for pid in collisions.get("pattern_ids") or []:
            ledger.setdefault("collisions", []).append(
                {"type": "REDUNDANT", "patterns": [pid], "rationale": "imported from forge collision report"}
            )
    payoff = signals.get("payoff") or {}
    for item in payoff.get("realized") or []:
        if item not in ledger.get("completed_payoffs", []):
            ledger.setdefault("completed_payoffs", []).append(item)
        ledger["unresolved_payoffs"] = [p for p in ledger.get("unresolved_payoffs", []) if p != item]
    for item in payoff.get("unresolved") or []:
        if item not in ledger.get("unresolved_payoffs", []) and item not in ledger.get("completed_payoffs", []):
            ledger.setdefault("unresolved_payoffs", []).append(item)
    for item in payoff.get("failed") or []:
        if item not in ledger.get("unresolved_payoffs", []):
            ledger.setdefault("unresolved_payoffs", []).append(item)
        ledger["symbolic_debt"] = round(float(ledger.get("symbolic_debt", 0) or 0) + 0.1, 4)
    for evidence in bundle.get("pattern_evidence") or []:
        record_pattern_use(
            ledger,
            str(evidence["pattern_id"]),
            bundle.get("project_id"),
            float(evidence.get("outcome_confidence", 0.5)),
            list(evidence.get("evidence_of_use") or ["forge_bundle"]),
        )
    ledger["revision"] = int(ledger.get("revision", 0)) + 1
    ledger["updated_at"] = now()
    ledger.setdefault("authority", {})["forge_canonical"] = True
    append_evidence(ledger, "forge_bundle", f"applied forge bundle {bundle.get('bundle_id')}", [bundle.get("bundle_id", "")])
    return ledger


def export_for_retrieval(d: dict[str, Any]) -> dict[str, Any]:
    """Compact ledger snapshot consumed by retrieval scoring."""
    return {
        "project_id": d.get("project_id"),
        "governing_grammar": d.get("governing_grammar"),
        "supporting_grammars": d.get("supporting_grammars") or [],
        "active_motifs": [
            {
                "motif_id": m.get("motif_id"),
                "observed_form": m.get("observed_form"),
                "current_state": m.get("current_state"),
                "recurrence_count": m.get("recurrence_count"),
                "pattern_links": m.get("pattern_links") or [],
            }
            for m in d.get("active_motifs") or []
            if isinstance(m, dict)
        ],
        "retired_motifs": d.get("retired_motifs") or [],
        "prohibited_motifs": d.get("prohibited_motifs") or [],
        "unresolved_payoffs": d.get("unresolved_payoffs") or [],
        "completed_payoffs": d.get("completed_payoffs") or [],
        "collisions": d.get("collisions") or [],
        "cultural_boundaries": d.get("cultural_boundaries") or [],
        "symbolic_debt": d.get("symbolic_debt", 0),
        "saturation_score": d.get("saturation_score", 0),
        "pattern_history": d.get("pattern_history") or [],
        "revision": d.get("revision", 0),
        "snapshot_hash": hashlib.sha256(
            json.dumps(
                {
                    "project_id": d.get("project_id"),
                    "motifs": [m.get("motif_id") for m in d.get("active_motifs") or []],
                    "revision": d.get("revision", 0),
                    "saturation": d.get("saturation_score", 0),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("init")
    a.add_argument("--project-id", required=True)
    a.add_argument("--out", required=True)

    a = sub.add_parser("audit")
    a.add_argument("--ledger", required=True)

    a = sub.add_parser("mutate")
    a.add_argument("--ledger", required=True)
    a.add_argument("--motif-id", required=True)
    a.add_argument("--observed-form", required=True)
    a.add_argument("--state", required=True)
    a.add_argument("--mutation", required=True)
    a.add_argument("--ownership")
    a.add_argument("--pattern-link", action="append", default=[])
    a.add_argument("--convergence-site", action="append", default=[])
    a.add_argument("--out")

    a = sub.add_parser("rehydrate")
    a.add_argument("--forge-export", required=True)
    a.add_argument("--project-id")
    a.add_argument("--out", required=True)

    a = sub.add_parser("apply-forge")
    a.add_argument("--ledger", required=True)
    a.add_argument("--forge-bundle", required=True)
    a.add_argument("--out")

    a = sub.add_parser("export-retrieval")
    a.add_argument("--ledger", required=True)
    a.add_argument("--out")

    a = sub.add_parser("record-pattern")
    a.add_argument("--ledger", required=True)
    a.add_argument("--pattern-id", required=True)
    a.add_argument("--outcome-confidence", type=float, default=0.5)
    a.add_argument("--evidence", action="append", default=["manual_record"])
    a.add_argument("--out")

    args = parser.parse_args()

    if args.cmd == "init":
        dump(init(args.project_id), Path(args.out))
        return

    if args.cmd == "rehydrate":
        data = rehydrate_from_forge(load(Path(args.forge_export)), args.project_id)
        dump(data, Path(args.out))
        return

    path = Path(args.ledger)
    data = load(path)

    if args.cmd == "audit":
        report = audit(data)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["status"] == "VALID" else 1)

    if args.cmd == "mutate":
        mutate(
            data,
            args.motif_id,
            args.observed_form,
            args.state,
            args.mutation,
            ownership=args.ownership,
            pattern_links=args.pattern_link,
            convergence_sites=args.convergence_site,
        )
        out = Path(args.out) if args.out else path
        dump(data, out)
        return

    if args.cmd == "apply-forge":
        apply_forge_bundle(data, load(Path(args.forge_bundle)))
        out = Path(args.out) if args.out else path
        dump(data, out)
        return

    if args.cmd == "export-retrieval":
        snap = export_for_retrieval(data)
        if args.out:
            dump(snap, Path(args.out))
        else:
            print(yaml.safe_dump(snap, sort_keys=False))
        return

    if args.cmd == "record-pattern":
        record_pattern_use(data, args.pattern_id, data.get("project_id"), args.outcome_confidence, args.evidence)
        data["revision"] = int(data.get("revision", 0)) + 1
        data["updated_at"] = now()
        out = Path(args.out) if args.out else path
        dump(data, out)
        return


if __name__ == "__main__":
    main()
