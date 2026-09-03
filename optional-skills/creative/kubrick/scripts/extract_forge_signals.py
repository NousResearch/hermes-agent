#!/usr/bin/env python3
"""Extract multi-signal observations from Continuity Forge outcome artifacts.

Forge remains the canonical authority. This script only emits OBSERVATION bundles
for retrieval, evolution proposals, and project ledger rehydration.
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


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
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


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def motif_ids(items: Any) -> list[str]:
    out: list[str] = []
    for item in as_list(items):
        if isinstance(item, dict):
            mid = item.get("motif_id") or item.get("id") or item.get("pattern_id")
            if mid:
                out.append(str(mid))
        elif item is not None:
            out.append(str(item))
    return out


def detect_kind(paths: list[Path], payloads: list[dict[str, Any]]) -> str:
    kinds: set[str] = set()
    for path, data in zip(paths, payloads):
        name = path.name.lower()
        keys = set(data.keys())
        nested = data.get("symbolic_revision_diff") or data.get("revision") or {}
        if "ledger" in name or "before" in keys or "after" in keys or "ledger_diff" in keys:
            kinds.add("ledger_diff")
        if "revision" in name or "symbolic_revision_diff" in keys or nested:
            kinds.add("revision_record")
        if "saturation" in name or "saturation_score" in keys or "saturation" in keys:
            kinds.add("saturation_report")
        if "collision" in name or "collisions" in keys:
            kinds.add("collision_report")
        if "ingest" in name or "ingestion" in keys or data.get("status") in {"COMMITTED", "REJECTED", "PARTIAL"}:
            kinds.add("ingestion_receipt")
        if "payoff" in name or "completed_payoffs" in keys or "unresolved_payoffs" in keys or "payoff" in keys:
            kinds.add("payoff_record")
    if not kinds:
        return "mixed"
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def extract_ledger_delta(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    before = after = None
    for data in payloads:
        if "before" in data and "after" in data:
            before, after = data["before"], data["after"]
            break
        if "ledger_diff" in data and isinstance(data["ledger_diff"], dict):
            diff = data["ledger_diff"]
            return {
                "motifs_added": motif_ids(diff.get("motifs_added")),
                "motifs_removed": motif_ids(diff.get("motifs_removed")),
                "motifs_mutated": motif_ids(diff.get("motifs_mutated")),
                "debt_delta": diff.get("debt_delta"),
                "saturation_delta": diff.get("saturation_delta"),
                "revision_span": diff.get("revision_span"),
            }
    if not isinstance(before, dict) or not isinstance(after, dict):
        # single ledger snapshot as after-only observation
        for data in payloads:
            if "active_motifs" in data or data.get("schema_version") == "1.0.0" and "project_id" in data:
                after = data
                break
        if not isinstance(after, dict):
            return {
                "motifs_added": [],
                "motifs_removed": [],
                "motifs_mutated": [],
                "debt_delta": None,
                "saturation_delta": None,
                "revision_span": None,
            }
        return {
            "motifs_added": motif_ids(after.get("active_motifs")),
            "motifs_removed": [],
            "motifs_mutated": [],
            "debt_delta": None,
            "saturation_delta": None,
            "revision_span": after.get("revision"),
        }
    before_ids = set(motif_ids(before.get("active_motifs")))
    after_ids = set(motif_ids(after.get("active_motifs")))
    before_states = {
        m.get("motif_id"): (m.get("current_state"), m.get("last_mutation"), m.get("recurrence_count"))
        for m in as_list(before.get("active_motifs"))
        if isinstance(m, dict) and m.get("motif_id")
    }
    after_states = {
        m.get("motif_id"): (m.get("current_state"), m.get("last_mutation"), m.get("recurrence_count"))
        for m in as_list(after.get("active_motifs"))
        if isinstance(m, dict) and m.get("motif_id")
    }
    mutated = sorted(
        mid for mid in before_ids & after_ids if before_states.get(mid) != after_states.get(mid)
    )
    debt_b = float(before.get("symbolic_debt", 0) or 0)
    debt_a = float(after.get("symbolic_debt", 0) or 0)
    sat_b = float(before.get("saturation_score", 0) or 0)
    sat_a = float(after.get("saturation_score", 0) or 0)
    rev_b = int(before.get("revision", 0) or 0)
    rev_a = int(after.get("revision", 0) or 0)
    return {
        "motifs_added": sorted(after_ids - before_ids),
        "motifs_removed": sorted(before_ids - after_ids),
        "motifs_mutated": mutated,
        "debt_delta": round(debt_a - debt_b, 4),
        "saturation_delta": round(sat_a - sat_b, 4),
        "revision_span": max(0, rev_a - rev_b),
    }


def extract_revision(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "preserved",
        "removed",
        "weakened",
        "strengthened",
        "orphaned_setups",
        "broken_payoffs",
        "required_repairs",
    ]
    out = {f: [] for f in fields}
    for data in payloads:
        rev = data.get("symbolic_revision_diff") or data.get("revision") or data
        if not isinstance(rev, dict):
            continue
        for field in fields:
            for item in as_list(rev.get(field)):
                text = item if not isinstance(item, dict) else item.get("motif_id") or item.get("note") or json.dumps(item, sort_keys=True)
                if text and str(text) not in out[field]:
                    out[field].append(str(text))
    return out


def extract_saturation(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    score = debt = None
    trend = "unknown"
    deltas: list[float] = []
    for data in payloads:
        if "saturation" in data and isinstance(data["saturation"], dict):
            sat = data["saturation"]
            if sat.get("score") is not None:
                score = float(sat["score"])
            if sat.get("debt") is not None:
                debt = float(sat["debt"])
            if sat.get("trend") in {"rising", "falling", "stable", "unknown"}:
                trend = sat["trend"]
        if data.get("saturation_score") is not None:
            score = float(data["saturation_score"])
        if data.get("symbolic_debt") is not None:
            debt = float(data["symbolic_debt"])
        if data.get("saturation_delta") is not None:
            deltas.append(float(data["saturation_delta"]))
        if isinstance(data.get("after"), dict) and data["after"].get("saturation_score") is not None:
            score = float(data["after"]["saturation_score"])
            debt = float(data["after"].get("symbolic_debt", debt or 0) or 0)
            if isinstance(data.get("before"), dict) and data["before"].get("saturation_score") is not None:
                deltas.append(score - float(data["before"]["saturation_score"]))
    if deltas:
        mean = sum(deltas) / len(deltas)
        trend = "rising" if mean > 0.02 else "falling" if mean < -0.02 else "stable"
    return {"score": score, "trend": trend, "debt": debt}


def extract_collisions(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    types: list[str] = []
    pattern_ids: list[str] = []
    count = 0
    for data in payloads:
        collisions = data.get("collisions")
        if collisions is None and isinstance(data.get("after"), dict):
            collisions = data["after"].get("collisions")
        for item in as_list(collisions):
            count += 1
            if isinstance(item, dict):
                if item.get("type"):
                    types.append(str(item["type"]))
                pattern_ids.extend(str(x) for x in as_list(item.get("patterns")))
            else:
                types.append(str(item))
    return {
        "count": count,
        "types": sorted(set(types)),
        "pattern_ids": sorted(set(pattern_ids)),
    }


def extract_ingestion(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    status = None
    motif_ok = mutation_ok = None
    notes: list[str] = []
    for data in payloads:
        ing = data.get("ingestion") if isinstance(data.get("ingestion"), dict) else data
        cand = ing.get("status")
        if cand in {"COMMITTED", "REJECTED", "PARTIAL", "UNKNOWN"}:
            status = cand
        if "motif_identity_preserved" in ing:
            motif_ok = bool(ing["motif_identity_preserved"])
        if "mutation_history_preserved" in ing:
            mutation_ok = bool(ing["mutation_history_preserved"])
        for note in as_list(ing.get("notes")):
            notes.append(str(note))
        if data.get("committed") is True:
            status = "COMMITTED"
        if data.get("rejected") is True:
            status = "REJECTED"
    return {
        "status": status,
        "motif_identity_preserved": motif_ok,
        "mutation_history_preserved": mutation_ok,
        "notes": notes,
    }


def extract_payoff(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    realized: list[str] = []
    unresolved: list[str] = []
    failed: list[str] = []
    for data in payloads:
        sources = [data, data.get("after") if isinstance(data.get("after"), dict) else {}, data.get("payoff") if isinstance(data.get("payoff"), dict) else {}]
        for src in sources:
            if not isinstance(src, dict):
                continue
            for item in as_list(src.get("completed_payoffs") or src.get("realized")):
                realized.append(str(item if not isinstance(item, dict) else item.get("id") or item.get("payoff") or item))
            for item in as_list(src.get("unresolved_payoffs") or src.get("unresolved")):
                unresolved.append(str(item if not isinstance(item, dict) else item.get("id") or item.get("payoff") or item))
            for item in as_list(src.get("failed_payoffs") or src.get("failed") or src.get("broken_payoffs")):
                failed.append(str(item if not isinstance(item, dict) else item.get("id") or item.get("payoff") or item))
    return {
        "realized": sorted(set(realized)),
        "unresolved": sorted(set(unresolved)),
        "failed": sorted(set(failed)),
    }


def build_pattern_evidence(
    payloads: list[dict[str, Any]],
    project_id: str,
    collisions: dict[str, Any],
    payoff: dict[str, Any],
    ledger_delta: dict[str, Any],
) -> list[dict[str, Any]]:
    by_pattern: dict[str, dict[str, Any]] = {}

    def ensure(pid: str) -> dict[str, Any]:
        if pid not in by_pattern:
            by_pattern[pid] = {
                "pattern_id": pid,
                "evidence_of_use": [],
                "source_projects": [project_id],
                "outcome_confidence": 0.5,
                "debt_contribution": 0.0,
                "collision_count": 0,
                "payoff_status": "unknown",
            }
        return by_pattern[pid]

    for data in payloads:
        for pid in as_list(data.get("pattern_ids") or data.get("selected_patterns") or []):
            if isinstance(pid, dict):
                pid = pid.get("pattern_id") or pid.get("primary")
            if not pid:
                continue
            rec = ensure(str(pid))
            rec["evidence_of_use"].append("explicit_pattern_reference")
        for motif in as_list(data.get("active_motifs") if "active_motifs" in data else (data.get("after") or {}).get("active_motifs")):
            if not isinstance(motif, dict):
                continue
            for link in as_list(motif.get("pattern_links")):
                rec = ensure(str(link))
                rec["evidence_of_use"].append(f"motif:{motif.get('motif_id', 'unknown')}")
                if motif.get("motif_id") in ledger_delta.get("motifs_mutated", []):
                    rec["evidence_of_use"].append("mutation_observed")
                    rec["outcome_confidence"] = max(rec["outcome_confidence"], 0.7)
                if motif.get("motif_id") in ledger_delta.get("motifs_removed", []):
                    rec["evidence_of_use"].append("motif_removed")
                    rec["outcome_confidence"] = min(rec["outcome_confidence"], 0.35)

    for pid in collisions.get("pattern_ids", []):
        rec = ensure(str(pid))
        rec["collision_count"] += 1
        rec["evidence_of_use"].append("collision_detected")
        rec["outcome_confidence"] = min(rec["outcome_confidence"], 0.4)
        rec["debt_contribution"] = round(rec["debt_contribution"] + 0.1, 4)

    # payoff linkage is project-level; attach to all observed patterns when present
    if payoff["failed"]:
        for rec in by_pattern.values():
            rec["payoff_status"] = "failed"
            rec["evidence_of_use"].append("payoff_failed")
            rec["outcome_confidence"] = min(rec["outcome_confidence"], 0.3)
            rec["debt_contribution"] = round(rec["debt_contribution"] + 0.15, 4)
    elif payoff["realized"] and not payoff["unresolved"]:
        for rec in by_pattern.values():
            rec["payoff_status"] = "realized"
            rec["evidence_of_use"].append("payoff_realized")
            rec["outcome_confidence"] = max(rec["outcome_confidence"], 0.8)
    elif payoff["unresolved"]:
        for rec in by_pattern.values():
            if rec["payoff_status"] == "unknown":
                rec["payoff_status"] = "unresolved"
                rec["evidence_of_use"].append("payoff_unresolved")

    results = []
    for rec in by_pattern.values():
        rec["evidence_of_use"] = sorted(set(rec["evidence_of_use"])) or ["forge_observation"]
        rec["source_projects"] = sorted(set(rec["source_projects"]))
        results.append(rec)
    return sorted(results, key=lambda r: r["pattern_id"])


def build_bundle(project_id: str, paths: list[Path], forge_document_key: str | None, forge_state_hash: str | None) -> dict[str, Any]:
    payloads = [load(p) for p in paths]
    kind = detect_kind(paths, payloads)
    ledger_delta = extract_ledger_delta(payloads)
    revision = extract_revision(payloads)
    saturation = extract_saturation(payloads)
    collisions = extract_collisions(payloads)
    ingestion = extract_ingestion(payloads)
    payoff = extract_payoff(payloads)
    pattern_evidence = build_pattern_evidence(payloads, project_id, collisions, payoff, ledger_delta)
    errors: list[str] = []
    if not paths:
        errors.append("no source paths")
    if saturation["score"] is not None and not 0 <= saturation["score"] <= 1:
        errors.append("saturation score out of range")
    core = {
        "schema_version": "1.0.0",
        "project_id": project_id,
        "source": {
            "kind": kind,
            "paths": [str(p) for p in paths],
            "forge_document_key": forge_document_key,
            "forge_state_hash": forge_state_hash,
        },
        "signals": {
            "ledger_delta": ledger_delta,
            "revision": revision,
            "saturation": saturation,
            "collisions": collisions,
            "ingestion": ingestion,
            "payoff": payoff,
        },
        "pattern_evidence": pattern_evidence,
        "authority": {
            "state": "OBSERVATION",
            "forge_canonical": True,
            "automatic_corpus_change_allowed": False,
        },
        "validation": {
            "status": "VALID" if not errors else "INVALID",
            "errors": errors,
        },
    }
    core["bundle_id"] = hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return core


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract multi-signal Forge observation bundles")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--input", action="append", required=True, help="Forge artifact path (repeatable)")
    parser.add_argument("--forge-document-key")
    parser.add_argument("--forge-state-hash")
    parser.add_argument("--output")
    args = parser.parse_args()
    paths = [Path(p) for p in args.input]
    bundle = build_bundle(args.project_id, paths, args.forge_document_key, args.forge_state_hash)
    dump(bundle, Path(args.output) if args.output else None)
    raise SystemExit(0 if bundle["validation"]["status"] == "VALID" else 1)


if __name__ == "__main__":
    main()
