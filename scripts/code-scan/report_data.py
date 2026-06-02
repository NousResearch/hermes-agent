#!/usr/bin/env python3
"""report_data.py — Phase 4 D7a: Deterministic report data builder.

Combines upstream UA artifacts (scan, classified-imports, entrypoints,
graph, orphan-triage, hub-ranking, semantic-signals, delta, readiness)
into a single intermediate JSON report-data model.

Does NOT render Markdown — that is D7b (markdown renderer).

Usage:
    python scripts/code-scan/report_data.py --scan scan.json \
        --classified-imports classified-imports.json \
        --entrypoints entrypoints.json --graph graph.json \
        --orphan-triage orphan-triage.json --hubs hubs.json \
        --semantic semantic-signals.json --delta delta.json \
        --readiness readiness.json --output report-data.json

At minimum --scan is required. All other artifacts are optional.

Stdlib only — no external dependencies.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# ── Bounded-list limits ───────────────────────────────────────────────

MAX_READING_PLAN = 100
MAX_SUSPICIOUS_ORPHANS = 50
MAX_HUB_RANKINGS = 20

# ── Schema version ─────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0.0"

# ── Artifact source key mapping ────────────────────────────────────────

_ARTIFACT_KEYS = [
    ("scan", "scan"),
    ("classified_imports", "classified-imports"),
    ("entrypoints", "entrypoints"),
    ("graph", "graph"),
    ("orphan_triage", "orphan-triage"),
    ("hub_rankings", "hub-rankings"),
    ("semantic_signals", "semantic-signals"),
    ("delta", "delta"),
    ("readiness", "readiness"),
]

# ── File path lookup helpers ───────────────────────────────────────────

_SCAN_LOOKUP: dict = {}  # populated once per build_report_data call


def _scan_lookup_init(scan_data: dict) -> None:
    """Build a fast path → relative_path lookup from scan data."""
    global _SCAN_LOOKUP
    _SCAN_LOOKUP = {}
    for f in scan_data.get("files", []):
        rel = f.get("relative_path", "")
        abs_path = f.get("path", "")
        if rel:
            _SCAN_LOOKUP[rel] = rel
        if abs_path:
            _SCAN_LOOKUP[abs_path] = rel


def _resolve_file_path(
    file_hint: str, node_id: str = ""
) -> str:
    """Resolve a reference from graph/orphan/hub data to a file path.

    Uses the scan lookup table first, falls back to the hint itself.
    """
    if file_hint in _SCAN_LOOKUP:
        return _SCAN_LOOKUP[file_hint]
    # Might be a node_id; try to find file_path in scan
    for path, rel in _SCAN_LOOKUP.items():
        if node_id in path or node_id == path:
            return rel
    return file_hint


# ── JSON loading (tolerant) ────────────────────────────────────────────


def _load_artifact(
    path: Optional[str],
    label: str,
    warnings: list[str],
) -> Optional[dict]:
    """Load a JSON artifact from *path*, returning None on any error.

    Appends a warning if the artifact could not be loaded.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        warnings.append(f"{label}: file not found ({path})")
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            warnings.append(f"{label}: expected JSON object, got {type(data).__name__}")
            return None
        return data
    except (json.JSONDecodeError, OSError) as exc:
        warnings.append(f"{label}: invalid or unreadable JSON ({exc})")
        return None


# ── Section builders ───────────────────────────────────────────────────


def _build_scan_section(scan_data: dict) -> dict:
    """Build the scan summary section."""
    return {
        "project_root": scan_data.get("project_root", ""),
        "scanned_at": scan_data.get("scanned_at", ""),
        "total_files": scan_data.get("total_files", 0),
        "total_lines": scan_data.get("total_lines", 0),
        "languages": dict(sorted(scan_data.get("languages", {}).items())),
        "categories": dict(sorted(scan_data.get("categories", {}).items())),
        "frameworks": sorted(scan_data.get("frameworks", [])),
    }


def _build_classification_section(
    classified: Optional[dict],
    warnings: list[str],
) -> Any:
    if classified is None:
        return "not_available"
    try:
        # Validate expected structure
        if "totals" not in classified and "files" not in classified:
            warnings.append(
                "classified_imports: missing expected keys (totals/files)"
            )
            return "not_available"
        totals = classified.get("totals", {})
        file_count = len(classified.get("files", {}))
        return {
            "totals": {
                "stdlib": totals.get("stdlib", 0),
                "third_party": totals.get("third_party", 0),
                "local": totals.get("local", 0),
                "relative": totals.get("relative", 0),
                "unknown": totals.get("unknown", 0),
            },
            "files_classified": file_count,
        }
    except Exception as exc:
        warnings.append(f"classification: error extracting data ({exc})")
        return "not_available"


def _build_entrypoints_section(
    entrypoints: Optional[dict],
    warnings: list[str],
) -> Any:
    if entrypoints is None:
        return "not_available"
    try:
        # Validate expected structure
        if "entrypoints" not in entrypoints and "totals" not in entrypoints:
            warnings.append(
                "entrypoints: missing expected keys (entrypoints/totals)"
            )
            return "not_available"
        ep_list = entrypoints.get("entrypoints", [])
        return {
            "entrypoints": sorted(
                [
                    {
                        "file": ep.get("file", ""),
                        "type": ep.get("type", ""),
                        "confidence": ep.get("confidence", 0),
                        "signals": ep.get("signals", []),
                    }
                    for ep in ep_list
                ],
                key=lambda e: (-e["confidence"], e["file"]),
            ),
            "totals": entrypoints.get("totals", {}),
        }
    except Exception as exc:
        warnings.append(f"entrypoints: error extracting data ({exc})")
        return "not_available"


def _build_graph_section(
    graph: Optional[dict],
    warnings: list[str],
) -> Any:
    if graph is None:
        return "not_available"
    try:
        analytics = graph.get("analytics", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        file_nodes = [
            n for n in nodes if n.get("node_type") == "file"
        ]
        return {
            "nodes_count": len(nodes),
            "file_nodes_count": len(file_nodes),
            "edges_count": len(edges),
            "analytics": analytics,
            "warnings": graph.get("warnings", []),
        }
    except Exception as exc:
        warnings.append(f"graph: error extracting data ({exc})")
        return "not_available"


def _build_orphan_triage_section(
    triage: Optional[dict],
    warnings: list[str],
) -> Any:
    if triage is None:
        return "not_available"
    try:
        orphans = triage.get("orphans", {})
        return {
            "categories": {
                "expected": len(orphans.get("expected", [])),
                "entrypoint_candidate": len(orphans.get("entrypoint_candidate", [])),
                "suspicious": len(orphans.get("suspicious", [])),
                "unknown": len(orphans.get("unknown", [])),
            },
            "totals": triage.get("totals", {}),
        }
    except Exception as exc:
        warnings.append(f"orphan_triage: error extracting data ({exc})")
        return "not_available"


def _build_hub_rankings_section(
    hubs: Optional[dict],
    warnings: list[str],
) -> Any:
    if hubs is None:
        return "not_available"
    try:
        rankings = hubs.get("hub_rankings", [])
        original_count = len(rankings)
        if original_count > MAX_HUB_RANKINGS:
            rankings = rankings[:MAX_HUB_RANKINGS]
            warnings.append(
                f"hub_rankings: truncated from {original_count} to {MAX_HUB_RANKINGS}"
            )
        return {
            "hubs": [
                {
                    "file_path": h.get("file_path", ""),
                    "hub_score": h.get("hub_score", 0),
                    "in_degree": h.get("in_degree", 0),
                    "out_degree": h.get("out_degree", 0),
                    "confidence": h.get("confidence", "unknown"),
                }
                for h in rankings
            ],
            "totals": hubs.get("totals", {}),
            "disclaimer": hubs.get("disclaimer", ""),
        }
    except Exception as exc:
        warnings.append(f"hub_rankings: error extracting data ({exc})")
        return "not_available"


def _build_semantic_section(
    semantic: Optional[dict],
    warnings: list[str],
) -> Any:
    if semantic is None:
        return "not_available"
    try:
        files = semantic.get("files", {})
        symbol_count = 0
        files_processed = 0
        for fdata in files.values():
            files_processed += 1
            symbol_count += len(fdata.get("symbols", []))
        return {
            "files_processed": files_processed,
            "symbols": symbol_count,
            "totals": semantic.get("totals", {}),
        }
    except Exception as exc:
        warnings.append(f"semantic_signals: error extracting data ({exc})")
        return "not_available"


def _build_delta_section(
    delta: Optional[dict],
    warnings: list[str],
) -> Any:
    if delta is None:
        return "not_available"
    try:
        files = delta.get("files", {})
        return {
            "files": {
                "added": sorted(files.get("added", [])),
                "removed": sorted(files.get("removed", [])),
                "common_count": files.get("common_count", 0),
            },
            "languages": delta.get("languages", {}),
            "categories": delta.get("categories", {}),
            "frameworks": delta.get("frameworks", {}),
        }
    except Exception as exc:
        warnings.append(f"delta: error extracting data ({exc})")
        return "not_available"


def _build_readiness_section(
    readiness: Optional[dict],
    warnings: list[str],
) -> Any:
    if readiness is None:
        return "not_available"
    try:
        return {
            "detected_stacks": readiness.get("detected_stacks", []),
            "verification_status": readiness.get("verification_status", "unknown"),
            "blockers": readiness.get("blockers", []),
        }
    except Exception as exc:
        warnings.append(f"readiness: error extracting data ({exc})")
        return "not_available"


# ── Reading-plan builders ──────────────────────────────────────────────

# Type-label map for reading plan entries
_TYPE_PRIORITY = {
    "entrypoint": "HIGH",
    "hub": "MEDIUM",
    "suspicious_orphan": "MEDIUM",
    "semantic_hotspot": "LOW",
}


def _reading_plan_from_entrypoints(
    entrypoints: Optional[dict],
    candidates: list[dict],
    warnings: list[str],
) -> None:
    if entrypoints is None:
        return
    for ep in entrypoints.get("entrypoints", []):
        file_path = ep.get("file", "")
        confidence = ep.get("confidence", 0)
        signals = ep.get("signals", [])
        candidates.append({
            "file": file_path,
            "type": "entrypoint",
            "priority": _TYPE_PRIORITY["entrypoint"],
            "confidence": confidence,
            "reason": "; ".join(signals) if signals else "detected entrypoint",
        })


def _reading_plan_from_hubs(
    hubs: Optional[dict],
    candidates: list[dict],
    warnings: list[str],
) -> None:
    if hubs is None:
        return
    rankings = hubs.get("hub_rankings", [])
    original = len(rankings)
    if original > MAX_HUB_RANKINGS:
        rankings = rankings[:MAX_HUB_RANKINGS]
    for h in rankings:
        file_path = h.get("file_path", "")
        score = h.get("hub_score", 0)
        candidates.append({
            "file": file_path,
            "type": "hub",
            "priority": _TYPE_PRIORITY["hub"],
            "confidence": min(score / 10.0, 1.0),
            "reason": f"hub_score={score}",
        })


def _resolve_orphan_file(orphans_data: dict, node_id: str, label: str = "") -> str:
    """Resolve an orphan node_id to a file path using the scan lookup."""
    # Try scanning the node_id against _SCAN_LOOKUP
    if node_id in _SCAN_LOOKUP:
        return _SCAN_LOOKUP[node_id]
    # Try as a substring match
    for path, rel in _SCAN_LOOKUP.items():
        if node_id in path:
            return rel
    # Fallback: return the label if it looks like a path
    if label and "/" in label:
        return label
    # Last resort: return the node_id itself
    return node_id


def _reading_plan_from_suspicious_orphans(
    triage: Optional[dict],
    candidates: list[dict],
    warnings: list[str],
) -> None:
    if triage is None:
        return
    suspicious = triage.get("orphans", {}).get("suspicious", [])
    original = len(suspicious)
    if original > MAX_SUSPICIOUS_ORPHANS:
        suspicious = suspicious[:MAX_SUSPICIOUS_ORPHANS]
        warnings.append(
            f"suspicious orphans: truncated from {original} to {MAX_SUSPICIOUS_ORPHANS}"
        )
    for o in suspicious:
        node_id = o.get("node_id", "")
        reason = o.get("reason", "suspicious orphan")
        file_path = _resolve_orphan_file(triage, node_id)
        candidates.append({
            "file": file_path,
            "type": "suspicious_orphan",
            "priority": _TYPE_PRIORITY["suspicious_orphan"],
            "confidence": 0.5,
            "reason": reason,
        })


def _reading_plan_from_semantic_hotspots(
    semantic: Optional[dict],
    candidates: list[dict],
    warnings: list[str],
) -> None:
    if semantic is None:
        return
    files = semantic.get("files", {})
    for file_path, fdata in sorted(files.items()):
        symbols = fdata.get("symbols", [])
        if not symbols:
            continue
        symbol_count = len(symbols)
        # Files with many symbols are "hotspots"
        candidates.append({
            "file": file_path,
            "type": "semantic_hotspot",
            "priority": _TYPE_PRIORITY["semantic_hotspot"],
            "confidence": min(symbol_count / 10.0, 0.8),
            "reason": f"{symbol_count} symbols extracted",
        })


# ── Totals builder ─────────────────────────────────────────────────────


def _build_totals(
    scan_data: Optional[dict],
    classified: Optional[dict],
    entrypoints: Optional[dict],
    graph: Optional[dict],
    triage: Optional[dict],
    hubs: Optional[dict],
    semantic: Optional[dict],
) -> dict:
    totals: dict[str, Any] = {}
    if scan_data:
        totals["total_files"] = scan_data.get("total_files", 0)
        totals["total_lines"] = scan_data.get("total_lines", 0)
        totals["languages"] = dict(sorted(scan_data.get("languages", {}).items()))
        totals["categories"] = dict(sorted(scan_data.get("categories", {}).items()))
    if classified:
        ct = classified.get("totals", {})
        totals["classified_imports"] = {
            "stdlib": ct.get("stdlib", 0),
            "third_party": ct.get("third_party", 0),
            "local": ct.get("local", 0),
            "relative": ct.get("relative", 0),
            "unknown": ct.get("unknown", 0),
        }
    if entrypoints:
        totals["entrypoints_count"] = entrypoints.get("totals", {}).get(
            "entrypoints_found", 0
        )
    if graph:
        totals["graph_edges"] = len(graph.get("edges", []))
        totals["graph_nodes"] = len(graph.get("nodes", []))
    if triage:
        totals["orphan_count"] = triage.get("totals", {}).get("total_orphans", 0)
    if hubs:
        totals["hubs_count"] = len(hubs.get("hub_rankings", []))
    if semantic:
        totals["symbol_count"] = semantic.get("totals", {}).get("symbols", 0)
    return totals


# ── Main builder function ──────────────────────────────────────────────


def build_report_data(
    scan: Optional[dict] = None,
    classified_imports: Optional[dict] = None,
    entrypoints: Optional[dict] = None,
    graph: Optional[dict] = None,
    orphan_triage: Optional[dict] = None,
    hub_rankings: Optional[dict] = None,
    semantic_signals: Optional[dict] = None,
    delta: Optional[dict] = None,
    readiness: Optional[dict] = None,
) -> dict[str, Any]:
    """Build the deterministic report-data dict from available artifacts.

    Each argument is either a parsed JSON dict (already loaded) or None.
    Returns a dict conforming to the report-data JSON schema.
    """
    if scan is None:
        raise ValueError("scan data is required (at minimum)")

    warnings: list[str] = []

    # Initialize scan lookup for orphan file resolution
    _scan_lookup_init(scan)

    # ── Build sources dict ─────────────────────────────────────
    sources: dict[str, str] = {}
    source_map = {
        "scan": scan,
        "classified_imports": classified_imports,
        "entrypoints": entrypoints,
        "graph": graph,
        "orphan_triage": orphan_triage,
        "hub_rankings": hub_rankings,
        "semantic_signals": semantic_signals,
        "delta": delta,
        "readiness": readiness,
    }
    for key in sorted(source_map.keys()):
        sources[key] = "loaded" if source_map[key] is not None else "not_provided"

    # ── Warn about missing optional artifacts ──────────────────
    optional_labels = {
        "classified_imports": "classified_imports",
        "entrypoints": "entrypoints",
        "graph": "graph",
        "orphan_triage": "orphan_triage",
        "hub_rankings": "hub_rankings",
        "semantic_signals": "semantic_signals",
        "delta": "delta",
        "readiness": "readiness",
    }
    for key, label in optional_labels.items():
        if source_map[key] is None:
            warnings.append(f"{label}: not provided")

    # ── Build sections ─────────────────────────────────────────
    sections: dict[str, Any] = {}

    sections["scan"] = _build_scan_section(scan)
    sections["classification"] = _build_classification_section(
        classified_imports, warnings
    )
    sections["entrypoints"] = _build_entrypoints_section(entrypoints, warnings)
    sections["graph_analysis"] = _build_graph_section(graph, warnings)
    sections["orphan_triage"] = _build_orphan_triage_section(
        orphan_triage, warnings
    )
    sections["hub_rankings"] = _build_hub_rankings_section(hub_rankings, warnings)
    sections["semantic_signals"] = _build_semantic_section(
        semantic_signals, warnings
    )
    sections["delta"] = _build_delta_section(delta, warnings)
    sections["readiness"] = _build_readiness_section(readiness, warnings)

    # Keep "not_available" markers so consumers can check section presence

    # ── Build reading plan ─────────────────────────────────────
    reading_plan: list[dict] = []
    _reading_plan_from_entrypoints(entrypoints, reading_plan, warnings)
    _reading_plan_from_hubs(hub_rankings, reading_plan, warnings)
    _reading_plan_from_suspicious_orphans(orphan_triage, reading_plan, warnings)
    _reading_plan_from_semantic_hotspots(semantic_signals, reading_plan, warnings)

    # Sort deterministically: confidence descending, then file ascending
    reading_plan.sort(key=lambda p: (-p.get("confidence", 0), p.get("file", "")))

    # Cap reading plan at MAX_READING_PLAN
    if len(reading_plan) > MAX_READING_PLAN:
        truncated_count = len(reading_plan) - MAX_READING_PLAN
        reading_plan = reading_plan[:MAX_READING_PLAN]
        warnings.append(
            f"reading_plan: truncated {truncated_count} candidates (limit {MAX_READING_PLAN})"
        )

    # ── Build totals ───────────────────────────────────────────
    totals = _build_totals(
        scan,
        classified_imports,
        entrypoints,
        graph,
        orphan_triage,
        hub_rankings,
        semantic_signals,
    )

    # ── Assemble report ────────────────────────────────────────
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": sources,
        "sections": sections,
        "reading_plan": reading_plan,
        "warnings": warnings,
        "totals": totals,
    }


# ── CLI entry point ────────────────────────────────────────────────────


def main() -> int:
    """CLI entry point. Reads artifact JSON files, emits report-data JSON."""
    parser = argparse.ArgumentParser(
        description="Build deterministic report-data JSON from UA scan artifacts."
    )
    parser.add_argument("--scan", required=True, help="Path to scan.json (required)")
    parser.add_argument(
        "--classified-imports",
        dest="classified_imports",
        default=None,
        help="Path to classified-imports.json",
    )
    parser.add_argument(
        "--entrypoints",
        default=None,
        help="Path to entrypoints.json",
    )
    parser.add_argument(
        "--graph",
        default=None,
        help="Path to graph.json",
    )
    parser.add_argument(
        "--orphan-triage",
        dest="orphan_triage",
        default=None,
        help="Path to orphan-triage.json",
    )
    parser.add_argument(
        "--hubs",
        default=None,
        help="Path to hubs.json",
    )
    parser.add_argument(
        "--semantic",
        default=None,
        help="Path to semantic-signals.json",
    )
    parser.add_argument(
        "--delta",
        default=None,
        help="Path to delta.json",
    )
    parser.add_argument(
        "--readiness",
        default=None,
        help="Path to readiness.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write output JSON (default: stdout)",
    )
    args = parser.parse_args()

    warnings: list[str] = []

    # Load scan (required)
    scan = _load_artifact(args.scan, "scan", warnings)
    if scan is None:
        print("Error: scan.json is required and could not be loaded.", file=sys.stderr)
        return 1

    # Load optional artifacts
    classified_imports = _load_artifact(
        args.classified_imports, "classified_imports", warnings
    )
    entrypoints = _load_artifact(args.entrypoints, "entrypoints", warnings)
    graph = _load_artifact(args.graph, "graph", warnings)
    orphan_triage = _load_artifact(args.orphan_triage, "orphan_triage", warnings)
    hub_rankings = _load_artifact(args.hubs, "hub_rankings", warnings)
    semantic_signals = _load_artifact(args.semantic, "semantic_signals", warnings)
    delta = _load_artifact(args.delta, "delta", warnings)
    readiness = _load_artifact(args.readiness, "readiness", warnings)

    # Build report
    try:
        report = build_report_data(
            scan=scan,
            classified_imports=classified_imports,
            entrypoints=entrypoints,
            graph=graph,
            orphan_triage=orphan_triage,
            hub_rankings=hub_rankings,
            semantic_signals=semantic_signals,
            delta=delta,
            readiness=readiness,
        )
    except Exception as exc:
        print(f"Error: failed to build report data: {exc}", file=sys.stderr)
        return 1

    # Merge CLI-load warnings into the report
    report["warnings"].extend(warnings)

    # Output
    output_json = json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False)

    if args.output:
        try:
            out_path = Path(args.output)
            out_path.write_text(output_json + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"Error: could not write output file: {exc}", file=sys.stderr)
            return 1
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
