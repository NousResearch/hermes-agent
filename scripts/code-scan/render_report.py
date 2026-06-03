#!/usr/bin/env python3
"""render_report.py — Phase 4 D7b: Deterministic Markdown report renderer.

Renders a D7a report-data JSON dict into a compact Markdown artifact that
Hermes and subagents can read quickly before deeper code inspection.

Usage:
    python scripts/code-scan/render_report.py report-data.json \\
        --output UA_REPORT.md --max-bytes 500000

Stdlib only — no external dependencies.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_MAX_BYTES = 500_000
_TRUNCATION_MARKER = "\n\n---\n\n> [TRUNCATED — output exceeded --max-bytes cap.  " \
                     "Consider increasing --max-bytes.]\n"

# ── Section ordering (stable, deterministic) ───────────────────────────────

_SECTION_ORDER = [
    "scan",
    "classification",
    "entrypoints",
    "hub_rankings",
    "import_profile",  # derived from classification
    "orphan_triage",
    "semantic_signals",
    "domain_surfaces",
    "delta",
    "readiness",
    "reading_plan",
    "sources",
    "warnings",
    "totals",
    "graph_analysis",
]


# ── Markdown escaping helpers ──────────────────────────────────────────────

def _escape_for_md(text: str) -> str:
    """Escape text that will appear inline in Markdown tables/lists."""
    if not text:
        return ""
    # Escape pipe chars that could break tables
    return text.replace("|", "\\|")


def _safe_inline(text: str) -> str:
    """Return a safe inline string for Markdown rendering."""
    if not text:
        return "—"
    return _escape_for_md(str(text))


# ── Section renderers ──────────────────────────────────────────────────────

def _render_project_overview(scan: dict) -> str:
    """Render Project Overview from scan section."""
    lines = ["# Project Overview", ""]

    root = scan.get("project_root", "")
    scanned_at = scan.get("scanned_at", "")

    # Summary table
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Project root | {_safe_inline(root)} |")
    lines.append(f"| Scanned at | {_safe_inline(scanned_at)} |")
    total_files = scan.get("total_files", 0)
    total_lines = scan.get("total_lines", 0)
    lines.append(f"| Total files | {total_files} |")
    lines.append(f"| Total lines | {total_lines} |")
    lines.append("")

    # Languages
    languages = scan.get("languages", {})
    if languages:
        lines.append("### Languages")
        lines.append("")
        for lang, count in sorted(languages.items()):
            lines.append(f"- **{lang}**: {count}")
        lines.append("")

    # Categories
    categories = scan.get("categories", {})
    if categories:
        lines.append("### Categories")
        lines.append("")
        for cat, count in sorted(categories.items()):
            lines.append(f"- **{cat}**: {count}")
        lines.append("")

    # Frameworks
    frameworks = scan.get("frameworks", [])
    if frameworks:
        lines.append("### Frameworks")
        lines.append("")
        for fw in sorted(frameworks):
            lines.append(f"- {_safe_inline(fw)}")
        lines.append("")

    return "\n".join(lines)


def _render_deterministic_inventory(
    scan: dict, classification: Any, *, report_data: dict
) -> str:
    """Render Deterministic Inventory section."""
    lines = ["## Deterministic Inventory", ""]

    # Scan-derived inventory
    lines.append("### Scan-derived facts")
    lines.append("")
    total_files = scan.get("total_files", 0)
    total_lines = scan.get("total_lines", 0)
    lines.append(f"- **{total_files}** files scanned, **{total_lines}** total lines")
    lines.append("")

    # Classification summary
    if classification == "not_available":
        lines.append("### Import classification")
        lines.append("")
        lines.append("*Not available — classified-imports artifact not provided.*")
        lines.append("")
    elif isinstance(classification, dict):
        lines.append("### Import classification")
        lines.append("")
        totals = classification.get("totals", {})
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for cat in sorted(totals.keys()):
            lines.append(f"| {cat} | {totals[cat]} |")
        lines.append("")
        files_classified = classification.get("files_classified", 0)
        lines.append(f"Files classified: **{files_classified}**")
        lines.append("")

    # Totals
    totals_data = report_data.get("totals", {})
    if totals_data:
        lines.append("### Aggregated totals")
        lines.append("")
        for key in sorted(totals_data.keys()):
            val = totals_data[key]
            if isinstance(val, dict):
                lines.append(f"### {key.replace('_', ' ').title()}")
                lines.append("")
                for k in sorted(val.keys()):
                    lines.append(f"- **{k}**: {val[k]}")
                lines.append("")
            else:
                lines.append(f"- **{key}**: {val}")
        lines.append("")

    return "\n".join(lines)


def _render_entrypoints(entrypoints: Any) -> str:
    """Render Entrypoints / Where to Start section."""
    lines = ["## Entrypoints / Where to Start", ""]

    if entrypoints == "not_available":
        lines.append("*Not available — entrypoints artifact not provided.*")
        lines.append("")
        return "\n".join(lines)

    if not isinstance(entrypoints, dict):
        lines.append("*Not available.*")
        lines.append("")
        return "\n".join(lines)

    ep_list = entrypoints.get("entrypoints", [])
    if not ep_list:
        lines.append("*No entrypoints detected.*")
        lines.append("")
        return "\n".join(lines)

    lines.append("| File | Type | Confidence | Signals |")
    lines.append("|------|------|------------|---------|")
    for ep in sorted(
        ep_list, key=lambda e: (-e.get("confidence", 0), e.get("file", ""))
    ):
        file_name = _safe_inline(ep.get("file", ""))
        ep_type = _safe_inline(ep.get("type", ""))
        confidence = ep.get("confidence", 0)
        signals = _safe_inline("; ".join(ep.get("signals", [])))
        lines.append(
            f"| `{file_name}` | {ep_type} | {confidence} | {signals} |"
        )
    lines.append("")

    ep_totals = entrypoints.get("totals", {})
    found = ep_totals.get("entrypoints_found", 0)
    lines.append(f"**{found}** entrypoint(s) found (deterministic detection).")
    lines.append("")

    return "\n".join(lines)


def _render_hub_rankings(hub_rankings: Any) -> str:
    """Render Architectural Hubs section."""
    lines = ["## Architectural Hubs", ""]

    if hub_rankings == "not_available":
        lines.append("*Not available — hub-rankings artifact not provided.*")
        lines.append("")
        return "\n".join(lines)

    if not isinstance(hub_rankings, dict):
        lines.append("*Not available.*")
        lines.append("")
        return "\n".join(lines)

    hubs = hub_rankings.get("hubs", [])
    if not hubs:
        lines.append("*No hubs detected.*")
        lines.append("")
        return "\n".join(lines)

    lines.append("| File | Hub Score | In-Degree | Out-Degree | Confidence |")
    lines.append("|------|-----------|-----------|------------|------------|")
    for h in sorted(hubs, key=lambda x: (-x.get("hub_score", 0), x.get("file_path", ""))):
        fp = _safe_inline(h.get("file_path", ""))
        score = h.get("hub_score", 0)
        ind = h.get("in_degree", 0)
        outd = h.get("out_degree", 0)
        conf = h.get("confidence", "")
        lines.append(f"| `{fp}` | {score} | {ind} | {outd} | {conf} |")
    lines.append("")

    # Disclaimer
    disclaimer = hub_rankings.get("disclaimer", "")
    if disclaimer:
        lines.append(f"> {disclaimer}")
        lines.append("")

    totals = hub_rankings.get("totals", {})
    if totals:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key in sorted(totals.keys()):
            lines.append(f"| {key} | {totals[key]} |")
        lines.append("")

    return "\n".join(lines)


def _render_import_profile(classification: Any) -> str:
    """Render Import Profile section."""
    lines = ["## Import Profile", ""]

    if classification == "not_available":
        lines.append("*Not available — classified-imports artifact not provided.*")
        lines.append("")
        return "\n".join(lines)

    if not isinstance(classification, dict):
        lines.append("*Not available.*")
        lines.append("")
        return "\n".join(lines)

    totals = classification.get("totals", {})
    lines.append("| Import Category | Count |")
    lines.append("|-----------------|-------|")
    for cat in sorted(totals.keys()):
        lines.append(f"| {cat} | {totals[cat]} |")
    lines.append("")

    files_classified = classification.get("files_classified", 0)
    lines.append(f"Files with classified imports: **{files_classified}**")
    lines.append("")

    return "\n".join(lines)


def _render_orphan_triage(orphan_triage: Any) -> str:
    """Render Orphan Triage section."""
    lines = ["## Orphan Triage", ""]

    if orphan_triage == "not_available":
        lines.append("*Not available — orphan-triage artifact not provided.*")
        lines.append("")
        return "\n".join(lines)

    if not isinstance(orphan_triage, dict):
        lines.append("*Not available.*")
        lines.append("")
        return "\n".join(lines)

    categories = orphan_triage.get("categories", {})
    totals = orphan_triage.get("totals", {})

    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    for cat in sorted(categories.keys()):
        lines.append(f"| {cat} | {categories[cat]} |")
    lines.append("")

    total_orphans = totals.get("total_orphans", 0)
    lines.append(f"**{total_orphans}** total orphan files detected.")
    lines.append("")

    return "\n".join(lines)


def _render_semantic_signals(semantic: Any) -> str:
    """Render Semantic Signals section."""
    lines = ["## Semantic Signals", ""]

    if semantic == "not_available":
        lines.append("*Not available — semantic-signals artifact not provided.*")
        lines.append("")
        return "\n".join(lines)

    if not isinstance(semantic, dict):
        lines.append("*Not available.*")
        lines.append("")
        return "\n".join(lines)

    files_processed = semantic.get("files_processed", 0)
    symbols = semantic.get("symbols", 0)
    totals = semantic.get("totals", {})

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Files processed | {files_processed} |")
    lines.append(f"| Symbols extracted | {symbols} |")
    lines.append("")

    if totals:
        lines.append("| Total Metric | Value |")
        lines.append("|--------------|-------|")
        for key in sorted(totals.keys()):
            lines.append(f"| {key} | {totals[key]} |")
        lines.append("")

    return "\n".join(lines)


def _render_domain_surfaces(domain_surfaces: Any) -> str:
    """Render deterministic domain-surface inventory section."""
    lines = ["## Domain Surfaces", ""]
    if domain_surfaces == "not_available":
        lines.append("*Not available — domain-surfaces artifact not provided.*")
        lines.append("")
        return "\n".join(lines)
    if not isinstance(domain_surfaces, dict):
        lines.append("*Not available.*")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "Inventory-only deterministic path signals; these are not semantic, security, "
        "runtime, RLS, or deployment-validity claims."
    )
    lines.append("")
    summary = domain_surfaces.get("summary", {})
    lines.append(f"Total surfaces: **{summary.get('total_surfaces', 0)}**")
    lines.append("")

    surface_types = summary.get("surface_types", {}) or {}
    if surface_types:
        lines.append("### Surface counts")
        lines.append("")
        lines.append("| Surface | Count |")
        lines.append("|---------|-------|")
        for surface, count in sorted(surface_types.items()):
            lines.append(f"| {_safe_inline(surface)} | {count} |")
        lines.append("")

    surfaces = domain_surfaces.get("surfaces", []) or []
    if surfaces:
        lines.append("### Surface paths")
        lines.append("")
        lines.append("| Surface | Path | Claim | Status |")
        lines.append("|---------|------|-------|--------|")
        for item in sorted(surfaces, key=lambda x: (x.get("surface", ""), x.get("path", ""))):
            lines.append(
                f"| {_safe_inline(item.get('surface', ''))} | `{_safe_inline(item.get('path', ''))}` | "
                f"{_safe_inline(item.get('claim_type', ''))} | {_safe_inline(item.get('semantic_status', ''))} |"
            )
        lines.append("")
    return "\n".join(lines)


def _render_delta(delta: Any) -> str:
    """Render Delta Summary section."""
    lines = ["## Delta Summary", ""]

    if delta == "not_available":
        lines.append("*Not provided — no delta comparison available.*")
        lines.append("")
        return "\n".join(lines)

    if not isinstance(delta, dict):
        lines.append("*Not available.*")
        lines.append("")
        return "\n".join(lines)

    files_data = delta.get("files", {})

    added = files_data.get("added", [])
    if added:
        lines.append("### Added files")
        lines.append("")
        for f in sorted(added):
            lines.append(f"- `{f}`")
        lines.append("")
    else:
        lines.append("No added files detected.")
        lines.append("")

    removed = files_data.get("removed", [])
    if removed:
        lines.append("### Removed files")
        lines.append("")
        for f in sorted(removed):
            lines.append(f"- `{f}`")
        lines.append("")
    else:
        lines.append("No removed files detected.")
        lines.append("")

    common_count = files_data.get("common_count", 0)
    lines.append(f"Common files: **{common_count}**")
    lines.append("")

    # Languages delta
    languages = delta.get("languages", {})
    if languages:
        lines.append("### Language changes")
        lines.append("")
        lines.append("| Language | Change |")
        lines.append("|----------|--------|")
        for lang in sorted(languages.keys()):
            lines.append(f"| {lang} | {languages[lang]} |")
        lines.append("")

    # Categories delta
    categories = delta.get("categories", {})
    if categories:
        lines.append("### Category changes")
        lines.append("")
        lines.append("| Category | Change |")
        lines.append("|----------|--------|")
        for cat in sorted(categories.keys()):
            lines.append(f"| {cat} | {categories[cat]} |")
        lines.append("")

    # Frameworks delta
    frameworks = delta.get("frameworks", {})
    if frameworks:
        lines.append("### Framework changes")
        lines.append("")
        lines.append("| Framework | Change |")
        lines.append("|-----------|--------|")
        for fw in sorted(frameworks.keys()):
            lines.append(f"| {fw} | {frameworks[fw]} |")
        lines.append("")

    return "\n".join(lines)


def _render_readiness(readiness: Any) -> str:
    """Render Readiness section."""
    lines = ["## Readiness", ""]

    if readiness == "not_available":
        lines.append("*Not available — readiness artifact not provided.*")
        lines.append("")
        return "\n".join(lines)

    if not isinstance(readiness, dict):
        lines.append("*Not available.*")
        lines.append("")
        return "\n".join(lines)

    stacks = readiness.get("detected_stacks", [])
    if stacks:
        lines.append("### Detected stacks")
        lines.append("")
        for s in sorted(stacks):
            lines.append(f"- {s}")
        lines.append("")

    status = readiness.get("verification_status", "")
    if status:
        lines.append(f"Verification status: **{_safe_inline(status)}**")
        lines.append("")

    blockers = readiness.get("blockers", [])
    if blockers:
        lines.append("### Blockers")
        lines.append("")
        for b in sorted(blockers):
            lines.append(f"- {_safe_inline(b)}")
        lines.append("")

    return "\n".join(lines)


def _render_reading_plan(reading_plan: list) -> str:
    """Render Suggested Reading Path section."""
    lines = ["## Suggested Reading Path", ""]

    if not reading_plan:
        lines.append("*No reading candidates identified.*")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Priority | Type | File | Reason |")
    lines.append("|----------|------|------|--------|")
    for item in reading_plan:
        priority = _safe_inline(item.get("priority", ""))
        item_type = _safe_inline(item.get("type", ""))
        file_name = _safe_inline(item.get("file", ""))
        reason = _safe_inline(item.get("reason", ""))
        lines.append(
            f"| {priority} | {item_type} | `{file_name}` | {reason} |"
        )
    lines.append("")

    lines.append(f"**{len(reading_plan)}** reading candidate(s) listed (deterministic selection).")
    lines.append("")

    return "\n".join(lines)


def _render_sources(sources: dict) -> str:
    """Render sources tracking table."""
    lines = ["## Artifact Sources", ""]

    lines.append("| Artifact | Status |")
    lines.append("|----------|--------|")
    for key in sorted(sources.keys()):
        status = sources[key]
        display = "Loaded" if status == "loaded" else "Not provided"
        lines.append(f"| {key} | {display} |")
    lines.append("")

    return "\n".join(lines)


def _render_warnings(warnings: list) -> str:
    """Render warnings as a list."""
    lines = ["## Warnings", ""]

    if not warnings:
        lines.append("*No warnings.*")
        lines.append("")
        return "\n".join(lines)

    for w in warnings:
        lines.append(f"- {w}")
    lines.append("")

    return "\n".join(lines)


def _render_graph_analysis(graph: Any) -> str:
    """Render Graph/Validation section."""
    lines = ["## Graph / Validation", ""]

    if graph == "not_available":
        lines.append("*Not available — graph artifact not provided.*")
        lines.append("")
        return "\n".join(lines)

    if not isinstance(graph, dict):
        lines.append("*Not available.*")
        lines.append("")
        return "\n".join(lines)

    nodes_count = graph.get("nodes_count", 0)
    file_nodes_count = graph.get("file_nodes_count", 0)
    edges_count = graph.get("edges_count", 0)

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total nodes | {nodes_count} |")
    lines.append(f"| File nodes | {file_nodes_count} |")
    lines.append(f"| Edges | {edges_count} |")
    lines.append("")

    analytics = graph.get("analytics", {})
    if analytics:
        if "top_in_degree" in analytics:
            lines.append("### Top in-degree nodes")
            lines.append("")
            lines.append("| Node | In-Degree |")
            lines.append("|------|-----------|")
            for n in analytics["top_in_degree"]:
                nid = _safe_inline(n.get("node_id", n.get("filePath", "")))
                deg = n.get("in_degree", "")
                lines.append(f"| `{nid}` | {deg} |")
            lines.append("")

        if "top_out_degree" in analytics:
            lines.append("### Top out-degree nodes")
            lines.append("")
            lines.append("| Node | Out-Degree |")
            lines.append("|------|------------|")
            for n in analytics["top_out_degree"]:
                nid = _safe_inline(n.get("node_id", n.get("filePath", "")))
                deg = n.get("out_degree", "")
                lines.append(f"| `{nid}` | {deg} |")
            lines.append("")

    graph_warnings = graph.get("warnings", [])
    if graph_warnings:
        lines.append("### Graph warnings")
        lines.append("")
        for w in graph_warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


# ── Deterministic hints disclaimer ─────────────────────────────────────────

_DISCLAIMER = (
    "> **Note:** All results in this report are *deterministic hints "
    "derived from static analysis*, not proof or conclusive evidence. "
    "Hub scores, entrypoint confidence, and triage categories are "
    "heuristic indicators intended to guide further manual inspection. "
    "Caveats and missing artifacts are noted where applicable.\n"
)


# ── Main renderer ──────────────────────────────────────────────────────────

def render_report_data(report_data: dict, *, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Render a D7a report-data dict to Markdown.

    Args:
        report_data: The report-data JSON dict from D7a build_report_data().
        max_bytes: Maximum size of the rendered output in bytes.

    Returns:
        Deterministic Markdown string.
    """
    sections = report_data.get("sections", {})
    reading_plan = report_data.get("reading_plan", [])
    warnings = report_data.get("warnings", [])
    sources = report_data.get("sources", {})
    totals = report_data.get("totals", {})

    scan = sections.get("scan", {})
    classification = sections.get("classification", "not_available")
    entrypoints = sections.get("entrypoints", "not_available")
    hub_rankings = sections.get("hub_rankings", "not_available")
    orphan_triage = sections.get("orphan_triage", "not_available")
    semantic = sections.get("semantic_signals", "not_available")
    domain_surfaces = sections.get("domain_surfaces", "not_available")
    delta = sections.get("delta", "not_available")
    readiness = sections.get("readiness", "not_available")
    graph = sections.get("graph_analysis", "not_available")

    parts: list[str] = []

    # 1. Project Overview (always present, needs scan)
    if isinstance(scan, dict):
        parts.append(_render_project_overview(scan))

    # 2. Deterministic Inventory
    if isinstance(scan, dict):
        parts.append(
            _render_deterministic_inventory(scan, classification, report_data=report_data)
        )

    # 3. Entrypoints / Where to Start
    parts.append(_render_entrypoints(entrypoints))

    # 4. Architectural Hubs
    parts.append(_render_hub_rankings(hub_rankings))

    # 5. Import Profile
    parts.append(_render_import_profile(classification))

    # 6. Orphan Triage
    parts.append(_render_orphan_triage(orphan_triage))

    # 7. Semantic Signals
    parts.append(_render_semantic_signals(semantic))

    # 7b. Domain Surfaces
    parts.append(_render_domain_surfaces(domain_surfaces))

    # 8. Delta Summary
    parts.append(_render_delta(delta))

    # 9. Readiness
    parts.append(_render_readiness(readiness))

    # 9b. Graph / Validation
    parts.append(_render_graph_analysis(graph))

    # 10. Suggested Reading Path
    parts.append(_render_reading_plan(reading_plan))

    # 11. Sources
    if sources:
        parts.append(_render_sources(sources))

    # 12. Warnings
    parts.append(_render_warnings(warnings))

    # 13. Caveats / Validation heading
    parts.append("## Validation / Caveats")
    parts.append("")
    parts.append(_DISCLAIMER.strip())
    parts.append("")

    # Assemble
    md = "\n".join(parts)

    # Enforce max-bytes cap
    encoded = md.encode("utf-8")
    if len(encoded) > max_bytes:
        # Truncate at max_bytes boundary, ensuring UTF-8 validity
        truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
        # Strip any incomplete trailing lines
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]
        md = truncated + _TRUNCATION_MARKER

    return md


# ── CLI entry point ────────────────────────────────────────────────────────

def main() -> int:
    """CLI entry point. Reads report-data JSON, emits Markdown."""
    parser = argparse.ArgumentParser(
        description="Render deterministic Markdown report from D7a report-data JSON."
    )
    parser.add_argument(
        "report_data",
        help="Path to report-data.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write output Markdown (default: stdout)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"Max output size in bytes (default: {DEFAULT_MAX_BYTES})",
    )
    args = parser.parse_args()

    # Load report-data
    report_path = Path(args.report_data)
    if not report_path.is_file():
        print(f"Error: file not found: {args.report_data}", file=sys.stderr)
        return 1

    try:
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report_data, dict):
            print(
                f"Error: expected JSON object, got {type(report_data).__name__}",
                file=sys.stderr,
            )
            return 1
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error: invalid or unreadable JSON ({exc})", file=sys.stderr)
        return 1

    # Validate minimum structure
    if "sections" not in report_data:
        print("Error: report-data missing 'sections' key", file=sys.stderr)
        return 1

    # Render
    try:
        md = render_report_data(report_data, max_bytes=args.max_bytes)
    except Exception as exc:
        print(f"Error: failed to render report: {exc}", file=sys.stderr)
        return 1

    # Output
    if args.output:
        try:
            out_path = Path(args.output)
            out_path.write_text(md, encoding="utf-8")
        except OSError as exc:
            print(f"Error: could not write output file: {exc}", file=sys.stderr)
            return 1
    else:
        sys.stdout.write(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
