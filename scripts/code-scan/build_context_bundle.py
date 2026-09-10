#!/usr/bin/env python3
"""build_context_bundle.py — UA-004 Subagent Context Envelope.

Produces a machine-readable ``subagent-context.json`` and a markdown handoff
block from a UA run-bundle directory.

The envelope aggregates artifacts from the canonical run bundle (UA-001) and
optionally enriches with severity analysis (UA-002) and graph analytics
(UA-003).  Missing optional artifacts are tracked in ``artifacts_missing``;
no fabrication occurs.

Usage:
    python build_context_bundle.py --bundle-dir /path/to/bundle --out /path/to/subagent-context.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

HANDOFF_VERSION = "1.0.0"


# ── Artifact definitions ─────────────────────────────────────────────────────

CORE_ARTIFACTS = [
    "scan.json",
    "manifest.json",
    "summary.json",
    "validation.json",
]

OPTIONAL_ARTIFACTS = [
    "graph_analytics.json",  # UA-003
    "severity_analysis.json",  # UA-002
    "graph.json",
    "imports.json",
    "REPORT.md",
]

ALL_ARTIFACTS = CORE_ARTIFACTS + OPTIONAL_ARTIFACTS


# ── Helper: load JSON artifact safely ─────────────────────────────────────────

def _load_json(bundle_dir: str, filename: str) -> Optional[dict]:
    """Load a JSON artifact from the bundle, returning None on failure."""
    path = os.path.join(bundle_dir, filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_text(bundle_dir: str, filename: str) -> Optional[str]:
    """Load a text artifact from the bundle, returning None on failure."""
    path = os.path.join(bundle_dir, filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


# ── Build the context envelope ────────────────────────────────────────────────

def build_context_envelope(
    bundle_dir: str,
    *,
    out_path: Optional[str] = None,
) -> dict:
    """Build a subagent context envelope from a run-bundle directory.

    Args:
        bundle_dir: Path to the canonical run-bundle directory.
        out_path: Optional path to write the resulting JSON.

    Returns:
        The envelope dict.
    """
    bundle_dir = os.path.realpath(bundle_dir)
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    # --- Determine which artifacts are present ---
    artifacts_included = []
    artifacts_missing = []

    for artifact in ALL_ARTIFACTS:
        path = os.path.join(bundle_dir, artifact)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            artifacts_included.append({
                "artifact": artifact,
                "size_bytes": size,
            })
        else:
            artifacts_missing.append({
                "artifact": artifact,
                "reason": "file not found in bundle",
            })

    # Load core data
    manifest = _load_json(bundle_dir, "manifest.json")
    scan = _load_json(bundle_dir, "scan.json")
    summary = _load_json(bundle_dir, "summary.json")
    validation = _load_json(bundle_dir, "validation.json")
    report_md = _load_text(bundle_dir, "REPORT.md")

    # Load optional data
    severity = _load_json(bundle_dir, "severity_analysis.json")
    analytics = _load_json(bundle_dir, "graph_analytics.json")
    graph = _load_json(bundle_dir, "graph.json")

    # --- scan_run_id ---
    scan_run_id = (
        manifest.get("run_id", "unknown")
        if manifest
        else "unknown"
    )

    # --- target ---
    target = (
        manifest.get("target_path", bundle_dir)
        if manifest
        else bundle_dir
    )

    # --- validation section ---
    validation_section = _build_validation_section(
        validation, severity, analytics,
    )

    # --- confidence ---
    confidence = _build_confidence(
        scan, manifest, severity, analytics, validation,
    )

    # --- recommended_files ---
    recommended_files = _build_recommended_files(
        scan, graph, severity, analytics,
    )

    # --- reading_budget ---
    reading_budget = _build_reading_budget(
        recommended_files, scan, graph,
    )

    # --- suggested_questions ---
    suggested_questions = _build_suggested_questions(
        validation, severity, analytics, scan, bundle_dir,
    )

    # --- truncation_warnings ---
    truncation_warnings = _build_truncation_warnings(
        scan, artifacts_missing, graph,
    )

    envelope = {
        "handoff_version": HANDOFF_VERSION,
        "scan_run_id": scan_run_id,
        "target": target,
        "artifacts_included": artifacts_included,
        "artifacts_missing": artifacts_missing,
        "validation": validation_section,
        "confidence": confidence,
        "recommended_files": recommended_files,
        "reading_budget": reading_budget,
        "truncation_warnings": truncation_warnings,
        "suggested_questions": suggested_questions,
    }

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2)
            f.write("\n")

    return envelope


# ── Validation section ────────────────────────────────────────────────────────

def _build_validation_section(
    validation: Optional[dict],
    severity: Optional[dict],
    analytics: Optional[dict],
) -> dict:
    """Build the validation sub-object."""
    section = {
        "verdict": "unknown",
        "issue_count": 0,
        "warning_count": 0,
    }

    if validation:
        issues = validation.get("issues", [])
        warnings = validation.get("warnings", [])
        section["issue_count"] = len(issues)
        section["warning_count"] = len(warnings)

        if len(issues) == 0 and len(warnings) == 0:
            section["verdict"] = "pass"
        elif len(issues) > 0:
            section["verdict"] = "issues_found"
        else:
            section["verdict"] = "warnings_only"
    else:
        section["verdict"] = "no_validation_data"

    # Severity summary (UA-002) — only if present
    if severity:
        severity_summary = {}
        for level in ("critical", "high", "medium", "low", "info"):
            count = severity.get(level, 0)
            if count > 0:
                severity_summary[level] = count
        section["severity_summary"] = severity_summary if severity_summary else None

    return section


# ── Confidence ─────────────────────────────────────────────────────────────────

def _build_confidence(
    scan: Optional[dict],
    manifest: Optional[dict],
    severity: Optional[dict],
    analytics: Optional[dict],
    validation: Optional[dict],
) -> dict:
    """Build confidence dict with labels reflecting data provenance."""
    confidence: dict[str, str] = {}

    # Raw scan counts — deterministic, directly measured
    if scan:
        confidence["total_files"] = "high"
        confidence["total_lines"] = "high"
        confidence["languages"] = "high"

    # Manifest metadata — deterministic from git/sha
    if manifest:
        confidence["scan_run_id"] = "high"
        confidence["target"] = "high"

    # Validation counts — deterministic from validation output
    if validation:
        confidence["validation_counts"] = "high"
    else:
        confidence["validation_counts"] = "low"

    # Severity analysis — present but may be heuristic
    if severity:
        confidence["severity"] = "medium"

    # Graph analytics — derived from graph data
    if analytics:
        confidence["graph_analytics"] = "medium"

    # Hub node hints — inferred from graph structure
    if analytics and analytics.get("hub_nodes"):
        confidence["hub_hints"] = "medium"

    # Inferred project description — not available from scans alone
    confidence["project_description"] = "low"

    return confidence


# ── Recommended files ─────────────────────────────────────────────────────────

def _build_recommended_files(
    scan: Optional[dict],
    graph: Optional[dict],
    severity: Optional[dict],
    analytics: Optional[dict],
) -> list[dict]:
    """Build a list of files recommended for subagent review."""
    recommended: list[dict] = []
    seen = set()

    # Files from scan data, sorted by lines (larger files first)
    if scan:
        files = scan.get("files", [])
        sorted_files = sorted(files, key=lambda f: f.get("lines", 0), reverse=True)
        for f in sorted_files:
            path = f.get("path", "")
            if path and path not in seen:
                recommended.append({
                    "path": path,
                    "reason": "listed in scan",
                    "lines": f.get("lines", 0),
                    "language": f.get("language", ""),
                })
                seen.add(path)

    # Hub nodes from graph analytics (UA-003)
    if analytics:
        for hub in analytics.get("hub_nodes", []):
            node_id = hub.get("node_id", "")
            # Extract path from node_id like "file:src/utils.py"
            if node_id.startswith("file:"):
                path = node_id[len("file:"):]
                if path not in seen:
                    recommended.append({
                        "path": path,
                        "reason": f"hub node (degree={hub.get('degree', '?')})",
                    })
                    seen.add(path)

    # Files mentioned in severity analysis (UA-002)
    if severity:
        for item in severity.get("items", []):
            path = item.get("file", "")
            if path and path not in seen:
                recommended.append({
                    "path": path,
                    "reason": f"severity: {item.get('severity', 'unknown')}",
                })
                seen.add(path)

    return recommended


# ── Reading budget ────────────────────────────────────────────────────────────

def _build_reading_budget(
    recommended_files: list[dict],
    scan: Optional[dict],
    graph: Optional[dict],
) -> list[dict]:
    """Build a concrete reading budget for subagents."""
    budget: list[dict] = []

    # Top files by size from recommended list
    for entry in recommended_files[:10]:
        budget.append({
            "path": entry["path"],
            "lines": entry.get("lines", "unknown"),
            "priority": "read",
        })

    return budget


# ── Truncation warnings ───────────────────────────────────────────────────────

def _build_truncation_warnings(
    scan: Optional[dict],
    artifacts_missing: list[dict],
    graph: Optional[dict],
) -> list[str]:
    """Build a list of truncation/completeness warnings."""
    warnings: list[str] = []

    missing_names = [a["artifact"] for a in artifacts_missing]
    if "validation.json" in missing_names:
        warnings.append("No validation data available — verdict may be incomplete")
    if "graph_analytics.json" in missing_names:
        warnings.append("Graph analytics (UA-003) not available — hub hints may be missing")
    if "severity_analysis.json" in missing_names:
        warnings.append("Severity analysis (UA-002) not available — severity_summary omitted")

    if scan:
        total_files = scan.get("total_files", 0)
        total_lines = scan.get("total_lines", 0)
        if total_files > 500:
            warnings.append(
                f"Large project: {total_files} files scanned — subagent may need pagination"
            )
        if total_lines > 100_000:
            warnings.append(
                f"Large codebase: {total_lines} total lines — context window may be exceeded"
            )

    return warnings


# ── Suggested questions ────────────────────────────────────────────────────────

def _build_suggested_questions(
    validation: Optional[dict],
    severity: Optional[dict],
    analytics: Optional[dict],
    scan: Optional[dict],
    bundle_dir: str,
) -> dict[str, list[str]]:
    """Build role-specific suggested questions for subagents."""
    questions: dict[str, list[str]] = {
        "researcher": [],
        "reviewer": [],
        "coder": [],
    }

    # Researcher questions — focused on understanding and exploring
    if scan:
        langs = scan.get("languages", {})
        lang_list = ", ".join(langs.keys())
        questions["researcher"].append(
            f"Project uses: {lang_list} — What is the overall architecture?"
        )
        questions["researcher"].append(
            f"Total files: {scan.get('total_files', '?')}, "
            f"total lines: {scan.get('total_lines', '?')} — "
            "What are the main entry points?"
        )
    else:
        questions["researcher"].append(
            "No scan data available — What can be determined from the raw bundle?"
        )

    if analytics and analytics.get("hub_nodes"):
        hub_names = ", ".join(
            h.get("label", h.get("node_id", "?"))
            for h in analytics["hub_nodes"][:3]
        )
        questions["researcher"].append(
            f"Hub nodes identified: {hub_names} — How do these modules interact?"
        )

    # Reviewer questions — focused on validation and risk
    if validation:
        issues = validation.get("issues", [])
        warnings = validation.get("warnings", [])
        if issues:
            questions["reviewer"].append(
                f"{len(issues)} validation issue(s) found — "
                "Review the following: " + "; ".join(issues[:3])
            )
        if warnings:
            questions["reviewer"].append(
                f"{len(warnings)} warning(s) found — "
                "Assess severity: " + "; ".join(warnings[:3])
            )
        if not issues and not warnings:
            questions["reviewer"].append(
                "Validation passed — What edge cases might the validator have missed?"
            )
    else:
        questions["reviewer"].append(
            "No validation data — Manual review of code quality is recommended"
        )

    if severity:
        critical = severity.get("critical", 0)
        high = severity.get("high", 0)
        if critical > 0:
            questions["reviewer"].append(
                f"{critical} critical severity item(s) — Immediate attention needed"
            )
        if high > 0:
            questions["reviewer"].append(
                f"{high} high severity item(s) — Review priority files"
            )

    # Coder questions — focused on actionable change suggestions
    questions["coder"].append(
        "What are the highest-impact refactorings or improvements?"
    )

    if scan:
        test_files = [
            f for f in scan.get("files", [])
            if "test" in f.get("path", "").lower()
        ]
        if not test_files:
            questions["coder"].append(
                "No test files detected in scan — Should test coverage be added?"
            )

    if severity:
        sev_items = severity.get("items", [])
        if sev_items:
            top_file = sev_items[0].get("file", "unknown")
            questions["coder"].append(
                f"File '{top_file}' has the highest severity issue — "
                "What specific code changes would address it?"
            )

    return questions


# ── Markdown handoff rendering ─────────────────────────────────────────────────

def render_markdown_handoff(envelope: dict) -> str:
    """Render the envelope as a markdown handoff block.

    Separates deterministic facts from interpretation prompts.
    """
    lines: list[str] = []

    lines.append("# Subagent Context Handoff")
    lines.append("")
    lines.append(f"**Handoff Version**: {envelope['handoff_version']}")
    lines.append(f"**Scan Run ID**: {envelope['scan_run_id']}")
    lines.append(f"**Target**: `{envelope['target']}`")
    lines.append("")

    # ── Deterministic Facts ─────────────────────────────────────────────
    lines.append("## Deterministic Facts")
    lines.append("")

    lines.append("### Artifacts")
    lines.append("")
    lines.append("| Artifact | Status | Size |")
    lines.append("|----------|--------|------|")
    for a in envelope.get("artifacts_included", []):
        lines.append(
            f"| {a['artifact']} | **included** | {a['size_bytes']} bytes |"
        )
    for a in envelope.get("artifacts_missing", []):
        lines.append(
            f"| {a['artifact']} | **missing** | — |"
        )
    lines.append("")

    # Validation facts
    v = envelope.get("validation", {})
    lines.append("### Validation")
    lines.append("")
    lines.append(f"- **Verdict**: {v.get('verdict', 'unknown')}")
    lines.append(f"- **Issues**: {v.get('issue_count', 0)}")
    lines.append(f"- **Warnings**: {v.get('warning_count', 0)}")
    if v.get("severity_summary"):
        lines.append(f"- **Severity Summary**: {json.dumps(v['severity_summary'])}")
    lines.append("")

    # Recommended files
    rec = envelope.get("recommended_files", [])
    if rec:
        lines.append("### Recommended Files (Top 10)")
        lines.append("")
        lines.append("| Path | Reason | Lines |")
        lines.append("|------|--------|-------|")
        for entry in rec[:10]:
            lang = entry.get("language", "")
            lines_str = str(entry.get("lines", "?"))
            lang_suffix = f" ({lang})" if lang else ""
            lines.append(
                f"| `{entry['path']}` | {entry.get('reason', '')} | {lines_str}{lang_suffix} |"
            )
        lines.append("")

    # Confidence
    conf = envelope.get("confidence", {})
    if conf:
        lines.append("### Data Confidence")
        lines.append("")
        for key, label in sorted(conf.items()):
            lines.append(f"- **{key}**: {label}")
        lines.append("")

    # Truncation warnings
    tw = envelope.get("truncation_warnings", [])
    if tw:
        lines.append("### Truncation Warnings")
        lines.append("")
        for w in tw:
            lines.append(f"- ⚠ {w}")
        lines.append("")

    # ── Interpretation Prompts (for LLM subagents) ──────────────────────
    lines.append("## Interpretation Prompts")
    lines.append("")
    lines.append("> These questions are for subagent interpretation. "
                 "They must be answered using the facts above, not fabricated.")
    lines.append("")

    sq = envelope.get("suggested_questions", {})
    for role in ("researcher", "reviewer", "coder"):
        questions = sq.get(role, [])
        if questions:
            lines.append(f"### {role.capitalize()} Questions")
            lines.append("")
            for q in questions:
                lines.append(f"- {q}")
            lines.append("")

    # Reading budget
    budget = envelope.get("reading_budget", [])
    if budget:
        lines.append("## Reading Budget")
        lines.append("")
        for entry in budget:
            lines.append(
                f"- `{entry['path']}` ({entry.get('lines', '?')} lines) — priority: {entry.get('priority', 'read')}"
            )
        lines.append("")

    lines.append("---")
    lines.append("*Generated by build_context_bundle.py (UA-004)*")
    return "\n".join(lines) + "\n"


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build a subagent context envelope from a UA run-bundle.",
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="Path to the canonical run-bundle directory",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the subagent-context.json output",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        default=False,
        help="Suppress markdown handoff output to stdout",
    )
    args = parser.parse_args()

    try:
        envelope = build_context_envelope(args.bundle_dir, out_path=args.out)

        # Always print the markdown handoff to stdout (unless suppressed)
        if not args.no_markdown:
            md = render_markdown_handoff(envelope)
            print(md, end="")

        print(f"Context envelope written to: {args.out}", file=sys.stderr)
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
