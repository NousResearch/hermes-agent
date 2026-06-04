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

from recommended_files import build_recommended_files

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
    "domain-surfaces.json",  # UA-P5-005
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


def _manifest_artifact_path(bundle_dir: str, raw_path: Any) -> Optional[str]:
    """Return an absolute artifact path from manifest metadata, if usable."""
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(bundle_dir) / path
    return str(path)


def _manifest_missing_items(manifest: dict) -> list[dict]:
    """Normalize manifest-level missing artifact metadata."""
    missing = manifest.get("artifacts_missing", [])
    if not isinstance(missing, list):
        return []

    normalized: list[dict] = []
    seen: set[str] = set()
    for item in missing:
        if isinstance(item, str):
            artifact = item
            reason = "listed as missing in manifest"
        elif isinstance(item, dict):
            artifact = str(
                item.get("artifact") or item.get("name") or item.get("path") or ""
            )
            reason = str(item.get("reason") or "listed as missing in manifest")
        else:
            continue
        if artifact and artifact not in seen:
            normalized.append({"artifact": artifact, "reason": reason})
            seen.add(artifact)
    return normalized


def _build_artifact_claims(bundle_dir: str, manifest: Optional[dict]) -> tuple[list[dict], list[dict]]:
    """Build included/missing artifact claims from manifest or fallback definitions."""
    artifacts_included: list[dict] = []
    artifacts_missing: list[dict] = []
    seen: set[str] = set()

    if manifest and isinstance(manifest.get("artifact_paths"), dict):
        artifact_paths = manifest.get("artifact_paths", {})
        for artifact in sorted(artifact_paths):
            artifact_path = _manifest_artifact_path(bundle_dir, artifact_paths.get(artifact))
            seen.add(artifact)
            if artifact_path and os.path.isfile(artifact_path):
                artifacts_included.append({
                    "artifact": artifact,
                    "size_bytes": os.path.getsize(artifact_path),
                })
            else:
                artifacts_missing.append({
                    "artifact": artifact,
                    "reason": "file not found at manifest artifact path",
                })

        for item in _manifest_missing_items(manifest):
            artifact = item["artifact"]
            if artifact not in seen:
                artifacts_missing.append(item)
                seen.add(artifact)

        return artifacts_included, artifacts_missing

    for artifact in ALL_ARTIFACTS:
        path = os.path.join(bundle_dir, artifact)
        if os.path.isfile(path):
            artifacts_included.append({
                "artifact": artifact,
                "size_bytes": os.path.getsize(path),
            })
        else:
            artifacts_missing.append({
                "artifact": artifact,
                "reason": "file not found in bundle",
            })
    return artifacts_included, artifacts_missing


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

    # Load core data
    manifest = _load_json(bundle_dir, "manifest.json")
    artifacts_included, artifacts_missing = _build_artifact_claims(bundle_dir, manifest)
    scan = _load_json(bundle_dir, "scan.json")
    summary = _load_json(bundle_dir, "summary.json")
    validation = _load_json(bundle_dir, "validation.json")
    report_md = _load_text(bundle_dir, "REPORT.md")

    # Load optional data
    severity = _load_json(bundle_dir, "severity_analysis.json")
    analytics = _load_json(bundle_dir, "graph_analytics.json")
    domain_surfaces = _load_json(bundle_dir, "domain-surfaces.json")
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

    # --- critic_packs (UA-P5-008) ---
    critic_packs = _build_critic_packs(
        manifest=manifest,
        scan=scan,
        validation=validation_section,
        severity=severity,
        analytics=analytics,
        domain_surfaces=domain_surfaces,
        artifacts_missing=artifacts_missing,
        recommended_files=recommended_files,
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
        "critic_packs": critic_packs,
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
    """Build a deterministic source/security-aware reading plan."""
    return build_recommended_files(scan, graph, severity, analytics)


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


# ── Critic packs (UA-P5-008) ─────────────────────────────────────────────────

def _bounded_list(values: list[Any], limit: int) -> list[Any]:
    """Return a deterministic bounded copy of a list."""
    return list(values[:limit])


def _artifact_missing_reason(artifacts_missing: list[dict], artifact: str) -> str:
    """Return the recorded reason for a missing artifact, if any."""
    for item in artifacts_missing:
        if item.get("artifact") == artifact:
            reason = str(item.get("reason") or "file not found in bundle")
            return f"artifact not present in bundle: {reason}"
    return "artifact not present in bundle"


def _build_trust_anchor_summary(manifest: Optional[dict]) -> str:
    """Summarize deterministic provenance anchors without interpreting runtime state."""
    if not manifest:
        return "Deterministic facts are limited: manifest.json is absent or unreadable."

    run_id = manifest.get("run_id", "unknown")
    mode = manifest.get("mode", "unknown")
    clean = manifest.get("cleanliness", {})
    if isinstance(clean, dict):
        clean_bits = [f"{key}={clean.get(key)}" for key in sorted(clean)]
        clean_text = "; ".join(clean_bits) if clean_bits else "cleanliness fields not present"
    else:
        clean_text = "cleanliness fields not present"
    return (
        f"Deterministic manifest facts only: run_id={run_id}; mode={mode}; "
        f"cleanliness={clean_text}."
    )


def _build_top_deterministic_facts(
    scan: Optional[dict],
    validation: dict,
    recommended_files: list[dict],
) -> list[str]:
    """Build a compact list of measured facts from artifacts."""
    facts: list[str] = []
    if scan:
        facts.append(f"total_files={scan.get('total_files', 0)}")
        facts.append(f"total_lines={scan.get('total_lines', 0)}")
        languages = scan.get("languages", {})
        if isinstance(languages, dict) and languages:
            lang_text = ", ".join(f"{k}:{languages[k]}" for k in sorted(languages)[:8])
            facts.append(f"languages={lang_text}")
    facts.append(
        "validation="
        f"{validation.get('verdict', 'unknown')}; "
        f"issues={validation.get('issue_count', 0)}; "
        f"warnings={validation.get('warning_count', 0)}"
    )
    if recommended_files:
        top_paths = [str(item.get("path", "")) for item in recommended_files[:5]]
        facts.append("top_recommended_files=" + ", ".join(p for p in top_paths if p))
    return _bounded_list(facts, 8)


def _build_warning_orphan_triage_summary(
    validation: dict,
    severity: Optional[dict],
) -> list[str]:
    """Summarize warnings/orphans/severity counts without assigning semantic cause."""
    triage = [
        f"validation_issues={validation.get('issue_count', 0)}",
        f"validation_warnings={validation.get('warning_count', 0)}",
        f"validation_verdict={validation.get('verdict', 'unknown')}",
    ]
    if severity:
        counts = []
        for level in ("critical", "high", "medium", "low", "info"):
            value = severity.get(level, 0)
            if value:
                counts.append(f"{level}={value}")
        if counts:
            triage.append("severity_counts=" + ", ".join(counts))
    return _bounded_list(triage, 8)


def _build_domain_surface_inventory_summary(
    domain_surfaces: Optional[dict],
    artifacts_missing: list[dict],
) -> dict:
    """Summarize domain-surfaces.json, or record an explicit missing reason."""
    if not domain_surfaces:
        return {
            "available": False,
            "reason": _artifact_missing_reason(artifacts_missing, "domain-surfaces.json"),
        }

    summary = domain_surfaces.get("summary", {}) if isinstance(domain_surfaces, dict) else {}
    surfaces = domain_surfaces.get("surfaces", []) if isinstance(domain_surfaces, dict) else []
    surface_types = summary.get("surface_types", {}) if isinstance(summary, dict) else {}
    top_surfaces = []
    if isinstance(surfaces, list):
        for item in sorted(
            surfaces,
            key=lambda x: (str(x.get("surface", "")), str(x.get("path", ""))),
        )[:8]:
            top_surfaces.append({
                "surface": item.get("surface", ""),
                "path": item.get("path", ""),
                "claim_type": item.get("claim_type", "deterministic_inventory"),
            })
    return {
        "available": True,
        "total_surfaces": summary.get("total_surfaces", len(surfaces) if isinstance(surfaces, list) else 0),
        "surface_types": dict(sorted(surface_types.items())) if isinstance(surface_types, dict) else {},
        "top_surfaces": top_surfaces,
        "claim_type": domain_surfaces.get("claim_type", "deterministic_inventory"),
        "semantic_status": domain_surfaces.get("semantic_status", "not_validated"),
    }


def _outside_ua_scope_boundaries() -> list[str]:
    """Return the fixed boundary contract for critic packs."""
    return [
        "Hermes owns final assessment; reviewer/researcher/coder packs are targeted critic prompts only.",
        "Deterministic facts are separate from interpretation; do not convert heuristic hints into proven findings.",
        "UA does not prove security, deployment readiness, RLS correctness, or runtime correctness unless those gates actually ran.",
        "No LLM summaries are embedded in this pack; all values come from bundle artifacts or explicit missing-artifact reasons.",
    ]


def _build_critic_pack(
    *,
    mission: str,
    trust_anchor_summary: str,
    top_deterministic_facts: list[str],
    warning_orphan_triage_summary: list[str],
    domain_surface_inventory_summary: dict,
    outside_ua_scope_boundaries: list[str],
    focus_questions: list[str],
) -> dict:
    """Build a single bounded role-specific critic pack."""
    return {
        "mission": mission,
        "trust_anchor_summary": trust_anchor_summary,
        "top_deterministic_facts": _bounded_list(top_deterministic_facts, 8),
        "warning_orphan_triage_summary": _bounded_list(warning_orphan_triage_summary, 8),
        "domain_surface_inventory_summary": domain_surface_inventory_summary,
        "outside_ua_scope_boundaries": outside_ua_scope_boundaries,
        "focus_questions": _bounded_list(focus_questions, 5),
    }


def _build_critic_packs(
    *,
    manifest: Optional[dict],
    scan: Optional[dict],
    validation: dict,
    severity: Optional[dict],
    analytics: Optional[dict],
    domain_surfaces: Optional[dict],
    artifacts_missing: list[dict],
    recommended_files: list[dict],
) -> dict:
    """Build deterministic role packs for targeted critics, not primary authorship."""
    trust_anchor_summary = _build_trust_anchor_summary(manifest)
    top_facts = _build_top_deterministic_facts(scan, validation, recommended_files)
    triage = _build_warning_orphan_triage_summary(validation, severity)
    domain_summary = _build_domain_surface_inventory_summary(domain_surfaces, artifacts_missing)
    boundaries = _outside_ua_scope_boundaries()

    hub_count = 0
    if analytics and isinstance(analytics.get("hub_nodes"), list):
        hub_count = len(analytics.get("hub_nodes", []))
    shared_context = top_facts + [f"hub_hints={hub_count}"]

    return {
        "reviewer_critic": _build_critic_pack(
            mission="Challenge validation blind spots, overclaims, and risk boundaries; do not author the final assessment.",
            trust_anchor_summary=trust_anchor_summary,
            top_deterministic_facts=shared_context,
            warning_orphan_triage_summary=triage,
            domain_surface_inventory_summary=domain_summary,
            outside_ua_scope_boundaries=boundaries,
            focus_questions=[
                "Which deterministic facts could be over-interpreted?",
                "Do warnings/orphans indicate areas needing manual inspection?",
                "Are outside-UA-scope boundaries preserved?",
            ],
        ),
        "researcher_scout": _build_critic_pack(
            mission="Scout architecture/domain questions from deterministic artifacts only; do not replace Hermes final synthesis.",
            trust_anchor_summary=trust_anchor_summary,
            top_deterministic_facts=shared_context,
            warning_orphan_triage_summary=triage,
            domain_surface_inventory_summary=domain_summary,
            outside_ua_scope_boundaries=boundaries,
            focus_questions=[
                "Which surfaces or languages deserve first-pass orientation?",
                "Which files should be read before forming architecture hypotheses?",
                "What facts are absent and must not be inferred?",
            ],
        ),
        "coder_preflight": _build_critic_pack(
            mission="Identify preflight implementation risks and candidate files from UA artifacts; do not change code without Hermes/user direction.",
            trust_anchor_summary=trust_anchor_summary,
            top_deterministic_facts=shared_context,
            warning_orphan_triage_summary=triage,
            domain_surface_inventory_summary=domain_summary,
            outside_ua_scope_boundaries=boundaries,
            focus_questions=[
                "Which recommended files are likely safest to inspect first?",
                "Which validation or surface facts suggest missing test/build gates?",
                "What commands are only suggested/not-run rather than verified?",
            ],
        ),
    }


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
