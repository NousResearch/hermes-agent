#!/usr/bin/env python3
"""project_state_append.py — UA-006 Project-State Integration Hook.

Append compact, deterministic UA run summaries to a project's
``.hermes/PROJECT_STATE.md`` ledger.  The ledger is **never** overwritten;
content is appended only.  When the ledger is absent, no state is written
and ``project_state_recorded`` is reported as ``false``.

Usage (programmatic):
    from project_state_append import append_project_state

    results = {
        \"manifest\": {\"run_id\": \"...\", \"mode\": \"structure\",
                       \"target_path\": \"/proj\", \"bundle_dir\": \"/bundle\",
                       \"artifact_paths\": {\"scan.json\": \"...\"}},
        \"scan\": {\"total_files\": 42, \"languages\": {\"python\": 30, ...}},
        \"graph\": {\"summary\": {\"total_nodes\": 120, \"total_edges\": 95}},
        \"validation\": {\"issues\": [...], \"warnings\": [...]},
        \"context\": {\"validation\": {\"verdict\": \"pass\"}},
    }

    status = append_project_state(results, project_root=\"/proj\")
    # status = {\"project_state_recorded\": true, \"ledger_path\": \"...\"}

Usage (CLI):
    python project_state_append.py /path/to/project --manifest /path/to/manifest.json

No huge JSON blobs are written to the ledger; only a path reference to the
artifact bundle is recorded.  All recorded information is deterministic
(no LLM/ML judgement).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

PROJECT_STATE_LEDGER_NAME = "PROJECT_STATE.md"
_HERMES_DIR = ".hermes"

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------


def append_project_state(
    results: Dict[str, Any],
    project_root: str,
) -> Dict[str, Any]:
    """Append a compact UA section to the project-state ledger.

    Args:
        results: Dict containing at minimum:
            - manifest: run metadata (run_id, mode, target_path, bundle_dir,
                        artifact_paths)
            - scan: scan output (total_files, languages)
            - graph: graph output (summary.total_nodes, summary.total_edges)
            - validation: validation output (issues, warnings)
            - context: context envelope (optional; validation.verdict)
            - runtime_readiness: (optional) readiness artifact summary
            - cleanliness: (optional) target cleanliness summary
        project_root: Absolute path to the project root.

    Returns:
        Dict with:
            - project_state_recorded: bool
            - ledger_path: str or None
            - project_state_append_status: "not_attempted" | "success" | "failed"
            - project_state_append_error: str or None (sanitized error message)
    """
    ledger_path = _find_ledger_path(project_root)

    if ledger_path is None:
        return {
            "project_state_recorded": False,
            "ledger_path": None,
            "project_state_append_status": "not_attempted",
            "project_state_append_error": None,
        }

    ua_section = _build_ua_section(results)

    try:
        # Read existing content, normalize EOF, then re-write + append
        with open(ledger_path, "r", encoding="utf-8") as f:
            existing = f.read()

        normalized = _normalize_eof(existing)

        with open(ledger_path, "w", encoding="utf-8") as f:
            f.write(normalized)
            f.write(ua_section)

        return {
            "project_state_recorded": True,
            "ledger_path": ledger_path,
            "project_state_append_status": "success",
            "project_state_append_error": None,
        }
    except Exception as exc:  # noqa: BLE001
        # Sanitize error message: keep it short, no paths/tracebacks
        msg = str(exc)
        if len(msg) > 200:
            msg = msg[:200] + "..."
        return {
            "project_state_recorded": False,
            "ledger_path": ledger_path,
            "project_state_append_status": "failed",
            "project_state_append_error": msg,
        }


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------


def _find_ledger_path(project_root: str) -> Optional[str]:
    """Return the path to ``.hermes/PROJECT_STATE.md`` if it exists.

    Returns None when the ledger is absent.  The search is non-recursive:
    it looks only in ``<project_root>/.hermes/PROJECT_STATE.md``.
    """
    candidate = os.path.join(
        project_root, _HERMES_DIR, PROJECT_STATE_LEDGER_NAME,
    )
    if os.path.isfile(candidate):
        return os.path.realpath(candidate)
    return None


def _normalize_eof(content: str) -> str:
    """Normalize trailing whitespace so content ends with exactly one ``\\n``."""
    return content.rstrip("\n") + "\n"


def _build_ua_section(results: Dict[str, Any]) -> str:
    """Build a compact, deterministic UA section for the ledger.

    The section contains only:
    - run_id
    - mode
    - target_path
    - artifact_bundle_path (linked, not embedded)
    - validation_verdict
    - issue_count, warning_count
    - file_count (total files scanned)
    - top 5 languages
    - graph node/edge count
    - verification_status (from runtime_readiness, if present)
    - blockers (from runtime_readiness, capped to 3)
    - target_cleanliness_status (from cleanliness, if present)
    - unexpected_changes_count (from cleanliness, if present)
    - next recommended action (deterministic heuristic)
    - timestamp

    No LLM/ML judgement fields are included.
    """
    manifest = results.get("manifest", {})
    scan = results.get("scan", {})
    graph = results.get("graph", {})
    validation = results.get("validation", {})
    context = results.get("context", {})
    runtime_readiness = results.get("runtime_readiness", {})
    cleanliness = results.get("cleanliness", {})

    run_id = manifest.get("run_id", "unknown")
    mode = manifest.get("mode", "unknown")
    target_path = manifest.get("target_path", "unknown")
    bundle_dir = manifest.get("bundle_dir", "unknown")

    # Validation verdict — deterministic from context or validation data
    verdict = _get_verdict(validation, context)

    # Counts
    issue_count = len(validation.get("issues", []))
    warning_count = len(validation.get("warnings", []))

    # Scan facts
    file_count = scan.get("total_files", 0)
    top_5 = _top_n_languages(scan.get("languages", {}), 5)

    # Graph facts
    graph_summary = graph.get("summary", {})
    node_count = graph_summary.get("total_nodes", 0)
    edge_count = graph_summary.get("total_edges", 0)

    # Next recommended action — deterministic heuristic
    next_action = _next_recommended_action(verdict, issue_count, warning_count)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "",
        f"## UA Run — {run_id}",
        "",
        f"- run_id: {run_id}",
        f"- mode: {mode}",
        f"- target_path: {target_path}",
        f"- artifact_bundle_path: {bundle_dir}",
        f"- validation_verdict: {verdict}",
        f"- issue_count: {issue_count}",
        f"- warning_count: {warning_count}",
        f"- file_count: {file_count}",
        f"- top_5_languages: {', '.join(top_5) if top_5 else 'none'}",
        f"- graph_nodes: {node_count}",
        f"- graph_edges: {edge_count}",
    ]

    # Runtime readiness summary (UA-P1-004)
    if runtime_readiness:
        verif_status = runtime_readiness.get("verification_status")
        if verif_status:
            lines.append(f"- verification_status: {verif_status}")
        blockers = runtime_readiness.get("blockers", [])
        if blockers:
            # Cap to top 3
            capped = blockers[:3]
            lines.append(f"- blockers (top 3): {'; '.join(capped)}")

    # Target cleanliness summary (UA-P1-004)
    if cleanliness:
        clean_status = cleanliness.get("target_cleanliness_status")
        if clean_status:
            lines.append(f"- target_cleanliness_status: {clean_status}")
        unexpected_count = cleanliness.get("unexpected_changes_count")
        if unexpected_count is not None:
            lines.append(f"- unexpected_changes_count: {unexpected_count}")

    lines.append(f"- next_recommended_action: {next_action}")
    lines.append(f"- timestamp: {timestamp}")
    lines.append("")

    return "\n".join(lines)


def _get_verdict(
    validation: Dict[str, Any],
    context: Dict[str, Any],
) -> str:
    """Derive a deterministic verdict from validation/context data."""
    # Prefer context.verdict if available (assembled by build_context_bundle)
    ctx_verdict = (
        context.get("validation", {}).get("verdict")
    )
    if ctx_verdict and ctx_verdict != "unknown":
        return ctx_verdict

    # Fall back to deterministic derivation from validation data
    issues = validation.get("issues", [])
    warnings = validation.get("warnings", [])

    if len(issues) == 0 and len(warnings) == 0:
        return "pass"
    elif len(issues) > 0:
        return "issues_found"
    else:
        return "warnings_only"


def _top_n_languages(languages: Dict[str, int], n: int = 5) -> list:
    """Return the top-n language names sorted by file count descending."""
    sorted_langs = sorted(languages.items(), key=lambda kv: kv[1], reverse=True)
    return [lang for lang, _count in sorted_langs[:n]]


def _next_recommended_action(
    verdict: str,
    issue_count: int,
    warning_count: int,
) -> str:
    """Deterministic next-step recommendation based on validation outcome."""
    if verdict == "pass":
        return "proceed to review or subagent handoff"
    elif verdict == "issues_found":
        if issue_count > 5:
            return "address critical issues before review"
        return f"resolve {issue_count} issue(s) and re-run validation"
    elif verdict == "warnings_only":
        return f"review {warning_count} warning(s); proceed if acceptable"
    else:
        return "investigate missing validation data"


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------


def main() -> int:
    """CLI entry point for appending project-state."""
    parser = argparse.ArgumentParser(
        description="Append UA run summary to .hermes/PROJECT_STATE.md ledger.",
    )
    parser.add_argument(
        "project_root",
        help="Path to the project root directory.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest.json from a UA bundle.",
    )
    args = parser.parse_args()

    project_root = os.path.realpath(args.project_root)
    if not os.path.isdir(project_root):
        print(
            f"Error: '{args.project_root}' is not a valid directory",
            file=sys.stderr,
        )
        return 1

    # Build results dict from manifest file if provided
    results: Dict[str, Any] = {"manifest": {}, "scan": {}, "graph": {},
                                "validation": {}, "context": {}}

    if args.manifest:
        try:
            with open(args.manifest, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            results["manifest"] = manifest

            # Try to load sibling artifacts from the same bundle
            bundle_dir = manifest.get("bundle_dir", "")
            artifact_paths = manifest.get("artifact_paths", {})
            for name in ("scan.json", "graph.json", "validation.json"):
                json_path = artifact_paths.get(name)
                if not json_path:
                    # Fallback: look in bundle_dir
                    if bundle_dir:
                        json_path = os.path.join(bundle_dir, name)
                if json_path and os.path.isfile(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        try:
                            results[name.replace(".json", "")] = json.load(f)
                        except (json.JSONDecodeError, OSError):
                            pass

            # Try context envelope
            ctx_path = artifact_paths.get("subagent-context.json")
            if not ctx_path and bundle_dir:
                ctx_path = os.path.join(bundle_dir, "subagent-context.json")
            if ctx_path and os.path.isfile(ctx_path):
                with open(ctx_path, "r", encoding="utf-8") as f:
                    try:
                        results["context"] = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        pass

        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error reading manifest: {exc}", file=sys.stderr)
            return 1

    status = append_project_state(results, project_root)
    print(json.dumps(status, indent=2))

    if status["project_state_recorded"]:
        return 0
    else:
        # Ledger absent — not an error, just informational
        return 0


if __name__ == "__main__":
    sys.exit(main())
