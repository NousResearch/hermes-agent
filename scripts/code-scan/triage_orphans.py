#!/usr/bin/env python3
"""Orphan Warning Triage for the UA Flywheel code-scan module.

Reads a graph.json (from assemble_graph) and scan.json (from scan_project),
optionally entrypoints.json (from detect_entrypoints), and classifies orphan
nodes into expected, entrypoint_candidate, suspicious, and unknown categories.

This is a companion artifact — it does NOT change validation gate semantics.
validation-gate still reports schema issues/warnings deterministically.
This script only summarizes which orphan warnings deserve human/agent attention.

JIT-only, read-only against target repos. No code execution. No new deps.
No tree-sitter or binary-runtime deps. No SQLite/vector. No LLM/provider strings.

Usage:
    python scripts/code-scan/triage_orphans.py <graph.json> <scan.json> \
        [--entrypoints entrypoints.json] > <orphan-triage.json>

Exit codes: 0 = success, 1 = error.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCHEMA_VERSION = "1.0.0"

# ── Recognized source-language set for suspicious/entrypoint classification ─

_SOURCE_LANGUAGES = {
    "python", "javascript", "typescript", "go", "rust", "c", "c++", "cpp",
    "java", "ruby", "php", "swift", "kotlin", "scala", "r", "perl",
    "lua", "bash", "shell", "powershell",
}

# ── Expected orphan patterns ──────────────────────────────────────────────

# Directory prefixes that indicate expected orphans (non-source)
_EXPECTED_DIR_PREFIXES = [
    "docs/", ".docs/", "documentation/",
    "tests/", "test/", "testsuite/", "testing/",
    "fixtures/", "fixture/", "test/fixtures/",
    ".github/workflows/", ".gitlab/", "ci/",
    "assets/", "images/", "media/", "static/", "img/",
    "templates/", "views/",
    "data/", "resources/",
]

# Filename patterns (lowercase) that indicate expected orphans
_EXPECTED_FILE_PATTERNS = [
    "readme", "license", "contributing", "code_of_conduct",
    "changelog", "changes", "history",
    ".env", ".env.example", ".gitignore", ".dockerignore",
    ".editorconfig", "makefile", "dockerfile",
    "setup.cfg", "setup.py", "pyproject.toml",
    "requirements.txt", "requirements",
    "package.json", "package-lock.json", "yarn.lock",
    "go.mod", "go.sum", "cargo.toml", "cargo.lock",
    "gemfile", "composer.json",
    ".flake8", ".pylintrc", "mypy.ini", ".mypy.ini",
    "tox.ini", ".pre-commit-config.yaml",
]

# File extensions that indicate expected orphans
_EXPECTED_EXTENSIONS = {
    ".md", ".rst", ".txt", ".text",  # documentation
    ".yaml", ".yml", ".toml", ".ini", ".cfg",  # config
    ".json", ".xml",  # data/config (when in non-source contexts)
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",  # images
    ".html", ".htm",  # templates
    ".css", ".scss", ".sass", ".less",  # styles
    ".csv", ".tsv",  # data
}


def _is_expected_orphan(node: dict) -> Tuple[str, bool]:
    """Check if a node matches expected-orphan heuristics.

    Returns (reason, is_expected).
    If matched, reason is one of: doc, config, test, fixture, workflow,
    asset, template, license.
    """
    file_path = node.get("filePath", "") or ""
    node_id = node.get("node_id", "") or ""
    language = (node.get("language", "") or "").lower()

    fp_lower = file_path.lower()
    basename = Path(file_path).name.lower() if file_path else Path(node_id).name.lower()

    # ── documentation ──
    for prefix in ["docs/", ".docs/", "documentation/"]:
        if fp_lower.startswith(prefix):
            return ("doc", True)

    for ext in (".md", ".rst", ".txt", ".text"):
        if basename.endswith(ext):
            return ("doc", True)

    for pattern in ("readme", "contributing"):
        if basename.startswith(pattern):
            return ("doc", True)

    # ── license ──
    for pattern in ("license", "licence"):
        if basename.startswith(pattern):
            return ("license", True)

    # ── changelog ──
    for pattern in ("changelog", "changes", "history"):
        if basename.startswith(pattern):
            return ("doc", True)

    # ── tests ──
    for prefix in ["tests/", "test/", "testsuite/", "testing/"]:
        if fp_lower.startswith(prefix):
            return ("test", True)

    # ── fixtures ──
    for prefix in ["fixtures/", "fixture/", "test/fixtures/"]:
        if fp_lower.startswith(prefix):
            return ("fixture", True)

    # ── workflows ──
    for prefix in [".github/workflows/", ".gitlab/", "ci/"]:
        if fp_lower.startswith(prefix):
            return ("workflow", True)

    # ── assets/images ──
    for prefix in ["assets/", "images/", "media/", "static/", "img/"]:
        if fp_lower.startswith(prefix):
            return ("asset", True)

    for ext in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp"):
        if basename.endswith(ext):
            return ("image", True)

    # ── templates ──
    for prefix in ["templates/", "views/"]:
        if fp_lower.startswith(prefix):
            return ("template", True)

    if basename.endswith((".html", ".htm")):
        return ("template", True)

    # ── config ──
    if not file_path or file_path is None:
        # Fall through — no path, can't classify as expected
        return ("", False)

    for prefix in _EXPECTED_DIR_PREFIXES:
        if fp_lower.startswith(prefix):
            reason = prefix.split("/")[0].strip(".")
            return (reason or "config", True)

    for pattern in _EXPECTED_FILE_PATTERNS:
        if basename == pattern.lower() or basename.startswith(pattern.lower()):
            return ("config", True)

    for ext in _EXPECTED_EXTENSIONS:
        if basename.endswith(ext):
            # Only classify as expected if not in a source code directory
            # and language is not a source language
            if language not in _SOURCE_LANGUAGES:
                return ("config", True)

    return ("", False)


def _is_entrypoint_candidate(node: dict, entrypoints_data: Optional[dict]) -> bool:
    """Check if a node's file is listed as an entrypoint in entrypoints.json."""
    if entrypoints_data is None:
        return False

    file_path = node.get("filePath", "")
    if not file_path:
        return False

    ep_files = set()
    for ep in entrypoints_data.get("entrypoints", []):
        ep_file = ep.get("file", "")
        if ep_file:
            ep_files.add(ep_file)

    # Match by exact relative path.
    return file_path in ep_files


def classify_orphan(
    node: dict,
    entrypoints_data: Optional[dict],
) -> Tuple[str, str]:
    """Classify a single orphan node into a category.

    Returns (category, reason) where category is one of:
    - expected: docs, config, tests, fixtures, workflows, images/assets, templates
    - entrypoint_candidate: source orphan marked as likely standalone entrypoint
    - suspicious: source orphan not plausible entrypoint
    - unknown: missing metadata or unsupported language
    """
    file_path = node.get("filePath") or ""
    node_id = node.get("node_id", "")
    language = (node.get("language", "") or "").lower()

    # ── Unknown: missing filePath ──
    if not file_path and not node_id:
        return ("unknown", "missing metadata")

    if not file_path:
        return ("unknown", "missing filePath")

    # ── Unknown: unsupported language with no recognized extension ──
    basename = Path(file_path).name.lower()
    ext = Path(file_path).suffix.lower()

    if language and language not in _SOURCE_LANGUAGES:
        # Not a source language — check if it matches expected patterns first
        reason, is_exp = _is_expected_orphan(node)
        if is_exp:
            return ("expected", reason)
        return ("unknown", f"unsupported language: {language}")

    # No language info at all
    if not language:
        if ext and ext in _EXPECTED_EXTENSIONS:
            return ("expected", "config by extension")
        # Unknown language, no recognized extension
        # Could still be expected if directory-based
        reason, is_exp = _is_expected_orphan(node)
        if is_exp:
            return ("expected", reason)
        return ("unknown", "missing metadata")

    # ── Expected: not a source file, matches expected patterns ──
    reason, is_exp = _is_expected_orphan(node)
    if is_exp:
        return ("expected", reason)

    # ── Source files: entrypoint candidate or suspicious ──
    if language in _SOURCE_LANGUAGES:
        if _is_entrypoint_candidate(node, entrypoints_data):
            return ("entrypoint_candidate", "marked as entrypoint")
        return ("suspicious", "unreferenced source")

    # Fallback
    return ("unknown", "unclassified")


def triage_orphans(
    graph: dict,
    scan: dict,
    entrypoints_data: Optional[dict],
) -> dict:
    """Main triage logic.

    Args:
        graph: The assembled graph JSON with nodes and edges.
        scan: The scan JSON with file records.
        entrypoints_data: Optional entrypoints JSON (may be None).

    Returns:
        Structured triage result dict.
    """
    # Build set of referenced node IDs from edges
    referenced: set = set()
    for edge in graph.get("edges", []):
        referenced.add(edge.get("source", ""))
        referenced.add(edge.get("target", ""))

    # Find orphan nodes (not referenced by any edge)
    orphan_nodes = []
    for node in graph.get("nodes", []):
        nid = node.get("node_id", "")
        if nid and nid not in referenced:
            orphan_nodes.append(node)

    # Classify each orphan
    result: Dict[str, List[dict]] = {
        "expected": [],
        "entrypoint_candidate": [],
        "suspicious": [],
        "unknown": [],
    }

    for node in orphan_nodes:
        category, reason = classify_orphan(node, entrypoints_data)
        result[category].append({
            "node_id": node.get("node_id", ""),
            "reason": reason,
        })

    # Build totals
    totals = {
        "total_orphans": len(orphan_nodes),
        "expected": len(result["expected"]),
        "entrypoint_candidate": len(result["entrypoint_candidate"]),
        "suspicious": len(result["suspicious"]),
        "unknown": len(result["unknown"]),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "orphans": result,
        "totals": totals,
    }


def _build_result(
    graph_path: str,
    scan_path: str,
    entrypoints_path: Optional[str],
) -> dict:
    """Load files and run triage. Used by tests and main()."""
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    with open(scan_path, "r", encoding="utf-8") as f:
        scan = json.load(f)

    entrypoints_data = None
    if entrypoints_path:
        with open(entrypoints_path, "r", encoding="utf-8") as f:
            entrypoints_data = json.load(f)

    return triage_orphans(graph, scan, entrypoints_data)


def main() -> int:
    """CLI entry point. Writes JSON to stdout."""
    parser = argparse.ArgumentParser(
        description="Triage orphan nodes in a dependency graph."
    )
    parser.add_argument(
        "graph_json",
        help="Path to graph.json (from assemble_graph)",
    )
    parser.add_argument(
        "scan_json",
        help="Path to scan.json (from scan_project)",
    )
    parser.add_argument(
        "--entrypoints",
        help="Optional path to entrypoints.json (from detect_entrypoints)",
        default=None,
    )
    args = parser.parse_args()

    if not Path(args.graph_json).is_file():
        print(f"Error: graph file '{args.graph_json}' not found", file=sys.stderr)
        return 1

    if not Path(args.scan_json).is_file():
        print(f"Error: scan file '{args.scan_json}' not found", file=sys.stderr)
        return 1

    try:
        result = _build_result(args.graph_json, args.scan_json, args.entrypoints)
        print(json.dumps(result, indent=2))
        return 0
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
