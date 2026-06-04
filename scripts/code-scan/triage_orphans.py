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

UA-P5-003 (V2 taxonomy): Each orphan entry carries a fine-grained V2 category
plus confidence, confidence_label (high/medium/low), orphan_type (alias of
category), reason, and recommended_action. The four top-level groups
(expected, entrypoint_candidate, suspicious, unknown) are preserved for
backward compatibility with report-data consumers.

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
    ".beads/", "beads/",
    "docs/", ".docs/", "documentation/", "manuals/", "manual/",
    "tests/", "test/", "testsuite/", "testing/",
    "fixtures/", "fixture/", "test/fixtures/",
    ".github/workflows/", ".gitlab/", "ci/",
    "assets/", "images/", "media/", "static/", "img/", "public/",
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

# ── V2 Taxonomy: fine-grained categories ─────────────────────────────────

# Migration directory patterns
_MIGRATION_DIR_PREFIXES = [
    "supabase/migrations/",
    "prisma/migrations/",
    "migrations/",
    "alembic/versions/",
    "db/migrations/",
    "database/migrations/",
    "migraciones/",
]

# Extension-to-V2-category map for expected orphans (when no directory match hit)
_EXT_TO_V2_CATEGORY = {
    ".md": "expected_doc",
    ".rst": "expected_doc",
    ".txt": "expected_doc",
    ".text": "expected_doc",
    ".yaml": "expected_config",
    ".yml": "expected_config",
    ".toml": "expected_config",
    ".ini": "expected_config",
    ".cfg": "expected_config",
    ".json": "expected_config",
    ".xml": "expected_config",
    ".png": "expected_asset",
    ".jpg": "expected_asset",
    ".jpeg": "expected_asset",
    ".gif": "expected_asset",
    ".svg": "expected_asset",
    ".ico": "expected_asset",
    ".webp": "expected_asset",
    ".html": "expected_static_template",
    ".htm": "expected_static_template",
    ".css": "expected_config",
    ".scss": "expected_config",
    ".sass": "expected_config",
    ".less": "expected_config",
    ".csv": "expected_config",
    ".tsv": "expected_config",
}

# V2 category recommended actions
_V2_RECOMMENDED_ACTIONS = {
    "expected_planning_doc": "no_action_needed",
    "expected_doc": "no_action_needed",
    "expected_asset": "no_action_needed",
    "expected_config": "no_action_needed",
    "expected_test_fixture": "no_action_needed",
    "expected_migration": "review_via_domain_analyzer",
    "expected_static_template": "review",
    "entrypoint_candidate": "review",
    "possible_dead_source": "verify_import_resolution_and_runtime_usage",
    "import_resolution_anomaly": "verify_import_resolution",
    "unknown": "investigate",
}

# V2 category default confidence
_V2_DEFAULT_CONFIDENCE = {
    "expected_planning_doc": 0.95,
    "expected_doc": 0.95,
    "expected_asset": 0.95,
    "expected_config": 0.9,
    "expected_test_fixture": 0.95,
    "expected_migration": 0.85,
    "expected_static_template": 0.9,
    "entrypoint_candidate": 0.7,
    "possible_dead_source": 0.5,
    "import_resolution_anomaly": 0.6,
    "unknown": 0.1,
}

_V2_REASON_LABELS = {
    "expected_planning_doc": "planning bead or handoff document",
    "expected_doc": "documentation file",
    "expected_asset": "asset file",
    "expected_config": "configuration file",
    "expected_test_fixture": "test or fixture file",
    "expected_migration": "migration file",
    "expected_static_template": "static template file",
    "entrypoint_candidate": "marked as entrypoint",
    "possible_dead_source": "unreferenced source",
    "import_resolution_anomaly": "unresolved imports",
    "unknown": "missing metadata",
}

MAX_REPRESENTATIVE_EXAMPLES = 3
MAX_TOP_SUSPICIOUS_EXAMPLES = 5


def _is_expected_orphan(node: dict) -> Tuple[str, bool]:
    """Check if a node matches expected-orphan heuristics and return V2 category.

    Returns (v2_category, is_expected) where v2_category is one of:
      expected_doc, expected_asset, expected_config, expected_test_fixture,
      expected_migration, expected_static_template.
    If not matched, returns ("", False).
    """
    file_path = node.get("filePath", "") or ""
    node_id = node.get("node_id", "") or ""
    language = (node.get("language", "") or "").lower()

    fp_lower = file_path.lower()
    basename = Path(file_path).name.lower() if file_path else Path(node_id).name.lower()
    ext = Path(file_path).suffix.lower() if file_path else ""

    # ── planning / handoff docs ──
    for prefix in [".beads/", "beads/"]:
        if fp_lower.startswith(prefix):
            return ("expected_planning_doc", True)

    # ── documentation ──
    for prefix in ["docs/", ".docs/", "documentation/", "manuals/", "manual/"]:
        if fp_lower.startswith(prefix):
            return ("expected_doc", True)

    for pattern in ("readme", "contributing"):
        if basename.startswith(pattern):
            return ("expected_doc", True)

    for pattern in ("changelog", "changes", "history"):
        if basename.startswith(pattern):
            return ("expected_doc", True)

    # ── tests & fixtures ──
    for prefix in ["tests/", "test/", "testsuite/", "testing/"]:
        if fp_lower.startswith(prefix):
            return ("expected_test_fixture", True)

    for prefix in ["fixtures/", "fixture/", "test/fixtures/"]:
        if fp_lower.startswith(prefix):
            return ("expected_test_fixture", True)

    # ── workflows ──
    for prefix in [".github/workflows/", ".gitlab/", "ci/"]:
        if fp_lower.startswith(prefix):
            # workflow is an expected type — map to config for V2
            return ("expected_config", True)

    # ── migrations ──
    for prefix in _MIGRATION_DIR_PREFIXES:
        if fp_lower.startswith(prefix):
            return ("expected_migration", True)

    # SQL files in migration-like dirs by extension
    if ext == ".sql":
        for prefix in ["migrations/", "db/", "database/"]:
            if fp_lower.startswith(prefix):
                return ("expected_migration", True)

    # ── assets/images ──
    for prefix in ["assets/", "images/", "media/", "static/", "public/"]:
        if fp_lower.startswith(prefix):
            return ("expected_asset", True)

    # ── templates ──
    for prefix in ["templates/", "views/"]:
        if fp_lower.startswith(prefix):
            return ("expected_static_template", True)

    # ── license ──
    for pattern in ("license", "licence"):
        if basename.startswith(pattern):
            return ("expected_doc", True)

    # ── extension-based mapping ──
    if ext in _EXT_TO_V2_CATEGORY:
        return (_EXT_TO_V2_CATEGORY[ext], True)

    # ── config patterns ──
    for pattern in _EXPECTED_FILE_PATTERNS:
        if basename == pattern.lower() or basename.startswith(pattern.lower()):
            return ("expected_config", True)

    # ── dir-prefix fallback ──
    if not file_path or file_path is None:
        return ("", False)

    for prefix in _EXPECTED_DIR_PREFIXES:
        if fp_lower.startswith(prefix):
            return ("expected_config", True)

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


# ── Confidence label mapping ───────────────────────────────────────────

_CONFIDENCE_THRESHOLDS = [
    (0.8, "high"),
    (0.5, "medium"),
    (0.0, "low"),
]


def _confidence_label(confidence: float) -> str:
    """Return a human-readable confidence label: high, medium, or low."""
    for threshold, label in _CONFIDENCE_THRESHOLDS:
        if confidence >= threshold:
            return label
    return "low"


def _make_entry(
    category: str,
    reason: str,
    confidence: float = None,
    recommended_action: str = None,
    node_id: str = "",
) -> dict:
    """Build a V2-enriched orphan entry dict."""
    conf = confidence if confidence is not None else _V2_DEFAULT_CONFIDENCE.get(category, 0.5)
    action = recommended_action if recommended_action else _V2_RECOMMENDED_ACTIONS.get(category, "investigate")
    return {
        "node_id": node_id,
        "category": category,
        "orphan_type": category,
        "confidence": conf,
        "confidence_label": _confidence_label(conf),
        "reason": reason,
        "recommended_action": action,
    }


def _entry_sort_key(entry: dict) -> tuple:
    """Sort orphan entries deterministically by category, then node_id."""
    return (entry.get("category", ""), entry.get("node_id", ""))


def _suspicious_sort_key(entry: dict) -> tuple:
    """Sort suspicious examples by review value, then node_id."""
    priority = {
        "import_resolution_anomaly": 0,
        "possible_dead_source": 1,
    }
    return (priority.get(entry.get("category", ""), 9), entry.get("node_id", ""))


def build_orphan_summary(orphans: dict) -> dict:
    """Build grouped counts and bounded representative orphan examples.

    The raw orphan entries remain in ``orphans``. This companion summary is for
    human reports, where dumping every expected doc/asset/migration is noisy.
    """
    category_counts: Dict[str, int] = {}
    category_entries: Dict[str, List[dict]] = {}
    for group in ("expected", "entrypoint_candidate", "suspicious", "unknown"):
        for entry in orphans.get(group, []) or []:
            category = entry.get("category") or entry.get("orphan_type") or group
            category_counts[category] = category_counts.get(category, 0) + 1
            category_entries.setdefault(category, []).append(entry)

    representative_examples = {}
    for category, entries in sorted(category_entries.items()):
        representative_examples[category] = [
            entry.get("node_id", "")
            for entry in sorted(entries, key=_entry_sort_key)[:MAX_REPRESENTATIVE_EXAMPLES]
        ]

    suspicious_entries = []
    for entry in orphans.get("suspicious", []) or []:
        suspicious_entries.append({
            "node_id": entry.get("node_id", ""),
            "category": entry.get("category") or entry.get("orphan_type") or "suspicious",
            "reason": entry.get("reason", ""),
            "recommended_action": entry.get("recommended_action", ""),
        })

    return {
        "category_counts": dict(sorted(category_counts.items())),
        "representative_examples": representative_examples,
        "top_suspicious_examples": sorted(
            suspicious_entries, key=_suspicious_sort_key
        )[:MAX_TOP_SUSPICIOUS_EXAMPLES],
        "example_limit_per_category": MAX_REPRESENTATIVE_EXAMPLES,
        "top_suspicious_limit": MAX_TOP_SUSPICIOUS_EXAMPLES,
    }


def classify_orphan(
    node: dict,
    entrypoints_data: Optional[dict],
) -> Tuple[str, str, dict]:
    """Classify a single orphan node into a category with V2 enrichments.

    Returns (group, reason, entry_dict) where:
      - group is one of the four top-level groups:
        expected, entrypoint_candidate, suspicious, unknown
      - reason is a human-readable classification reason
      - entry_dict is a V2-enriched dict with:
        node_id, category, orphan_type (= category alias), confidence,
        confidence_label (high/medium/low), reason, recommended_action

    V2 categories: expected_doc, expected_asset, expected_config,
      expected_test_fixture, expected_migration, expected_static_template,
      entrypoint_candidate, possible_dead_source, import_resolution_anomaly, unknown.
    """
    file_path = node.get("filePath") or ""
    node_id = node.get("node_id", "")
    language = (node.get("language", "") or "").lower()

    # ── Unknown: missing filePath ──
    if not file_path and not node_id:
        v2 = "unknown"
        reason = "missing metadata"
        return ("unknown", reason, _make_entry(v2, reason, node_id=node_id))

    if not file_path:
        v2 = "unknown"
        reason = "missing filePath"
        return ("unknown", reason, _make_entry(v2, reason, node_id=node_id))

    # ── Check for import resolution anomaly before generic source classification ──
    unresolved = node.get("unresolved_imports") or node.get("import_errors") or []
    if language in _SOURCE_LANGUAGES and unresolved:
        v2 = "import_resolution_anomaly"
        reason = f"unresolved imports: {', '.join(unresolved[:3])}"
        return ("suspicious", reason, _make_entry(v2, reason, node_id=node_id))

    # ── Unknown: unsupported language with no recognized extension ──
    basename = Path(file_path).name.lower()
    ext = Path(file_path).suffix.lower()

    if language and language not in _SOURCE_LANGUAGES:
        # Not a source language — check if it matches expected patterns first
        v2_cat, is_exp = _is_expected_orphan(node)
        if is_exp:
            group = "expected"
            # Derive reason from V2 category
            reason_map = {
                "expected_doc": "documentation file",
                "expected_asset": "asset file",
                "expected_config": "configuration file",
                "expected_test_fixture": "test or fixture file",
                "expected_migration": "migration file",
                "expected_static_template": "static template file",
            }
            reason = reason_map.get(v2_cat, v2_cat)
            return (group, reason, _make_entry(v2_cat, reason, node_id=node_id))
        v2 = "unknown"
        reason = f"unsupported language: {language}"
        return ("unknown", reason, _make_entry(v2, reason, node_id=node_id))

    # No language info at all
    if not language:
        if ext and ext in _EXPECTED_EXTENSIONS:
            v2_cat, is_exp = _is_expected_orphan(node)
            if is_exp:
                reason_map = {
                    "expected_doc": "documentation file",
                    "expected_asset": "asset file",
                    "expected_config": "configuration file",
                    "expected_test_fixture": "test or fixture file",
                    "expected_migration": "migration file",
                    "expected_static_template": "static template file",
                }
                reason = reason_map.get(v2_cat, "config by extension")
                return ("expected", reason, _make_entry(v2_cat, reason, node_id=node_id))
            return ("expected", "config by extension", _make_entry("expected_config", "config by extension"))

        # Unknown language, no recognized extension
        reason, is_exp = _is_expected_orphan(node)
        if is_exp:
            v2_cat = reason
            reason_map = {
                "expected_doc": "documentation file",
                "expected_asset": "asset file",
                "expected_config": "configuration file",
                "expected_test_fixture": "test or fixture file",
                "expected_migration": "migration file",
                "expected_static_template": "static template file",
            }
            reason_str = reason_map.get(v2_cat, v2_cat)
            return ("expected", reason_str, _make_entry(v2_cat, reason_str, node_id=node_id))
        return ("unknown", "missing metadata", _make_entry("unknown", "missing metadata", node_id=node_id, confidence=0.1))

    # ── Expected: not a source file, matches expected patterns ──
    v2_cat, is_exp = _is_expected_orphan(node)
    if is_exp:
        reason_map = {
            "expected_doc": "documentation file",
            "expected_asset": "asset file",
            "expected_config": "configuration file",
            "expected_test_fixture": "test or fixture file",
            "expected_migration": "migration file",
            "expected_static_template": "static template file",
        }
        reason = reason_map.get(v2_cat, v2_cat)
        return ("expected", reason, _make_entry(v2_cat, reason, node_id=node_id))

    # ── Source files: entrypoint candidate or possible_dead_source ──
    if language in _SOURCE_LANGUAGES:
        if _is_entrypoint_candidate(node, entrypoints_data):
            v2 = "entrypoint_candidate"
            reason = "marked as entrypoint"
            return ("entrypoint_candidate", reason, _make_entry(v2, reason, node_id=node_id))
        v2 = "possible_dead_source"
        reason = "unreferenced source"
        return ("suspicious", reason, _make_entry(v2, reason, node_id=node_id))

    # Fallback
    v2 = "unknown"
    reason = "unclassified"
    return ("unknown", reason, _make_entry(v2, reason, node_id=node_id))


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
        group, reason, entry = classify_orphan(node, entrypoints_data)
        result[group].append(entry)

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
        "summary": build_orphan_summary(result),
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
