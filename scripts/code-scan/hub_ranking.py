#!/usr/bin/env python3
"""hub_ranking.py — Phase 4 D4: Architectural Hub Ranking.

Identifies the files most likely to be architectural anchors by computing
in-degree and out-degree centrality on the project dependency graph, scoring
hubs with a deterministic formula, and optionally weighting project-local
edges higher when classified-imports data is supplied.

Usage:
    python scripts/code-scan/hub_ranking.py <graph.json> \
        [--classified-imports classified-imports.json] \
        [--top 20] \
        [--include-non-code] \
        > hubs.json

Stdlib only — no external dependencies.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# ── Path classification ────────────────────────────────────────────────

# Patterns that indicate a non-code file (docs, config, assets).
_NON_CODE_PATTERNS: set[str] = {
    # Extensions
    ".md", ".rst", ".txt", ".markdown",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".env.example",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".css", ".scss", ".less",
    ".dat", ".bin",
    # Filenames (whole basename match)
    "makefile", "dockerfile", "readme", "changelog", "license", "authors",
    "contributors",
    # Config files
    "config.yaml", "config.yml", "config.json", "config.toml", "config.ini",
    "package.json", "package-lock.json", "yarn.lock", "poetry.lock",
    "pyproject.toml", "setup.cfg", "tox.ini", ".env",
}

_NON_CODE_DIRS: set[str] = {
    "docs", "doc", "documentation",
    "assets", "asset", "static", "images", "image", "img", "public",
    "media", "media",
}


def is_non_code_path(file_path: str) -> bool:
    """Return True if the file path looks like docs, config, or assets.

    Checks both the directory components and the file extension/basename
    against known non-code patterns.
    """
    if not file_path:
        return False

    parts = Path(file_path).parts

    # Check directory components
    for part in parts[:-1]:  # everything except the filename itself
        if part.lower() in _NON_CODE_DIRS:
            return True

    basename_lower = parts[-1].lower()

    # Check known filenames
    if basename_lower in _NON_CODE_PATTERNS:
        return True

    # Check extensions
    suffix = Path(file_path).suffix.lower()
    if suffix and suffix in _NON_CODE_PATTERNS:
        return True

    return False


# ── Degree computation ─────────────────────────────────────────────────

def _file_nodes(graph: dict) -> dict[str, dict]:
    """Return a mapping of node_id -> node_dict for file-type nodes only."""
    result: dict[str, dict] = {}
    for node in graph.get("nodes", []):
        if node.get("node_type") == "file":
            nid = node.get("node_id", "")
            if nid:
                result[nid] = node
    return result


def compute_in_degree(graph: dict) -> dict[str, int]:
    """Compute in-degree (incoming edges) for each file node.

    Only counts edges where both source and target are file nodes.
    """
    file_node_ids = set(_file_nodes(graph).keys())
    in_deg: dict[str, int] = {nid: 0 for nid in file_node_ids}

    for edge in graph.get("edges", []):
        target = edge.get("target", "")
        source = edge.get("source", "")
        if source in file_node_ids and target in file_node_ids:
            in_deg[target] = in_deg.get(target, 0) + 1

    return in_deg


def compute_out_degree(graph: dict) -> dict[str, int]:
    """Compute out-degree (outgoing edges) for each file node.

    Only counts edges where both source and target are file nodes.
    """
    file_node_ids = set(_file_nodes(graph).keys())
    out_deg: dict[str, int] = {nid: 0 for nid in file_node_ids}

    for edge in graph.get("edges", []):
        source = edge.get("source", "")
        target = edge.get("target", "")
        if source in file_node_ids and target in file_node_ids:
            out_deg[source] = out_deg.get(source, 0) + 1

    return out_deg


# ── Classified import helpers ──────────────────────────────────────────

def _get_classified_file_set(classified: Optional[dict]) -> set[str]:
    """Return the set of file paths present in classified-imports data."""
    if not classified or "files" not in classified:
        return set()
    return set(classified["files"].keys())


def _edge_is_local(
    source_node: dict,
    target_node: dict,
    classified: Optional[dict],
    path_to_id: dict[str, str],
) -> bool:
    """Determine if an edge represents a project-local import.

    Uses classified-imports data when available. If classified data covers
    the source file and marks the edge target as 'local' or 'relative', the
    edge is considered local. If classified data is absent for the source
    but both endpoints are file nodes within the project, treat it as local.
    """
    source_path = source_node.get("filePath", "")
    if not source_path:
        return False

    classified_files = classified.get("files", {}) if classified else {}
    classified_entry = classified_files.get(source_path)

    if classified_entry:
        # Check if any classified import for this source points to the target
        target_path = target_node.get("filePath", "")
        target_base = Path(target_path).stem if target_path else ""
        target_module = str(Path(target_path).with_suffix("")).replace("/", ".")

        for imp_entry in classified_entry.get("imports", []):
            classification = imp_entry.get("classification", "")
            if classification in ("local", "relative"):
                mod = imp_entry.get("module", "")
                # Check if this classified local import corresponds to the target
                mod_base = mod.split(".")[-1].split("/")[-1]
                if mod_base == target_base or mod == target_module:
                    return True
        # If source has classified data but target isn't marked local, it's not local
        return False

    # Without classified data, edges between two file nodes are treated as local
    target_path = target_node.get("filePath", "")
    return bool(target_path)


# ── Hub scoring ────────────────────────────────────────────────────────

def compute_hub_scores(
    graph: dict,
    classified: Optional[dict] = None,
) -> dict[str, float]:
    """Compute hub scores for all file nodes.

    Base formula: hub_score = in_degree + out_degree

    When classified imports are available, the score is refined:
    - Only edges identified as project-local (local/relative classification)
      contribute to the score. This filters out stdlib/third_party noise.
    - Local edges get a weight of 1.0.

    Without classified imports, all file-to-file edges count equally
    (weight 1.0).

    Returns a dict mapping node_id -> hub_score.
    """
    file_nodes = _file_nodes(graph)
    file_node_ids = set(file_nodes.keys())
    id_to_node = file_nodes

    in_deg = compute_in_degree(graph)
    out_deg = compute_out_degree(graph)

    classified_present = classified is not None and "files" in classified

    if classified_present:
        # With classified imports: score based on local edges only
        local_in: dict[str, float] = {nid: 0.0 for nid in file_node_ids}
        local_out: dict[str, float] = {nid: 0.0 for nid in file_node_ids}

        for edge in graph.get("edges", []):
            source_id = edge.get("source", "")
            target_id = edge.get("target", "")

            if source_id not in file_node_ids or target_id not in file_node_ids:
                continue

            source_node = id_to_node.get(source_id, {})
            target_node = id_to_node.get(target_id, {})

            if _edge_is_local(source_node, target_node, classified, {}):
                local_in[target_id] += 1.0
                local_out[source_id] += 1.0

        return {
            nid: local_in[nid] + local_out[nid]
            for nid in file_node_ids
        }
    else:
        # Without classified imports: use all file-to-file edges
        return {
            nid: float(in_deg.get(nid, 0) + out_deg.get(nid, 0))
            for nid in file_node_ids
        }


# ── Ranking ────────────────────────────────────────────────────────────

def rank_hubs(
    graph: dict,
    scores: dict[str, float],
    top: int = 20,
    classified: Optional[dict] = None,
    include_non_code: bool = False,
) -> list[dict[str, Any]]:
    """Rank file nodes by hub score, returning the top N.

    Non-code paths (docs, config, assets) are excluded by default unless
    include_non_code is True.

    Deterministic tie-breaking: nodes with equal scores are sorted by
    node_id lexicographically (ascending).

    Confidence is determined by classification coverage:
    - "low": no classified-imports data supplied
    - "high": all file nodes have classified-imports entries
    - "medium": partial classified-imports coverage
    """
    file_nodes = _file_nodes(graph)

    # Determine confidence based on classified coverage
    classified_file_set = _get_classified_file_set(classified)
    classified_present = len(classified_file_set) > 0
    all_files_have_classified = (
        classified_present
        and all(
            node.get("filePath", "") in classified_file_set
            for node in file_nodes.values()
        )
    )

    # Build candidate list
    candidates: list[dict[str, Any]] = []
    for nid, score in scores.items():
        node = file_nodes.get(nid)
        if not node:
            continue

        file_path = node.get("filePath", "")
        if not file_path:
            continue

        # Exclude non-code unless requested
        if not include_non_code and is_non_code_path(file_path):
            continue

        in_deg = compute_in_degree(graph).get(nid, 0)
        out_deg = compute_out_degree(graph).get(nid, 0)

        # Determine per-node confidence
        if not classified_present:
            confidence = "low"
        elif file_path in classified_file_set:
            confidence = "high"
        else:
            # Node exists but isn't in classified data
            if all_files_have_classified:
                confidence = "high"
            else:
                confidence = "medium"

        candidates.append({
            "node_id": nid,
            "file_path": file_path,
            "hub_score": score,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "confidence": confidence,
        })

    # Sort: by score descending, then node_id ascending for deterministic tie-breaking
    candidates.sort(key=lambda h: (-h["hub_score"], h["node_id"]))

    return candidates[:top]


# ── Output ─────────────────────────────────────────────────────────────

_DISCLAIMER = (
    "Hub scores are ranking hints, not proof of importance. "
    "They indicate files with many incoming or outgoing import edges, "
    "which may correlate with architectural centrality but should be "
    "interpreted alongside other signals (code reviews, module responsibilities, etc.)."
)


def _build_output(
    ranked: list[dict[str, Any]],
    top: int,
    classification_present: bool,
) -> dict[str, Any]:
    """Build the final output dict conforming to the hub-ranking schema."""
    output: dict[str, Any] = {
        "schema_version": "1.0.0",
        "hub_rankings": ranked,
        "entrypoint_like": [],
        "totals": {
            "files_ranked": len(ranked),
            "top_n": top,
        },
        "disclaimer": _DISCLAIMER,
    }

    if not classification_present:
        output["notes"] = (
            "No classified-imports data was supplied. "
            "Hub scores are computed from all file-to-file edges without "
            "distinguishing local vs third-party imports. "
            f"Coverage: 0% of files have classified import data."
        )

    return output


# ── Loading ────────────────────────────────────────────────────────────

def _load_graph(path: str) -> dict:
    """Load and return the graph JSON from *path*."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_classified_imports(path: str) -> Optional[dict]:
    """Load and return classified-imports JSON, or None if path is empty."""
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Classified-imports file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> int:
    """CLI entry point. Reads a graph.json, computes hub rankings, writes JSON."""
    parser = argparse.ArgumentParser(
        description="Compute architectural hub rankings from a dependency graph.",
    )
    parser.add_argument(
        "graph_json",
        help="Path to graph.json (assembled dependency graph)",
    )
    parser.add_argument(
        "--classified-imports",
        dest="classified_imports",
        default=None,
        help="Path to classified-imports.json (optional)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top hubs to return (default: 20)",
    )
    parser.add_argument(
        "--include-non-code",
        dest="include_non_code",
        action="store_true",
        default=False,
        help="Include docs/config/assets in rankings",
    )
    args = parser.parse_args()

    # Load graph
    try:
        graph = _load_graph(args.graph_json)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Load classified imports (optional)
    classified: Optional[dict] = None
    if args.classified_imports:
        try:
            classified = _load_classified_imports(args.classified_imports)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            print(f"Warning: could not load classified imports: {exc}", file=sys.stderr)

    classification_present = classified is not None

    # Compute scores
    scores = compute_hub_scores(graph, classified=classified)

    # Rank
    ranked = rank_hubs(
        graph,
        scores,
        top=args.top,
        classified=classified,
        include_non_code=args.include_non_code,
    )

    # Build output
    output = _build_output(ranked, top=args.top, classification_present=classification_present)

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
