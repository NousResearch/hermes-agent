#!/usr/bin/env python3
"""analyze_graph.py — UA-003 Deterministic Graph Analytics Layer.

Reads a graph.json produced by assemble_graph.py and emits deterministic
analytics as analytics.json (written to stdout or --output path).

Analytics are hints, not definitive architectural judgments. All output fields include
confidence scores and deterministic reason strings.

No network calls. Stdlib only.

Usage:
    python analyze_graph.py <graph.json> [--output analytics.json]

Stdout: machine-readable JSON.
Stderr: progress/log messages only.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# ── Degree computation ───────────────────────────────────────────────────────

def compute_in_degree(graph: dict) -> Dict[str, int]:
    """Compute in-degree for every node referenced in edges.

    In-degree = number of edges where the node is the target.
    Returns {node_id: count} for nodes with in-degree > 0.
    """
    degrees: Dict[str, int] = defaultdict(int)
    for edge in graph.get("edges", []):
        target = edge.get("target", "")
        if target:
            degrees[target] += 1
    return dict(degrees)


def compute_out_degree(graph: dict) -> Dict[str, int]:
    """Compute out-degree for every node referenced in edges.

    Out-degree = number of edges where the node is the source.
    Returns {node_id: count} for nodes with out-degree > 0.
    """
    degrees: Dict[str, int] = defaultdict(int)
    for edge in graph.get("edges", []):
        source = edge.get("source", "")
        if source:
            degrees[source] += 1
    return dict(degrees)


# ── Top-K ranking ────────────────────────────────────────────────────────────

def top_k_by_degree(
    degrees: Dict[str, int],
    graph: dict,
    *,
    k: int = 5,
    degree_type: str = "in",
) -> List[dict]:
    """Return the top-k nodes by degree, with node metadata.

    Each entry: {"node_id": ..., "degree": N, "label": ..., "filePath": ...,
                  "confidence": 1.0, "reason": ...}
    Sorted descending by degree; ties broken by node_id for determinism.
    """
    if not degrees:
        return []

    # Build a lookup from node_id -> node data
    node_map: Dict[str, dict] = {}
    for node in graph.get("nodes", []):
        nid = node.get("node_id", "")
        if nid:
            node_map[nid] = node

    # Sort: primary by degree descending, secondary by node_id ascending (deterministic)
    sorted_nodes = sorted(degrees.items(), key=lambda x: (-x[1], x[0]))
    top_k = sorted_nodes[:k]

    result = []
    for node_id, degree in top_k:
        node_data = node_map.get(node_id, {})
        if degree_type == "in":
            reason = f"Highest in-degree: {degree} files/modules import this"
        else:
            reason = f"Highest out-degree: this file/module imports {degree} others"

        entry = {
            "node_id": node_id,
            "degree": degree,
            "label": node_data.get("label", ""),
            "filePath": node_data.get("filePath"),
            "confidence": 1.0,
            "reason": reason,
        }
        result.append(entry)

    return result


# ── Bidirectional / cyclic import detection ──────────────────────────────────

def find_bidirectional_imports(graph: dict) -> List[Tuple[str, str]]:
    """Detect bidirectional import pairs between file-level nodes.

    A → B and B → A (via module references that map back to files) forms
    a bidirectional pair. Returns list of sorted (file_id_a, file_id_b) tuples.

    For file nodes, we check if file X imports a module named after file Y
    AND file Y imports a module named after file X.
    """
    # Build adjacency: file_id -> set of module targets
    file_to_modules: Dict[str, set] = defaultdict(set)
    for edge in graph.get("edges", []):
        source = edge.get("source", "")
        target = edge.get("target", "")
        if source.startswith("file:") and target.startswith("module:"):
            file_to_modules[source].add(target)

    # Build reverse mapping: which file does a module name suggest?
    # module:src.utils → file:src/utils.py heuristic
    file_ids = set()
    for node in graph.get("nodes", []):
        if node.get("node_type") == "file":
            file_ids.add(node["node_id"])

    module_to_file: Dict[str, str] = {}
    for node in graph.get("nodes", []):
        if node.get("node_type") == "file":
            node_id = node["node_id"]
            file_path = node.get("filePath", "")
            # Derive module name from file path (e.g., src/utils.py → src.utils)
            if file_path:
                module_name = file_path.replace("/", ".").replace(".py", "")
                module_id = f"module:{module_name}"
                module_to_file[module_id] = node_id

    # Find bidirectional pairs
    seen_pairs: set = set()
    bidirectional: List[Tuple[str, str]] = []

    for file_a, modules_a in file_to_modules.items():
        for mod in modules_a:
            file_b = module_to_file.get(mod)
            if file_b and file_b in file_to_modules:
                modules_b = file_to_modules[file_b]
                # Can file_b reach something mapping to file_a?
                for mod_b in modules_b:
                    file_a_via_b = module_to_file.get(mod_b)
                    if file_a_via_b == file_a:
                        pair: Tuple[str, str] = tuple(sorted([file_a, file_b]))  # type: ignore[assignment]
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            bidirectional.append(pair)

    return bidirectional


# ── Entrypoint detection ─────────────────────────────────────────────────────

def find_entrypoint_candidates(graph: dict) -> List[dict]:
    """Identify likely entrypoint files using deterministic heuristics.

    Heuristics (scored):
    - File named "main" (in name or path): +0.5
    - Has a function named "main": +0.3
    - High out-degree relative to peers (imports many others): up to 0.2
    - Is in root directory or top-level: +0.1
    - Not in tests/ directory: +0.1

    Returns list of {"node_id", "confidence", "reason"} entries.
    Confidence is a float 0.0–1.0.
    """
    if not graph.get("nodes"):
        return []

    candidates: List[dict] = []

    # Compute out-degree for scoring
    out_deg = compute_out_degree(graph)
    max_out = max(out_deg.values()) if out_deg else 0

    for node in graph.get("nodes", []):
        if node.get("node_type") != "file":
            continue

        node_id = node["node_id"]
        file_path = node.get("filePath", "")
        label = node.get("label", "")
        functions = node.get("functions", [])

        score = 0.0
        reasons: List[str] = []

        # Heuristic: file name or path contains "main"
        if "main" in label.lower() or "main" in file_path.lower():
            score += 0.5
            reasons.append("filename contains 'main'")

        # Heuristic: has a function named "main"
        if "main" in functions:
            score += 0.3
            reasons.append("defines function 'main'")

        # Heuristic: high out-degree relative to max
        if max_out > 0 and node_id in out_deg:
            ratio = out_deg[node_id] / max_out
            score += ratio * 0.2
            reasons.append(f"high out-degree ({out_deg[node_id]})")

        # Heuristic: not in tests/ directory
        if not file_path.startswith("tests/") and "/tests/" not in file_path:
            score += 0.1
            reasons.append("not in tests directory")

        # Heuristic: root-level file (no directory prefix)
        if "/" not in file_path:
            score += 0.1
            reasons.append("root-level file")

        # Cap confidence at 1.0
        confidence = min(score, 1.0)

        # Only include if there's some signal
        if confidence > 0:
            candidates.append({
                "node_id": node_id,
                "filePath": file_path,
                "confidence": round(confidence, 4),
                "reason": "; ".join(reasons),
            })

    # Sort deterministically: by confidence descending, then node_id ascending
    candidates.sort(key=lambda x: (-x["confidence"], x["node_id"]))

    return candidates


# ── Directory-level dependency summary ───────────────────────────────────────

def _get_directory(file_path: str) -> str:
    """Extract the top-level directory from a file path.

    'src/utils.py' → 'src'
    'tests/test_main.py' → 'tests'
    'main.py' → '.' (root)
    """
    parts = file_path.split("/")
    if len(parts) > 1:
        return parts[0]
    return "."


def compute_directory_summary(graph: dict) -> Dict[str, dict]:
    """Build a directory-level dependency summary.

    For each directory:
    - file_count: number of file nodes in the directory
    - internal_edges: edges between files within the same directory
    - external_edges: edges from this directory to other directories
    - languages: set of languages found in this directory

    Returns {dir_name: {file_count, internal_edges, external_edges, languages}}.
    """
    # Map node_id -> directory
    node_to_dir: Dict[str, str] = {}
    dir_files: Dict[str, set] = defaultdict(set)
    dir_languages: Dict[str, set] = defaultdict(set)

    for node in graph.get("nodes", []):
        if node.get("node_type") != "file":
            continue
        node_id = node["node_id"]
        file_path = node.get("filePath", "")
        if not file_path:
            continue
        d = _get_directory(file_path)
        node_to_dir[node_id] = d
        dir_files[d].add(node_id)
        lang = node.get("language", "unknown")
        dir_languages[d].add(lang)

    # Count edges
    dir_internal: Dict[str, int] = defaultdict(int)
    dir_external: Dict[str, int] = defaultdict(int)

    # For mapping module targets back to directories (via file nodes)
    module_to_dir: Dict[str, str] = {}
    for node in graph.get("nodes", []):
        if node.get("node_type") == "file":
            fp = node.get("filePath", "")
            if fp:
                mod_name = fp.replace("/", ".").replace(".py", "")
                module_to_dir[f"module:{mod_name}"] = _get_directory(fp)
        elif node.get("node_type") == "module":
            # For stdlib/external modules, use a special key
            pass

    for edge in graph.get("edges", []):
        source = edge.get("source", "")
        target = edge.get("target", "")
        src_dir = node_to_dir.get(source)
        if src_dir is None:
            # source might be a module node or unknown
            src_dir = module_to_dir.get(source)
        if src_dir is None:
            continue

        tgt_dir = node_to_dir.get(target)
        if tgt_dir is None:
            tgt_dir = module_to_dir.get(target)
        if tgt_dir is None:
            # External module — count as external edge
            dir_external[src_dir] += 1
            continue

        if tgt_dir == src_dir:
            dir_internal[src_dir] += 1
        else:
            dir_external[src_dir] += 1

    # Build summary
    all_dirs = set(dir_files.keys())
    result = {}
    for d in sorted(all_dirs):
        result[d] = {
            "file_count": len(dir_files[d]),
            "internal_edges": dir_internal.get(d, 0),
            "external_edges": dir_external.get(d, 0),
            "languages": sorted(dir_languages.get(d, set())),
        }

    return result


# ── Review priority candidates ───────────────────────────────────────────────

def find_review_priority_candidates(graph: dict) -> List[dict]:
    """Identify files that deserve review attention using deterministic signals.

    Signals:
    - High out-degree (depends on many things): score = normalized_out_degree * 0.4
    - High in-degree (many things depend on it): score = normalized_in_degree * 0.4
    - Orphan node (not connected): score = 0.3
    - Has functions AND classes (complex): score = 0.2
    - Bidirectional involvement: score = 0.2

    Returns list sorted by score descending with deterministic reasons.
    """
    if not graph.get("nodes"):
        return []

    in_deg = compute_in_degree(graph)
    out_deg = compute_out_degree(graph)
    bidi = find_bidirectional_imports(graph)
    bidi_files = set()
    for a, b in bidi:
        bidi_files.add(a)
        bidi_files.add(b)

    # Referenced nodes (for orphan detection)
    referenced: set = set()
    for edge in graph.get("edges", []):
        referenced.add(edge.get("source", ""))
        referenced.add(edge.get("target", ""))

    max_in = max(in_deg.values()) if in_deg else 0
    max_out = max(out_deg.values()) if out_deg else 0

    candidates: List[dict] = []

    for node in graph.get("nodes", []):
        if node.get("node_type") != "file":
            continue

        node_id = node["node_id"]
        file_path = node.get("filePath", "")
        functions = node.get("functions", [])
        classes = node.get("classes", [])

        score = 0.0
        reasons: List[str] = []

        # High in-degree signal
        if max_in > 0 and node_id in in_deg:
            ratio = in_deg[node_id] / max_in
            s = ratio * 0.4
            score += s
            reasons.append(f"high in-degree ({in_deg[node_id]})")

        # High out-degree signal
        if max_out > 0 and node_id in out_deg:
            ratio = out_deg[node_id] / max_out
            s = ratio * 0.4
            score += s
            reasons.append(f"high out-degree ({out_deg[node_id]})")

        # Orphan signal
        if node_id not in referenced:
            score += 0.3
            reasons.append("orphan node (no edges)")

        # Complexity signal
        if functions and classes:
            score += 0.2
            reasons.append(f"defines {len(functions)} function(s) and {len(classes)} class(es)")

        # Bidirectional involvement
        if node_id in bidi_files:
            score += 0.2
            reasons.append("involved in bidirectional import")

        score = round(min(score, 1.0), 4)

        if score > 0:
            candidates.append({
                "node_id": node_id,
                "filePath": file_path,
                "score": score,
                "reason": "; ".join(reasons),
            })

    candidates.sort(key=lambda x: (-x["score"], x["node_id"]))

    return candidates


# ── Orphan severity classes (UA-002 integration) ────────────────────────────

def collect_orphan_severity_classes(graph: dict) -> List[dict]:
    """Collect orphan warning severity classes if available from UA-002.

    UA-002 severity fields are attached to the graph via validation output.
    If absent (UA-002 not present), return empty list explicitly.
    """
    # The graph.json itself does not carry severity classes — they come
    # from validation.json. Since our input is graph.json alone, we check
    # for an optional "severity_classified_warnings" field that could have
    # been embedded by a prior pipeline step.
    classified = graph.get("severity_classified_warnings", [])
    if not classified:
        # UA-002 severity fields absent — omit explicitly
        return []
    return list(classified)


# ── Full analysis pipeline ───────────────────────────────────────────────────

_DEFAULT_TOP_K = 5


def analyze_graph(graph: dict) -> dict:
    """Run the full deterministic analytics pipeline on a graph dict.

    Returns analytics dict with:
    - top_in_degree: top-k nodes by in-degree
    - top_out_degree: top-k nodes by out-degree
    - bidirectional_imports: file-level bidirectional import pairs
    - entrypoint_candidates: likely entrypoints with confidence scores
    - directory_summary: per-directory dependency breakdown
    - review_priority: files flagged for review with deterministic reasons
    - orphan_severity_classes: UA-002 severity data if available, else []
    """
    in_deg = compute_in_degree(graph)
    out_deg = compute_out_degree(graph)

    return {
        "top_in_degree": top_k_by_degree(in_deg, graph, k=_DEFAULT_TOP_K, degree_type="in"),
        "top_out_degree": top_k_by_degree(out_deg, graph, k=_DEFAULT_TOP_K, degree_type="out"),
        "bidirectional_imports": [
            {"file_a": a, "file_b": b} for a, b in find_bidirectional_imports(graph)
        ],
        "entrypoint_candidates": find_entrypoint_candidates(graph),
        "directory_summary": compute_directory_summary(graph),
        "review_priority": find_review_priority_candidates(graph),
        "orphan_severity_classes": collect_orphan_severity_classes(graph),
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> int:
    """CLI entry point. Read graph.json, emit analytics JSON to stdout or file."""
    parser = argparse.ArgumentParser(
        description="Deterministic graph analytics — UA-003"
    )
    parser.add_argument(
        "graph_file",
        help="Path to graph.json input",
    )
    parser.add_argument(
        "--output",
        help="Path to write analytics.json (default: stdout)",
    )
    args = parser.parse_args()

    # Read input
    graph_path = Path(args.graph_file)
    if not graph_path.is_file():
        print(f"Error: graph file not found: {args.graph_file}", file=sys.stderr)
        return 1

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {args.graph_file}: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded graph from {args.graph_file}", file=sys.stderr)

    # Analyze
    analytics = analyze_graph(graph_data)

    # Output
    output_json = json.dumps(analytics, indent=2, sort_keys=False)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_json)
            f.write("\n")
        print(f"Analytics written to {args.output}", file=sys.stderr)
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
