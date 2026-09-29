#!/usr/bin/env python3
"""assemble_graph.py — Phase 3 D3: merge batch outputs into unified dependency graph.

Reads multiple JSON inputs (scan outputs, import maps), builds normalized node IDs,
creates edges from import relationships, deduplicates, validates with graph_schema.py,
and outputs a unified graph JSON.

Usage:
    python assemble_graph.py <input1.json> [input2.json ...] [--output file.json] [--verbose]

Stdlib only — no external dependencies.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Ensure scripts/code-scan is on sys.path for sibling imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from graph_schema import validate_graph


# ── Input loading ────────────────────────────────────────────────────────────

def load_batch_inputs(paths: List[str]) -> List[dict]:
    """Read and parse all input JSON files.

    Returns list of parsed dicts.
    Raises ValueError on invalid JSON or missing file.
    """
    if not paths:
        return []

    results = []
    for path in paths:
        if not os.path.isfile(path):
            raise ValueError(f"Input file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            results.append(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    return results


# ── Node ID normalization ────────────────────────────────────────────────────

def normalize_node_id(node_type: str, identifier: str) -> str:
    """Normalize a node ID using the ID prefix scheme.

    Known prefixes: file, module, func, class
    Unknown types get prefix 'unknown'.
    Path separators normalized to forward slashes.
    """
    node_type_lower = node_type.lower()
    # Normalize backslashes to forward slashes in identifier
    identifier_normalized = identifier.replace("\\", "/")

    valid_prefixes = ("file", "module", "func", "class")
    if node_type_lower not in valid_prefixes:
        return f"unknown:{identifier_normalized}"

    return f"{node_type_lower}:{identifier_normalized}"


# ── Node building ────────────────────────────────────────────────────────────

def build_nodes_from_scan(scan_data: dict) -> List[dict]:
    """Extract file-level nodes from a scan output.

    Each file in scan_data["files"] becomes a node.
    """
    nodes: List[dict] = []
    for entry in scan_data.get("files", []):
        rel_path = entry.get("relative_path", "")
        node_id = normalize_node_id("file", rel_path)

        node: dict = {
            "node_id": node_id,
            "node_type": "file",
            "filePath": rel_path,
            "label": Path(rel_path).name if rel_path else rel_path,
            "language": entry.get("language", "unknown"),
        }

        # Include optional list fields if present
        for field in ("functions", "classes", "imports"):
            if field in entry and entry[field]:
                node[field] = list(entry[field])

        nodes.append(node)

    return nodes


def build_nodes_from_imports(import_data: dict) -> List[dict]:
    """Extract module nodes from import data.

    For each unique imported module across all files, create a module node.
    """
    seen_modules: set[str] = set()
    nodes: List[dict] = []

    files_map = import_data.get("files", {})
    for rel_path, file_info in files_map.items():
        for module_name in file_info.get("imports", []):
            if module_name not in seen_modules:
                seen_modules.add(module_name)
                node_id = normalize_node_id("module", module_name)
                nodes.append({
                    "node_id": node_id,
                    "node_type": "module",
                    "filePath": None,
                    "label": module_name,
                })

    return nodes


# ── Edge building ────────────────────────────────────────────────────────────

def build_edges_from_imports(import_data: dict) -> List[dict]:
    """Extract import edges from import data.

    For each file's imports, create edges: file -> module.
    """
    edges: List[dict] = []
    files_map = import_data.get("files", {})

    for rel_path, file_info in files_map.items():
        source_id = normalize_node_id("file", rel_path)
        for module_name in file_info.get("imports", []):
            target_id = normalize_node_id("module", module_name)
            edges.append({
                "source": source_id,
                "target": target_id,
                "edge_type": "imports",
                "meta": {},
            })

    return edges


# ── Deduplication ────────────────────────────────────────────────────────────

def deduplicate_nodes(nodes: List[dict]) -> tuple[List[dict], int]:
    """Deduplicate nodes by node_id.

    Merges attributes: keeps first-seen scalar values,
    appends unique sorted items from functions/classes/imports lists.

    Returns (deduplicated_list, count_of_removed_duplicates).
    """
    if not nodes:
        return [], 0

    seen: Dict[str, dict] = {}
    duplicates_removed = 0

    for node in nodes:
        nid = node.get("node_id", "")
        if nid not in seen:
            # Create a copy to avoid mutating originals
            seen[nid] = dict(node)
        else:
            # Merge this duplicate into the first-seen
            existing = seen[nid]
            # For list fields, merge unique items
            for field in ("functions", "classes", "imports"):
                if field in node:
                    if field not in existing:
                        existing[field] = list(node[field])
                    else:
                        # Combine, deduplicate, sort
                        merged = set(existing[field])
                        merged.update(node[field])
                        existing[field] = sorted(merged)
            # Scalar fields keep first-seen values (don't overwrite)
            duplicates_removed += 1

    return list(seen.values()), duplicates_removed


def deduplicate_edges(edges: List[dict]) -> tuple[List[dict], int]:
    """Deduplicate edges by (source, target, edge_type).

    Keep first instance. Returns (deduplicated_list, count_of_removed).
    """
    if not edges:
        return [], 0

    seen: set[tuple] = set()
    result: List[dict] = []
    duplicates_removed = 0

    for edge in edges:
        key = (
            edge.get("source", ""),
            edge.get("target", ""),
            edge.get("edge_type", ""),
        )
        if key not in seen:
            seen.add(key)
            result.append(edge)
        else:
            duplicates_removed += 1

    return result, duplicates_removed


# ── Merging ──────────────────────────────────────────────────────────────────

def merge_nodes(batch_nodes_lists: List[List[dict]]) -> List[dict]:
    """Merge nodes from multiple batches.

    Concatenate all node lists, then deduplicate.
    """
    all_nodes: List[dict] = []
    for batch in batch_nodes_lists:
        all_nodes.extend(batch)
    merged, _ = deduplicate_nodes(all_nodes)
    return merged


def merge_edges(batch_edges_lists: List[List[dict]]) -> List[dict]:
    """Merge edges from multiple batches.

    Concatenate all edge lists, then deduplicate.
    """
    all_edges: List[dict] = []
    for batch in batch_edges_lists:
        all_edges.extend(batch)
    merged, _ = deduplicate_edges(all_edges)
    return merged


# ── Import-map key canonicalization ─────────────────────────────────────────

def _build_canonical_map(scan_data_list: List[dict]) -> Dict[str, str]:
    """Build a lookup from scan file records: absolute or normalized path → canonical relative_path.

    For each file in all scan outputs, map both its absolute ``path`` and its
    ``relative_path`` back to the canonical ``relative_path``.  During graph
    assembly this is used to translate import-map ``files`` keys into the
    same identifiers that ``build_nodes_from_scan`` uses, preventing
    edge-source mismatches.
    """
    canonical_map: Dict[str, str] = {}
    for scan_data in scan_data_list:
        for entry in scan_data.get("files", []):
            abs_path = entry.get("path", "")
            rel_path = entry.get("relative_path", "")
            if not rel_path:
                continue
            # Normalize separators
            abs_norm = abs_path.replace("\\", "/") if abs_path else ""
            rel_norm = rel_path.replace("\\", "/") if rel_path else ""
            if abs_norm:
                canonical_map[abs_norm] = rel_norm
            if rel_norm:
                canonical_map[rel_norm] = rel_norm
    return canonical_map


def _canonicalize_import_files(
    import_data: dict,
    canonical_map: Dict[str, str],
    project_root: str = "",
) -> dict:
    """Return a copy of *import_data* with its ``files`` keys rewritten to
    canonical relative_paths when a scan record matches.

    Resolution order for each key:
    1. Exact match in *canonical_map* (built from scan records) → use the
       mapped relative_path.
    2. If key is an absolute path that starts with *project_root* from scan
       data → derive a relative path.
    3. Otherwise preserve the key, normalizing backslashes to forward slashes.
    """
    files_map = import_data.get("files", {})
    new_files: Dict[str, dict] = {}

    norm_root = project_root.replace("\\", "/").rstrip("/") + "/" if project_root else ""

    for key, value in files_map.items():
        key_norm = key.replace("\\", "/")

        # 1. Exact match in canonical map (built from scan file records)
        if key_norm in canonical_map:
            new_key = canonical_map[key_norm]
            new_files[new_key] = value
            continue

        # 2. Absolute key inside project_root — derive relative path
        if key_norm.startswith("/") and norm_root and key_norm.startswith(norm_root):
            new_key = key_norm[len(norm_root):]
            new_files[new_key] = value
            continue

        # 3. Preserve key as-is, normalized
        new_files[key_norm] = value

    # Return mutated copy of import_data
    result = dict(import_data)
    result["files"] = new_files
    return result


# ── Graph assembly ───────────────────────────────────────────────────────────

def count_orphans(nodes: List[dict], edges: List[dict]) -> int:
    """Count nodes not referenced as source or target in any edge."""
    if not nodes:
        return 0

    referenced: set[str] = set()
    for edge in edges:
        referenced.add(edge.get("source", ""))
        referenced.add(edge.get("target", ""))

    orphan_count = 0
    for node in nodes:
        if node.get("node_id", "") not in referenced:
            orphan_count += 1

    return orphan_count


def build_summary(
    nodes: List[dict],
    edges: List[dict],
    dedup_nodes: int,
    dedup_edges: int,
    orphans: int,
) -> dict:
    """Build the summary section of the output graph."""
    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "deduplicated_nodes": dedup_nodes,
        "deduplicated_edges": dedup_edges,
        "orphan_nodes": orphans,
    }


def auto_detect_input_type(data: dict) -> str:
    """Auto-detect whether input is a scan output or import map.

    Scan: files is a list (array of file records)
    Import map: files is a dict with imports/warnings per file
    """
    files_field = data.get("files")
    if isinstance(files_field, list):
        return "scan"
    elif isinstance(files_field, dict):
        return "import"
    return "unknown"


def assemble_graph(
    scans: List[dict],
    imports_list: Optional[List[dict]] = None,
) -> dict:
    """Full assembly pipeline.

    Build nodes from all scans and imports, build edges from imports,
    merge and deduplicate, validate with graph_schema.py, return graph dict.
    """
    # Build canonicalization map from scan records so import-map file keys
    # (which may use absolute paths from extract_imports.py) are translated
    # to the same relative_path identifiers that build_nodes_from_scan uses.
    canonical_map = _build_canonical_map(scans) if scans else {}

    # Derive project_root from the first scan that has it
    project_root = ""
    for scan in scans:
        pr = scan.get("project_root", "")
        if pr:
            project_root = pr
            break

    # Canonicalize import data against scan records before any further processing
    canonicalized_imports: List[dict] = []
    if imports_list:
        for imp in imports_list:
            canonicalized_imports.append(
                _canonicalize_import_files(imp, canonical_map, project_root)
            )

    # Build nodes and edges from scans
    scan_nodes_lists: List[List[dict]] = []
    for scan in scans:
        nodes = build_nodes_from_scan(scan)
        scan_nodes_lists.append(nodes)

    # Build nodes and edges from canonicalized imports
    import_nodes_lists: List[List[dict]] = []
    all_edges_lists: List[List[dict]] = []
    if canonicalized_imports:
        for imp in canonicalized_imports:
            import_nodes_lists.append(build_nodes_from_imports(imp))
            all_edges_lists.append(build_edges_from_imports(imp))

    # Merge all nodes (concatenate without dedup first)
    all_node_batches = scan_nodes_lists + import_nodes_lists
    all_nodes_concat: List[dict] = []
    for batch in all_node_batches:
        all_nodes_concat.extend(batch)

    # Merge all edges (concatenate without dedup first)
    all_edges_concat: List[dict] = []
    for batch in all_edges_lists:
        all_edges_concat.extend(batch)

    # Deduplicate to get counts and final lists
    final_nodes, dedup_node_count = deduplicate_nodes(all_nodes_concat)
    final_edges, dedup_edge_count = deduplicate_edges(all_edges_concat)

    # Count orphans
    orphans = count_orphans(final_nodes, final_edges)

    # Build summary
    summary = build_summary(
        final_nodes, final_edges,
        dedup_node_count, dedup_edge_count, orphans,
    )

    # Build graph output
    graph = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_files": [],  # Populated by main()
        "nodes": final_nodes,
        "edges": final_edges,
        "summary": summary,
    }

    # Validate with graph_schema (print issues to stderr but don't fail)
    validation = validate_graph(graph)
    if validation.get("issues"):
        for issue in validation["issues"]:
            print(f"Graph validation issue: {issue}", file=sys.stderr)
    if validation.get("warnings"):
        for warning in validation["warnings"]:
            print(f"Graph validation warning: {warning}", file=sys.stderr)

    return graph


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> int:
    """CLI entry point. Parse args, load inputs, assemble graph, output JSON."""
    parser = argparse.ArgumentParser(
        description="Merge batch scan/import outputs into a unified dependency graph."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Path(s) to input JSON files (scan outputs or import maps)",
    )
    parser.add_argument(
        "--output",
        help="Path to write output JSON (default: stdout)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    args = parser.parse_args()

    if not args.inputs:
        print("Error: no input files specified", file=sys.stderr)
        return 1

    # Load inputs
    try:
        all_data = load_batch_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded {len(all_data)} input files", file=sys.stderr)

    # Auto-detect input types and separate into scans vs imports
    scans: List[dict] = []
    imports_list: List[dict] = []
    for data, path in zip(all_data, args.inputs):
        input_type = auto_detect_input_type(data)
        if input_type == "scan":
            scans.append(data)
            if args.verbose:
                print(f"Detected scan: {path}", file=sys.stderr)
        elif input_type == "import":
            imports_list.append(data)
            if args.verbose:
                print(f"Detected import map: {path}", file=sys.stderr)
        else:
            print(f"Warning: unknown input type for {path}, skipping", file=sys.stderr)

    if not scans and not imports_list:
        print("Error: no valid scan or import inputs found", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Processing {len(scans)} scans, {len(imports_list)} import maps", file=sys.stderr)

    # Assemble graph
    graph = assemble_graph(scans, imports_list or None)
    graph["source_files"] = list(args.inputs)

    # Output
    json_str = json.dumps(graph, indent=2)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(json_str)
                fh.write("\n")
            if args.verbose:
                print(f"Written to {args.output}", file=sys.stderr)
        except OSError as exc:
            print(f"Error: could not write output: {exc}", file=sys.stderr)
            return 1
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
