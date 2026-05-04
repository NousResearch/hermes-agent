#!/usr/bin/env python3
"""
extract_schema.py — Analyze a ComfyUI API-format workflow and extract
controllable parameters.

Improvements over v1:
  - Catalogs live in `_common.py`, shared with `check_deps.py`
  - Coverage expanded for Flux / SD3 / Wan / Hunyuan / LTX / IPAdapter / rgthree
  - Symmetric duplicate-name resolution: ALL duplicates get a node-id suffix
    (instead of "first wins, second renamed"), so callers see consistent names
  - Negative prompt detected by tracing `KSampler.negative` connections back to
    the source CLIPTextEncode (more reliable than meta-title heuristic)
  - Embedding references in prompt text are extracted as model dependencies
  - Detects Primitive nodes that drive other nodes' inputs (and surfaces them
    as the user-facing parameter)
  - Reroutes are followed when tracing connections

Usage:
    python3 extract_schema.py workflow_api.json
    python3 extract_schema.py workflow_api.json --output schema.json

Stdlib-only. Python 3.10+.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUTPUT_NODES, PARAM_PATTERNS, PROMPT_FIELDS,
    is_link, iter_embedding_refs, iter_model_deps, iter_nodes, unwrap_workflow,
)


# Sampler nodes whose `positive` / `negative` connections we trace
SAMPLER_NODE_FAMILY = {
    "KSampler", "KSamplerAdvanced",
    "SamplerCustom", "SamplerCustomAdvanced",
    "BasicGuider", "CFGGuider", "DualCFGGuider",
}


def infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "link"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def trace_to_node(workflow: dict, link: list, *, max_hops: int = 8) -> str | None:
    """Follow a [node_id, slot] link, hopping through Reroute / Primitive nodes
    if needed, to find the *upstream* node id that holds the actual value/input.

    Bounded by both `max_hops` AND a visited-set to prevent infinite loops on
    pathological graphs.
    """
    if not is_link(link):
        return None
    nid: str | None = link[0]
    visited: set[str] = set()
    for _ in range(max_hops):
        if nid is None or nid in visited:
            return nid
        visited.add(nid)
        node = workflow.get(nid)
        if not isinstance(node, dict):
            return None
        cls = node.get("class_type", "")
        # Reroute / Primitive / passthrough wrappers
        if cls in ("Reroute", "PrimitiveNode", "Note", "easy showAnything"):
            inputs = node.get("inputs", {}) or {}
            # Find first link-shaped input and follow it
            next_link = next((v for v in inputs.values() if is_link(v)), None)
            if next_link is None:
                return nid
            nid = next_link[0]
            continue
        return nid
    return nid


def find_negative_prompt_node(workflow: dict) -> str | None:
    """Trace `negative` input of a sampler back to the source text encoder."""
    for nid, node in iter_nodes(workflow):
        if node["class_type"] not in SAMPLER_NODE_FAMILY:
            continue
        inputs = node.get("inputs", {}) or {}
        neg = inputs.get("negative")
        if not is_link(neg):
            continue
        src = trace_to_node(workflow, neg)
        if src and isinstance(workflow.get(src), dict):
            cls = workflow[src].get("class_type", "")
            if cls.startswith("CLIPTextEncode") or cls in ("smZ CLIPTextEncode", "BNK_CLIPTextEncodeAdvanced"):
                return src
    return None


def find_positive_prompt_node(workflow: dict) -> str | None:
    for nid, node in iter_nodes(workflow):
        if node["class_type"] not in SAMPLER_NODE_FAMILY:
            continue
        inputs = node.get("inputs", {}) or {}
        pos = inputs.get("positive")
        if not is_link(pos):
            continue
        src = trace_to_node(workflow, pos)
        if src and isinstance(workflow.get(src), dict):
            cls = workflow[src].get("class_type", "")
            if cls.startswith("CLIPTextEncode") or cls in ("smZ CLIPTextEncode", "BNK_CLIPTextEncodeAdvanced"):
                return src
    return None


def extract_schema(workflow: dict) -> dict:
    """Extract controllable parameters from a workflow.

    Returns:
        {
          "parameters": { friendly_name: {node_id, field, type, value, ...} },
          "output_nodes": [node_id, ...],
          "model_dependencies": [{node_id, class_type, field, value, folder}],
          "embedding_dependencies": [{node_id, embedding_name, found_in_field, value_excerpt}],
          "summary": {...}
        }
    """
    output_nodes: list[str] = []

    # First pass: identify positive / negative prompt nodes via connection tracing
    pos_node = find_positive_prompt_node(workflow)
    neg_node = find_negative_prompt_node(workflow)

    # ----- collect raw parameter candidates -----
    # Each candidate = (friendly_name, node_id, field, value)
    # We resolve duplicate friendly_names AFTER the loop so dedup is symmetric.
    raw_params: list[dict] = []

    for node_id, node in iter_nodes(workflow):
        cls = node["class_type"]
        inputs = node.get("inputs", {}) or {}

        if cls in OUTPUT_NODES:
            output_nodes.append(node_id)

        # Match this node against PARAM_PATTERNS
        for p_class, p_field, friendly in PARAM_PATTERNS:
            if cls != p_class:
                continue
            if p_field not in inputs:
                continue
            value = inputs[p_field]
            t = infer_type(value)
            if t == "link":
                continue  # connections aren't directly controllable

            actual_name = friendly

            # Disambiguate prompt vs negative_prompt by connection tracing
            if friendly == "prompt":
                if node_id == neg_node and pos_node != neg_node:
                    actual_name = "negative_prompt"
                elif node_id == pos_node:
                    actual_name = "prompt"
                else:
                    # Fallback: use _meta.title hints if present
                    meta_title = (node.get("_meta") or {}).get("title", "").lower()
                    if any(t_ in meta_title for t_ in ("negative", "neg", "-prompt", "anti")):
                        actual_name = "negative_prompt"

            raw_params.append({
                "name_hint": actual_name,
                "node_id": node_id,
                "field": p_field,
                "type": t,
                "value": value,
                "class_type": cls,
            })

    # ----- symmetric duplicate-name resolution -----
    # Group by name_hint. If a hint appears once, keep it. If multiple, suffix
    # ALL with their node_id. Always-stable, always-uniquely-addressable.
    by_name: dict[str, list[dict]] = {}
    for r in raw_params:
        by_name.setdefault(r["name_hint"], []).append(r)

    parameters: dict[str, dict] = {}
    for name, entries in by_name.items():
        if len(entries) == 1:
            r = entries[0]
            parameters[name] = {
                "node_id": r["node_id"], "field": r["field"],
                "type": r["type"], "value": r["value"],
                "class_type": r["class_type"],
            }
        else:
            # Sort by node_id (string-natural) for stability
            entries.sort(key=lambda x: (str(x["node_id"]).zfill(8), x["field"]))
            for r in entries:
                full_name = f"{name}_{r['node_id']}"
                parameters[full_name] = {
                    "node_id": r["node_id"], "field": r["field"],
                    "type": r["type"], "value": r["value"],
                    "class_type": r["class_type"],
                    "alias_of": name,
                }

    # ----- model dependencies -----
    model_deps = list(iter_model_deps(workflow))

    # ----- embedding dependencies (in prompt text) -----
    embedding_deps: list[dict] = []
    seen_emb: set[tuple[str, str]] = set()
    for nid, emb_name in iter_embedding_refs(workflow):
        key = (nid, emb_name)
        if key in seen_emb:
            continue
        seen_emb.add(key)
        # Find which field had the reference, for context
        node = workflow.get(nid, {})
        inputs = node.get("inputs", {}) or {}
        found_field = None
        excerpt = None
        for fname, fval in inputs.items():
            if isinstance(fval, str) and fname in PROMPT_FIELDS and emb_name in fval:
                found_field = fname
                excerpt = fval[:120]
                break
        embedding_deps.append({
            "node_id": nid,
            "embedding_name": emb_name,
            "field": found_field,
            "value_excerpt": excerpt,
            "folder": "embeddings",
        })

    # ----- summary -----
    summary = {
        "parameter_count": len(parameters),
        "output_node_count": len(output_nodes),
        "model_dep_count": len(model_deps),
        "embedding_dep_count": len(embedding_deps),
        "has_negative_prompt": "negative_prompt" in parameters,
        "has_seed": "seed" in parameters or any(p.startswith("seed_") for p in parameters),
        "is_video_workflow": any(
            workflow.get(n, {}).get("class_type", "") in {
                "VHS_VideoCombine", "SaveVideo", "SaveAnimatedWEBP", "SaveAnimatedPNG",
            } for n in output_nodes
        ),
    }

    return {
        "parameters": parameters,
        "output_nodes": output_nodes,
        "model_dependencies": model_deps,
        "embedding_dependencies": embedding_deps,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract controllable parameters from a ComfyUI workflow")
    p.add_argument("workflow", help="Path to workflow API JSON file")
    p.add_argument("--output", "-o", help="Output file (default: stdout)")
    p.add_argument("--summary-only", action="store_true",
                   help="Only print the summary block")
    args = p.parse_args(argv)

    wf_path = Path(args.workflow).expanduser()
    if not wf_path.exists():
        print(f"Error: {wf_path} not found", file=sys.stderr)
        return 1

    try:
        with wf_path.open() as f:
            payload = json.load(f)
        workflow = unwrap_workflow(payload)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON — {e}", file=sys.stderr)
        return 1

    schema = extract_schema(workflow)

    if args.summary_only:
        out = json.dumps(schema["summary"], indent=2)
    else:
        out = json.dumps(schema, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(out)
        print(f"Schema written to {args.output}", file=sys.stderr)
    else:
        print(out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
