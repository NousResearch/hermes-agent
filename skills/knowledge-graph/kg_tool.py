#!/usr/bin/env python3
"""CLI for the Hermes knowledge graph skill.

Self-contained: no external dependencies beyond the Python standard library.
Persists the graph as a JSON file (default: ~/.hermes/knowledge_graph.json).

Usage:
    kg_tool.py add --subject S --subject-type T --predicate P --object O --object-type T2
    kg_tool.py context --node NODE_ID [--hops K]
    kg_tool.py clusters
    kg_tool.py patterns
    kg_tool.py list [--type ENTITY_TYPE]
    kg_tool.py path --from FROM_ID --to TO_ID
    kg_tool.py stats
    kg_tool.py export
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Storage ──────────────────────────────────────────────────────────────────

DEFAULT_KG_PATH = Path.home() / ".hermes" / "knowledge_graph.json"


def _kg_path() -> Path:
    return Path(os.environ.get("KG_PATH", DEFAULT_KG_PATH))


def _load() -> dict:
    path = _kg_path()
    if not path.exists():
        return {"nodes": {}, "triples": []}
    return json.loads(path.read_text())


def _save(data: dict) -> None:
    path = _kg_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ── Helpers ──────────────────────────────────────────────────────────────────

VALID_TYPES = {
    "person", "project", "document", "workflow", "claim",
    "risk", "tool", "policy", "repository", "topic", "organization",
}


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", text.lower().strip())[:64]


def _node_id(entity_type: str, label: str) -> str:
    return f"{entity_type}:{_slugify(label)}"


def _ensure_node(
    data: dict,
    entity_type: str,
    label: str,
    summary: str = "",
) -> str:
    nid = _node_id(entity_type, label)
    if nid not in data["nodes"]:
        data["nodes"][nid] = {
            "id": nid,
            "label": label,
            "entity_type": entity_type,
            "summary": summary,
            "created_at": datetime.utcnow().isoformat(),
        }
    elif summary and not data["nodes"][nid].get("summary"):
        data["nodes"][nid]["summary"] = summary
    return nid


def _triple_id(sub: str, pred: str, obj: str) -> str:
    return f"{sub}|{pred}|{obj}"


# ── Commands ──────────────────────────────────────────────────────────────────


def cmd_add(args: argparse.Namespace) -> None:
    if args.subject_type not in VALID_TYPES:
        sys.exit(f"Invalid --subject-type '{args.subject_type}'. Choose from: {sorted(VALID_TYPES)}")
    if args.object_type not in VALID_TYPES:
        sys.exit(f"Invalid --object-type '{args.object_type}'. Choose from: {sorted(VALID_TYPES)}")

    data = _load()
    sub_id = _ensure_node(data, args.subject_type, args.subject, args.subject_summary or "")
    obj_id = _ensure_node(data, args.object_type, args.object, args.object_summary or "")
    tid = _triple_id(sub_id, args.predicate, obj_id)

    existing_ids = {t["id"] for t in data["triples"]}
    if tid in existing_ids:
        print(f"Triple already exists: {sub_id} -> {args.predicate} -> {obj_id}")
        return

    triple: dict = {
        "id": tid,
        "subject_id": sub_id,
        "predicate": args.predicate,
        "object_id": obj_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    if args.source or args.confidence is not None:
        triple["evidence"] = {
            "source": args.source or "",
            "confidence": float(args.confidence) if args.confidence is not None else 1.0,
            "timestamp": datetime.utcnow().isoformat(),
        }
    data["triples"].append(triple)
    _save(data)
    print(f"Added: {args.subject} -> {args.predicate} -> {args.object}")


def cmd_context(args: argparse.Namespace) -> None:
    data = _load()
    node_id = args.node
    node = data["nodes"].get(node_id)
    if node is None:
        print(f"No knowledge available about '{node_id}'.")
        return

    k = args.hops
    triples = data["triples"]

    # Build adjacency for k-hop BFS
    outgoing: dict[str, list[dict]] = defaultdict(list)
    incoming: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        outgoing[t["subject_id"]].append(t)
        incoming[t["object_id"]].append(t)

    def k_hop(start: str, hops: int) -> set[str]:
        visited = {start}
        frontier = {start}
        for _ in range(hops):
            nxt: set[str] = set()
            for nid in frontier:
                for t in outgoing.get(nid, []):
                    if t["object_id"] not in visited:
                        nxt.add(t["object_id"])
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        visited.discard(start)
        return visited

    neighbors = k_hop(node_id, k)
    matching = outgoing.get(node_id, []) + incoming.get(node_id, [])

    lines = [
        f"Entity: {node['label']} [{node['entity_type']}]",
        f"Summary: {node.get('summary') or '(no summary)'}",
    ]
    if matching:
        lines.append("Relationships:")
        for t in matching:
            subj = data["nodes"].get(t["subject_id"], {})
            obj = data["nodes"].get(t["object_id"], {})
            conf_str = ""
            if ev := t.get("evidence"):
                conf_str = f" (confidence={ev['confidence']:.2f})"
            lines.append(
                f"  {subj.get('label', t['subject_id'])} -> {t['predicate']} "
                f"-> {obj.get('label', t['object_id'])}{conf_str}"
            )
    if neighbors:
        lines.append("Nearby entities:")
        for nid in list(neighbors)[:10]:
            n = data["nodes"].get(nid, {})
            lines.append(f"  - {n.get('label', nid)} [{n.get('entity_type', '?')}]")

    print("\n".join(lines))


def cmd_clusters(args: argparse.Namespace) -> None:
    data = _load()
    nodes = data["nodes"]
    triples = data["triples"]
    if not nodes:
        print("Graph is empty.")
        return

    # Undirected BFS over all nodes
    adj: dict[str, set[str]] = defaultdict(set)
    for t in triples:
        adj[t["subject_id"]].add(t["object_id"])
        adj[t["object_id"]].add(t["subject_id"])

    visited: set[str] = set()
    components: list[list[str]] = []
    for nid in nodes:
        if nid not in visited:
            component: list[str] = []
            queue = [nid]
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                component.append(cur)
                for nb in adj.get(cur, set()):
                    if nb not in visited:
                        queue.append(nb)
            components.append(component)

    components.sort(key=len, reverse=True)
    print(f"Found {len(components)} cluster(s):\n")
    for i, comp in enumerate(components, 1):
        type_counts: dict[str, int] = defaultdict(int)
        for nid in comp:
            type_counts[nodes[nid]["entity_type"]] += 1
        dominant = max(type_counts, key=type_counts.__getitem__)
        internal = sum(
            1 for t in triples
            if t["subject_id"] in comp and t["object_id"] in comp
        )
        print(f"  Cluster {i}: {len(comp)} nodes, {internal} internal triples, dominant type: {dominant}")
        for nid in comp:
            n = nodes[nid]
            print(f"    - {n['label']} [{n['entity_type']}]")
        print()


def cmd_patterns(args: argparse.Namespace) -> None:
    data = _load()
    nodes = data["nodes"]
    triples = data["triples"]
    if not triples:
        print("No triples recorded yet.")
        return

    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for t in triples:
        stype = nodes.get(t["subject_id"], {}).get("entity_type", "?")
        otype = nodes.get(t["object_id"], {}).get("entity_type", "?")
        counts[(t["predicate"], stype, otype)] += 1

    patterns = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Relationship patterns ({len(patterns)}):\n")
    for (pred, stype, otype), count in patterns:
        print(f"  {stype} --[{pred}]--> {otype}  (x{count})")


def cmd_list(args: argparse.Namespace) -> None:
    data = _load()
    nodes = data["nodes"].values()
    if args.type:
        nodes = [n for n in nodes if n["entity_type"] == args.type]
    for n in sorted(nodes, key=lambda x: x["label"]):
        summary = f"  {n['summary']}" if n.get("summary") else ""
        print(f"  {n['id']}  ({n['entity_type']}){summary}")


def cmd_path(args: argparse.Namespace) -> None:
    data = _load()
    triples = data["triples"]
    nodes = data["nodes"]
    src, tgt = args.from_id, args.to_id

    if src not in nodes:
        sys.exit(f"Node not found: {src}")
    if tgt not in nodes:
        sys.exit(f"Node not found: {tgt}")
    if src == tgt:
        print(f"Path: {src}")
        return

    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for t in triples:
        outgoing[t["subject_id"]].append((t["predicate"], t["object_id"]))

    queue: list[list[str]] = [[src]]
    visited: set[str] = {src}
    while queue:
        path = queue.pop(0)
        if len(path) > 6:
            break
        current = path[-1]
        for pred, nxt in outgoing.get(current, []):
            if nxt == tgt:
                full_path = path + [nxt]
                labels = [nodes.get(n, {}).get("label", n) for n in full_path]
                print(" -> ".join(labels))
                return
            if nxt not in visited:
                visited.add(nxt)
                queue.append(path + [nxt])
    print(f"No directed path found from '{src}' to '{tgt}' within 5 hops.")


def cmd_stats(args: argparse.Namespace) -> None:
    data = _load()
    nodes = data["nodes"]
    triples = data["triples"]
    type_counts: dict[str, int] = defaultdict(int)
    for n in nodes.values():
        type_counts[n["entity_type"]] += 1
    pred_counts: dict[str, int] = defaultdict(int)
    for t in triples:
        pred_counts[t["predicate"]] += 1

    print(f"Nodes   : {len(nodes)}")
    print(f"Triples : {len(triples)}")
    print(f"File    : {_kg_path()}")
    if type_counts:
        print("\nEntities by type:")
        for etype, count in sorted(type_counts.items()):
            print(f"  {etype}: {count}")
    if pred_counts:
        print("\nPredicates:")
        for pred, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
            print(f"  {pred}: {count}")


def cmd_export(args: argparse.Namespace) -> None:
    print(json.dumps(_load(), indent=2))


# ── CLI wiring ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes knowledge graph CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = sub.add_parser("add", help="Record a triple")
    p_add.add_argument("--subject", required=True)
    p_add.add_argument("--subject-type", required=True, dest="subject_type")
    p_add.add_argument("--subject-summary", default="", dest="subject_summary")
    p_add.add_argument("--predicate", required=True)
    p_add.add_argument("--object", required=True)
    p_add.add_argument("--object-type", required=True, dest="object_type")
    p_add.add_argument("--object-summary", default="", dest="object_summary")
    p_add.add_argument("--source", default="")
    p_add.add_argument("--confidence", type=float, default=None)

    # context
    p_ctx = sub.add_parser("context", help="Bounded AI context for a node")
    p_ctx.add_argument("--node", required=True)
    p_ctx.add_argument("--hops", type=int, default=2)

    # clusters
    sub.add_parser("clusters", help="Detect knowledge clusters")

    # patterns
    sub.add_parser("patterns", help="Mine relationship patterns")

    # list
    p_list = sub.add_parser("list", help="List entities")
    p_list.add_argument("--type", default=None)

    # path
    p_path = sub.add_parser("path", help="Shortest path between two entities")
    p_path.add_argument("--from", required=True, dest="from_id")
    p_path.add_argument("--to", required=True, dest="to_id")

    # stats
    sub.add_parser("stats", help="Graph statistics")

    # export
    sub.add_parser("export", help="Export graph as JSON")

    args = parser.parse_args()
    {
        "add": cmd_add,
        "context": cmd_context,
        "clusters": cmd_clusters,
        "patterns": cmd_patterns,
        "list": cmd_list,
        "path": cmd_path,
        "stats": cmd_stats,
        "export": cmd_export,
    }[args.command](args)


if __name__ == "__main__":
    main()
