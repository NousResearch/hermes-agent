"""``hermes workflows`` subcommand — import visual workflows from Hermes Studio.

Hermes Studio (https://github.com/EKKOLearnAI/hermes-studio) ships a visual
workflow canvas and exports portable workflow definitions in the
``hermes-studio.workflow`` envelope (format version 1).  This command imports
those exports into the Hermes Agent workflow blueprint format used by the
dashboard visual workflow builder: ``~/.hermes/workflows/<name>.json`` holding
``{"nodes": [...], "edges": [...]}``.

Node mapping
------------
Studio workflows are Agent-only today (their import validation rejects any
non-``agent`` node).  Each Studio agent node becomes a Hermes Agent
``agent`` node carrying the goal text (``input``), a combined
``provider/model`` string, and the selected skills (mapped into the node's
``context`` so a delegated worker knows which skills to apply):

* ``approvalRequired: true`` → an ``agent`` node followed by a ``gate`` node,
  with the agent's outgoing edges rerouted through the gate.
* ``orchestration.join: "any"`` → not representable in the Agent executor
  (which requires every upstream to finish); downgraded to an all-join with a
  warning.
* ``claude-code`` / ``codex`` nodes → imported as Hermes ``agent`` nodes with a
  warning: the Agent visual-workflow executor delegates through Hermes
  subagents and does not spawn external coding agents.

Edge mapping
------------
* Studio ``condition`` semantics are not supported by the Agent executor yet;
  the edge is kept as a plain dependency and a warning is emitted.
* Studio ``feedback`` (loop) edges are dropped with a warning — the Agent
  executor rejects cycles.

Start/end synthesis
-------------------
Studio graphs do not include ``start``/``end`` nodes; Agent blueprints do.
After conversion the importer adds a synthetic ``start`` node (feeding every
node with no incoming edges) and an ``end`` node (collecting every node with
no outgoing edges), matching the blueprint shape the dashboard builder saves.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

# ---------------------------------------------------------------------------
# Format constants (mirror Hermes Studio's portability service)
# ---------------------------------------------------------------------------

STUDIO_FORMAT = "hermes-studio.workflow"
STUDIO_VERSION = 1
MAX_NODES = 500
MAX_EDGES = 2000

# Studio rejects imports that carry credential-looking fields; mirror that.
_CREDENTIAL_KEYS = {
    "token", "accesstoken", "access_token", "refreshtoken", "refresh_token",
    "apikey", "api_key", "password", "secret", "clientsecret",
    "client_secret", "privatekey", "private_key", "authorization", "bearer",
    "cookie", "sessionid", "session_id",
}

_SAFE_NAME = re.compile(r"^[a-zA-Z0-9_.-]+$")


class WorkflowImportError(Exception):
    """Raised when a Studio export cannot be imported."""


def _check_credentials(value: Any, depth: int = 0) -> None:
    """Reject credential-looking fields anywhere in the envelope."""
    if depth > 20:
        raise WorkflowImportError("workflow export exceeds maximum depth 20")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _check_credentials(item, depth + 1)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in _CREDENTIAL_KEYS:
                raise WorkflowImportError(f"workflow export contains credential field: {key}")
            _check_credentials(child, depth + 1)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9-_]", "-", name.lower()).strip("-")
    return slug or "workflow"


def _validate_name(name: str) -> str:
    if not name or not name.strip():
        raise WorkflowImportError("workflow export name is required")
    # Mirror the dashboard web server's _get_workflow_path defense: reject
    # traversal markers in the raw name before any slugging.
    if ".." in name or "/" in name or "\\" in name:
        raise WorkflowImportError(
            f"invalid workflow name {name!r}: use letters, digits, '.', '_' or '-'"
        )
    # Match the dashboard builder's save behavior: slugify to a safe filename.
    slug = _slugify(name)
    if not _SAFE_NAME.match(slug):
        raise WorkflowImportError(
            f"invalid workflow name {name!r}: use letters, digits, '.', '_' or '-'"
        )
    return slug


def _topological_ok(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> bool:
    """Kahn's algorithm — True if the graph is acyclic."""
    ids = {str(n["id"]) for n in nodes}
    in_degree = {nid: 0 for nid in ids}
    adj: Dict[str, List[str]] = {nid: [] for nid in ids}
    for edge in edges:
        src, tgt = str(edge.get("source", "")), str(edge.get("target", ""))
        if src in ids and tgt in ids and tgt not in adj[src]:
            adj[src].append(tgt)
            in_degree[tgt] += 1
    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        curr = queue.popleft()
        visited += 1
        for nxt in adj[curr]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    return visited == len(ids)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _convert_node(
    node: Dict[str, Any],
    *,
    index: int,
    warnings: List[str],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Convert one Studio agent node into an Agent agent node (+ optional gate).

    Returns ``(agent_node, gate_node_or_None)``.  When a gate is returned the
    caller must route the agent node's outgoing edges through the gate.
    """
    node_id = str(node.get("id") or f"agent-{index + 1}")
    data = node.get("data") or {}
    title = str(data.get("title") or f"Agent {index + 1}")
    agent_kind = str(data.get("agent") or "hermes")
    prompt = str(data.get("input") or "")
    provider = str(data.get("provider") or "")
    model = str(data.get("model") or "")
    if provider and model:
        model_ref = f"{provider}/{model}"
    elif model:
        model_ref = model
    else:
        model_ref = "default"

    if agent_kind in ("claude-code", "codex"):
        warnings.append(
            f"node {node_id!r} uses coding agent {agent_kind!r}; "
            "importing as a Hermes agent node (the Agent executor delegates "
            "through Hermes subagents)"
        )

    skills = data.get("skills") or []
    context_parts: List[str] = []
    if isinstance(skills, list) and skills:
        context_parts.append("Selected skills: " + ", ".join(str(s) for s in skills))
    if agent_kind not in ("hermes", ""):
        context_parts.append(f"Original agent backend: {agent_kind}")
    context = "\n".join(context_parts) if context_parts else ""

    agent_node = {
        "id": node_id,
        "type": "agent",
        "position": dict(node.get("position") or {"x": 250, "y": 200 + index * 40}),
        "data": {
            "label": title,
            "model": model_ref,
            "toolsets": [],
            "prompt": prompt,
            "context": context,
            "status": "idle",
        },
    }

    gate_node: Optional[Dict[str, Any]] = None
    if data.get("approvalRequired") is True:
        gate_node = {
            "id": f"{node_id}-gate",
            "type": "gate",
            "position": {
                "x": agent_node["position"].get("x", 250),
                "y": agent_node["position"].get("y", 200) + 80,
            },
            "data": {"label": f"Approve: {title}", "prompt": prompt, "status": "idle"},
        }
    return agent_node, gate_node


def convert_studio_envelope(
    envelope: Dict[str, Any],
    *,
    name_override: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Convert a parsed ``hermes-studio.workflow`` envelope to an Agent blueprint.

    Returns ``(blueprint, warnings)`` where blueprint is
    ``{"name": ..., "nodes": [...], "edges": [...]}``.
    """
    if not isinstance(envelope, dict):
        raise WorkflowImportError("workflow export envelope is required")
    if envelope.get("format") != STUDIO_FORMAT:
        raise WorkflowImportError(
            f"unsupported workflow export format {envelope.get('format')!r} "
            f"(expected {STUDIO_FORMAT!r})"
        )
    if envelope.get("version") != STUDIO_VERSION:
        raise WorkflowImportError(
            f"unsupported workflow export version {envelope.get('version')!r} "
            f"(expected {STUDIO_VERSION})"
        )
    _check_credentials(envelope)

    definition = envelope.get("definition")
    if not isinstance(definition, dict):
        raise WorkflowImportError("workflow export definition is required")

    name = _validate_name(name_override or definition.get("name"))
    nodes = definition.get("nodes")
    edges = definition.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise WorkflowImportError("workflow export nodes and edges must be arrays")
    if len(nodes) > MAX_NODES:
        raise WorkflowImportError(f"workflow export exceeds {MAX_NODES} nodes")
    if len(edges) > MAX_EDGES:
        raise WorkflowImportError(f"workflow export exceeds {MAX_EDGES} edges")

    warnings: List[str] = []
    converted_nodes: List[Dict[str, Any]] = []
    node_ids: List[str] = []
    gate_for: Dict[str, str] = {}  # source agent id -> gate node id

    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or node.get("type") != "agent":
            raise WorkflowImportError(
                f"node {index} is not an 'agent' node; Studio imports are agent-only"
            )
        agent_node, gate_node = _convert_node(node, index=index, warnings=warnings)
        node_id = agent_node["id"]
        if node_id in ("start", "end"):
            raise WorkflowImportError(
                f"node id {node_id!r} is reserved for synthetic Agent nodes"
            )
        converted_nodes.append(agent_node)
        node_ids.append(node_id)
        if gate_node is not None:
            converted_nodes.append(gate_node)
            gate_for[node_id] = gate_node["id"]

    # Studio edge conditions/loops are not representable in the Agent executor.
    kept_edges: List[Dict[str, Any]] = []
    dropped_feedback = 0
    dropped_conditions = 0
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src, tgt = str(edge.get("source", "")), str(edge.get("target", ""))
        edge_data = edge.get("data")
        orchestration = edge_data.get("orchestration") if isinstance(edge_data, dict) else None
        if isinstance(orchestration, dict):
            if orchestration.get("condition") is not None:
                dropped_conditions += 1
                warnings.append(
                    f"edge {src} -> {tgt} has a condition; conditions are not "
                    "supported by the Agent executor yet — keeping the plain edge"
                )
            if orchestration.get("feedback"):
                dropped_feedback += 1
                warnings.append(
                    f"edge {src} -> {tgt} is a feedback/loop edge; loops are not "
                    "supported by the Agent executor — dropping the edge"
                )
                continue
        kept_edges.append({"source": src, "target": tgt, "id": str(edge.get("id") or f"e{len(kept_edges)}")})

    # Reroute each gated agent node's outgoing edges through its gate.
    final_edges: List[Dict[str, Any]] = []
    for edge in kept_edges:
        src, tgt = edge["source"], edge["target"]
        if src in gate_for:
            gate_id = gate_for[src]
            if not any(e["source"] == gate_id and e["target"] == tgt for e in final_edges):
                final_edges.append({"source": gate_id, "target": tgt, "id": f"{gate_id}-{tgt}"})
            if not any(e["source"] == src and e["target"] == gate_id for e in final_edges):
                final_edges.append({"source": src, "target": gate_id, "id": f"{src}-{gate_id}"})
        else:
            final_edges.append(edge)

    # Warn about any-join downgrades (Studio may produce them on nodes).
    for node in nodes:
        data = node.get("data") or {}
        orch = data.get("orchestration") or {}
        if isinstance(orch, dict) and orch.get("join") == "any":
            warnings.append(
                f"node {node.get('id')!r} uses join='any'; the Agent executor "
                "requires all upstreams — downgraded to an all-join"
            )

    # Cycle check (the executor rejects cyclic graphs).
    if not _topological_ok(converted_nodes, final_edges):
        raise WorkflowImportError(
            "converted workflow contains a cycle; the Agent executor rejects cyclic graphs"
        )

    # Synthesize start/end.
    ids = {str(n["id"]) for n in converted_nodes}
    incoming: Dict[str, int] = {nid: 0 for nid in ids}
    outgoing: Dict[str, int] = {nid: 0 for nid in ids}
    for edge in final_edges:
        if edge["source"] in ids:
            outgoing[edge["source"]] += 1
        if edge["target"] in ids:
            incoming[edge["target"]] += 1

    start_id, end_id = "start", "end"
    start_node = {"id": start_id, "type": "start", "position": {"x": 250, "y": 50}, "data": {}}
    end_node = {"id": end_id, "type": "end", "position": {"x": 250, "y": 450}, "data": {}}
    blueprint_nodes: List[Dict[str, Any]] = [start_node]
    blueprint_nodes.extend(converted_nodes)
    blueprint_nodes.append(end_node)

    blueprint_edges = list(final_edges)
    for nid in ids:
        if incoming[nid] == 0:
            blueprint_edges.append({"source": start_id, "target": nid, "id": f"start-{nid}"})
        if outgoing[nid] == 0:
            blueprint_edges.append({"source": nid, "target": end_id, "id": f"{nid}-end"})

    return {"name": name, "nodes": blueprint_nodes, "edges": blueprint_edges}, warnings


def workflows_dir() -> Path:
    return get_hermes_home() / "workflows"


def import_studio_file(
    source: Path,
    *,
    name: Optional[str] = None,
    out_dir: Optional[Path] = None,
    print_only: bool = False,
) -> Tuple[Path, int, int, List[str]]:
    """Import a Studio export file. Returns (out_path, nodes, edges, warnings)."""
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkflowImportError(f"cannot read {source}: {exc}")
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkflowImportError(f"{source} is not valid JSON: {exc}")

    blueprint, warnings = convert_studio_envelope(envelope, name_override=name)

    target_dir = out_dir or workflows_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{blueprint['name']}.json"
    if not print_only and out_path.exists():
        raise WorkflowImportError(
            f"{out_path} already exists; remove it or pass a different --name"
        )
    payload = {"nodes": blueprint["nodes"], "edges": blueprint["edges"]}
    if print_only:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return out_path, len(blueprint["nodes"]), len(blueprint["edges"]), warnings


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Attach the ``workflows`` subcommand (and sub-actions) to ``subparsers``."""
    parser = parent_subparsers.add_parser(
        "workflows",
        help="Import and inspect visual workflow blueprints",
        description=(
            "Import visual workflows exported from Hermes Studio "
            "(hermes-studio.workflow envelope) into the Hermes Agent workflow "
            "blueprint store at ~/.hermes/workflows/, and list/inspect them."
        ),
    )
    subs = parser.add_subparsers(dest="workflows_command")

    p_import = subs.add_parser(
        "import-studio",
        help="Import a Hermes Studio workflow export",
    )
    p_import.add_argument("file", help="Path to a hermes-studio.workflow export (.json)")
    p_import.add_argument("--name", help="Blueprint name override (default: export name)")
    p_import.add_argument(
        "--out",
        type=Path,
        help="Output directory (default: ~/.hermes/workflows/)",
    )
    p_import.add_argument(
        "--print",
        action="store_true",
        help="Print the converted blueprint to stdout instead of writing a file",
    )

    p_list = subs.add_parser("list", help="List imported workflow blueprints")
    p_list.add_argument("--dir", type=Path, help="Blueprint directory (default: ~/.hermes/workflows/)")

    p_show = subs.add_parser("show", help="Show an imported workflow blueprint")
    p_show.add_argument("name", help="Blueprint name (without .json)")
    p_show.add_argument("--dir", type=Path, help="Blueprint directory (default: ~/.hermes/workflows/)")

    return parser


def workflows_command(args: argparse.Namespace) -> int:
    """Dispatch ``hermes workflows`` subcommands."""
    sub = getattr(args, "workflows_command", None)

    if sub == "import-studio":
        source = Path(args.file)
        if not source.exists():
            print(f"error: {source} does not exist", file=sys.stderr)
            return 1
        try:
            out_path, n_nodes, n_edges, warnings = import_studio_file(
                source,
                name=args.name,
                out_dir=args.out,
                print_only=args.print,
            )
        except WorkflowImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if not args.print:
            print(f"imported {n_nodes} nodes / {n_edges} edges -> {out_path}")
        return 0

    if sub == "list":
        directory = args.dir or workflows_dir()
        if not directory.exists():
            print("no workflows imported yet (~/.hermes/workflows/ is empty)")
            return 0
        files = sorted(directory.glob("*.json"))
        if not files:
            print("no workflows imported yet (~/.hermes/workflows/ is empty)")
            return 0
        for path in files:
            print(path.stem)
        return 0

    if sub == "show":
        directory = args.dir or workflows_dir()
        path = directory / f"{args.name}.json"
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 1
        try:
            print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
            return 1
        return 0

    print(
        "usage: hermes workflows <import-studio|list|show> ...\n"
        "  import-studio <file.json>   Import a Hermes Studio workflow export\n"
        "  list                        List imported workflow blueprints\n"
        "  show <name>                 Show an imported workflow blueprint",
        file=sys.stderr,
    )
    return 1
