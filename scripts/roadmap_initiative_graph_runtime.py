#!/usr/bin/env python3
"""Build a unified initiative/dependency/next-action graph for roadmap execution."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_CORE_QUEUE_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "core_roadmap_execution_queue.json"
DEFAULT_CORE_LAYER_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "core_roadmap_queue_layer.json"
DEFAULT_EXPANDED_QUEUE_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "true_expanded_roadmap_queue_layer.json"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest"
DEFAULT_GRAPH_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "roadmap_initiative_graph_latest.json"
DEFAULT_OPERATOR_BRIEF_PATH = DEFAULT_OUTPUT_DIR / "roadmap_initiative_graph_operator_brief.md"


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate a roadmap initiative graph snapshot")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    ap.add_argument("--core-queue-path", default=str(DEFAULT_CORE_QUEUE_PATH))
    ap.add_argument("--core-layer-path", default=str(DEFAULT_CORE_LAYER_PATH))
    ap.add_argument("--expanded-queue-path", default=str(DEFAULT_EXPANDED_QUEUE_PATH))
    ap.add_argument("--graph-output-path", default=str(DEFAULT_GRAPH_OUTPUT_PATH))
    ap.add_argument("--operator-brief-path", default=str(DEFAULT_OPERATOR_BRIEF_PATH))
    ap.add_argument("--json", action="store_true", help="Emit output manifest JSON")
    return ap.parse_args(argv)


def normalize_state(state: str) -> str:
    s = (state or "").strip().upper()
    if not s:
        return "unknown"
    if s == "DONE":
        return "done"
    if s in {"RUNNING", "IN_PROGRESS"}:
        return "running"
    if s.startswith("READY"):
        if "PENDING" in s:
            return "dependency_blocked"
        return "ready"
    if "BLOCKED" in s:
        return "dependency_blocked"
    if s.startswith("QUEUED"):
        return "queued"
    return "unknown"


def _core_id(value: Any) -> str:
    return f"CORE-{str(value).strip()}"


def _state_of(initiative_index: Dict[str, Dict[str, Any]], initiative_id: str) -> str:
    row = initiative_index.get(initiative_id) or {}
    return str(row.get("normalized_state") or "unknown")


def _unresolved_dependencies(
    initiative_index: Dict[str, Dict[str, Any]],
    dependencies: Iterable[str],
) -> Tuple[Dict[str, str], List[str]]:
    dep_states: Dict[str, str] = {}
    unresolved: List[str] = []
    for dep in dependencies:
        dep_id = str(dep or "").strip()
        if not dep_id:
            continue
        dep_state = _state_of(initiative_index, dep_id)
        dep_states[dep_id] = dep_state
        if dep_state != "done":
            unresolved.append(dep_id)
    return dep_states, unresolved


def _next_action(
    *,
    normalized_state: str,
    unresolved_dependencies: List[str],
) -> Dict[str, Any]:
    if normalized_state == "done":
        return {
            "kind": "none_done",
            "reason": "initiative already closed",
            "command_hint": None,
        }
    if normalized_state == "ready":
        return {
            "kind": "execute_now",
            "reason": "dependencies resolved and queue state is ready",
            "command_hint": "launch bounded execution slice and commit evidence on completion",
        }
    if normalized_state == "running":
        return {
            "kind": "monitor_running",
            "reason": "initiative is in progress",
            "command_hint": "monitor runtime evidence and fail closed on stale execution",
        }
    if normalized_state == "dependency_blocked":
        if unresolved_dependencies:
            return {
                "kind": "wait_dependencies",
                "reason": "blocked until dependencies reach done",
                "command_hint": f"resolve dependencies first: {', '.join(unresolved_dependencies)}",
            }
        return {
            "kind": "reconcile_blocked_state",
            "reason": "blocked state without unresolved dependencies (possible stale state)",
            "command_hint": "reconcile queue truth projection before launching",
        }
    if normalized_state == "queued":
        return {
            "kind": "queue_pending",
            "reason": "queued but not yet promoted to ready",
            "command_hint": "apply queue policy to promote when allowed",
        }
    return {
        "kind": "manual_review",
        "reason": "unknown state requires human validation",
        "command_hint": "inspect canonical queue artifacts and state taxonomy",
    }


def _collect_initiatives(core_queue: Dict[str, Any], expanded_queue: Dict[str, Any]) -> List[Dict[str, Any]]:
    initiatives: List[Dict[str, Any]] = []

    core_rows = core_queue.get("slices") if isinstance(core_queue.get("slices"), list) else []
    for row in core_rows:
        if not isinstance(row, dict):
            continue
        initiative_id = _core_id(row.get("id"))
        dependencies = [_core_id(dep) for dep in (row.get("dependencies") if isinstance(row.get("dependencies"), list) else [])]
        lanes = [str(x) for x in (row.get("lane") if isinstance(row.get("lane"), list) else []) if str(x).strip()]
        initiatives.append(
            {
                "id": initiative_id,
                "source_group": "core",
                "source_schema": str(core_queue.get("schema") or ""),
                "native_id": row.get("id"),
                "title": str(row.get("title") or ""),
                "objective": str(row.get("objective") or ""),
                "priority_tier": str(row.get("tier") or ""),
                "posture": "required",
                "lane_ids": lanes,
                "lane_name": ", ".join(lanes) if lanes else "",
                "state": str(row.get("state") or "UNKNOWN"),
                "normalized_state": normalize_state(str(row.get("state") or "")),
                "dependencies": dependencies,
                "recommended_order_hint": None,
            }
        )

    expanded_rows = expanded_queue.get("slices") if isinstance(expanded_queue.get("slices"), list) else []
    for row in expanded_rows:
        if not isinstance(row, dict):
            continue
        initiative_id = str(row.get("id") or "").strip()
        if not initiative_id:
            continue
        dependencies = [str(dep) for dep in (row.get("dependencies") if isinstance(row.get("dependencies"), list) else []) if str(dep).strip()]
        lane_id = str(row.get("lane_id") or "").strip()
        initiatives.append(
            {
                "id": initiative_id,
                "source_group": "expanded",
                "source_schema": str(expanded_queue.get("schema") or ""),
                "native_id": initiative_id,
                "title": str(row.get("title") or ""),
                "objective": str(row.get("objective") or ""),
                "priority_tier": str(row.get("tier") or ""),
                "posture": str(row.get("posture") or "required"),
                "lane_ids": [lane_id] if lane_id else [],
                "lane_name": str(row.get("lane_name") or ""),
                "state": str(row.get("state") or "UNKNOWN"),
                "normalized_state": normalize_state(str(row.get("state") or "")),
                "dependencies": dependencies,
                "recommended_order_hint": None,
            }
        )

    recommended_order = expanded_queue.get("recommended_order") if isinstance(expanded_queue.get("recommended_order"), list) else []
    order_lookup = {str(item): idx + 1 for idx, item in enumerate(recommended_order)}
    for item in initiatives:
        item_id = str(item.get("id") or "")
        if item_id in order_lookup:
            item["recommended_order_hint"] = order_lookup[item_id]

    return initiatives


def _augment_graph(initiatives: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    initiative_index = {str(item.get("id") or ""): item for item in initiatives}

    downstream: Dict[str, List[str]] = defaultdict(list)
    for item in initiatives:
        item_id = str(item.get("id") or "")
        for dep in item.get("dependencies") if isinstance(item.get("dependencies"), list) else []:
            dep_id = str(dep)
            downstream[dep_id].append(item_id)

    for item in initiatives:
        item_id = str(item.get("id") or "")
        dependencies = [str(dep) for dep in (item.get("dependencies") if isinstance(item.get("dependencies"), list) else []) if str(dep).strip()]
        dep_states, unresolved = _unresolved_dependencies(initiative_index, dependencies)
        item["dependency_states"] = dep_states
        item["unresolved_dependencies"] = unresolved
        item["dependents"] = sorted(set(downstream.get(item_id, [])))
        item["in_degree"] = len(dependencies)
        item["out_degree"] = len(item["dependents"])
        item["next_action"] = _next_action(
            normalized_state=str(item.get("normalized_state") or "unknown"),
            unresolved_dependencies=unresolved,
        )

    counts = Counter(str(item.get("normalized_state") or "unknown") for item in initiatives)
    actionable_now = sorted(str(item.get("id") or "") for item in initiatives if item.get("next_action", {}).get("kind") == "execute_now")
    blocked = [
        {
            "id": str(item.get("id") or ""),
            "unresolved_dependencies": list(item.get("unresolved_dependencies") or []),
        }
        for item in initiatives
        if str(item.get("normalized_state") or "") == "dependency_blocked"
    ]

    summary = {
        "total_initiatives": len(initiatives),
        "state_counts": {
            "done": int(counts.get("done", 0)),
            "ready": int(counts.get("ready", 0)),
            "running": int(counts.get("running", 0)),
            "dependency_blocked": int(counts.get("dependency_blocked", 0)),
            "queued": int(counts.get("queued", 0)),
            "unknown": int(counts.get("unknown", 0)),
        },
        "open_count": len(initiatives) - int(counts.get("done", 0)),
        "queue_empty": (len(initiatives) - int(counts.get("done", 0))) == 0,
        "actionable_now": actionable_now,
        "blocked": blocked,
    }

    adjacency = {
        "downstream_by_id": {k: sorted(set(v)) for k, v in sorted(downstream.items())},
    }

    return initiatives, {"summary": summary, "adjacency": adjacency}


def _lane_rollup(initiatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_lane: Dict[str, Dict[str, Any]] = {}
    for item in initiatives:
        lane_ids = [str(x) for x in (item.get("lane_ids") if isinstance(item.get("lane_ids"), list) else []) if str(x).strip()]
        if not lane_ids:
            lane_ids = ["UNSCOPED"]
        for lane_id in lane_ids:
            lane = by_lane.setdefault(
                lane_id,
                {
                    "lane_id": lane_id,
                    "initiative_ids": [],
                    "state_counts": Counter(),
                    "actionable_now": [],
                    "blocked": [],
                },
            )
            initiative_id = str(item.get("id") or "")
            lane["initiative_ids"].append(initiative_id)
            state = str(item.get("normalized_state") or "unknown")
            lane["state_counts"][state] += 1
            if item.get("next_action", {}).get("kind") == "execute_now":
                lane["actionable_now"].append(initiative_id)
            if state == "dependency_blocked":
                lane["blocked"].append(
                    {
                        "id": initiative_id,
                        "unresolved_dependencies": list(item.get("unresolved_dependencies") or []),
                    }
                )

    rows: List[Dict[str, Any]] = []
    for lane_id in sorted(by_lane.keys()):
        lane = by_lane[lane_id]
        counts = lane["state_counts"]
        total = len(lane["initiative_ids"])
        done_count = int(counts.get("done", 0))
        rows.append(
            {
                "lane_id": lane_id,
                "initiative_count": total,
                "done_count": done_count,
                "open_count": total - done_count,
                "state_counts": {
                    "done": done_count,
                    "ready": int(counts.get("ready", 0)),
                    "running": int(counts.get("running", 0)),
                    "dependency_blocked": int(counts.get("dependency_blocked", 0)),
                    "queued": int(counts.get("queued", 0)),
                    "unknown": int(counts.get("unknown", 0)),
                },
                "actionable_now": sorted(set(lane["actionable_now"])),
                "blocked": lane["blocked"],
            }
        )
    return rows


def _detect_non_authoritative_stale_signals(repo_root: Path, core_layer: Dict[str, Any], expanded_queue: Dict[str, Any]) -> List[Dict[str, Any]]:
    notices: List[Dict[str, Any]] = []
    probe = repo_root / "state" / "continuity" / "latest" / "core_roadmap_lane_execution_queue_2026-03-28.json"
    if not probe.exists():
        return notices

    try:
        payload = load_json(probe)
    except Exception as exc:  # pragma: no cover - defensive parsing branch
        notices.append(
            {
                "path": relpath(probe, repo_root),
                "status": "parse_error",
                "detail": str(exc),
            }
        )
        return notices

    core_empty = bool((core_layer.get("summary") or {}).get("queue_empty"))
    expanded_required_empty = bool((expanded_queue.get("summary") or {}).get("required_queue_empty"))
    lanes = payload.get("lanes") if isinstance(payload.get("lanes"), list) else []
    active_lane_count = sum(1 for lane in lanes if isinstance(lane, dict) and str(lane.get("status") or "").upper() == "ACTIVE")

    if core_empty and expanded_required_empty and active_lane_count > 0:
        notices.append(
            {
                "path": relpath(probe, repo_root),
                "status": "stale_non_authoritative_signal",
                "detail": "queue-authoritative surfaces are empty, but legacy lane queue still reports ACTIVE lanes",
            }
        )

    return notices


def _operator_surface(
    *,
    summary: Dict[str, Any],
    stale_notices: List[Dict[str, Any]],
) -> Dict[str, Any]:
    queue_empty = bool(summary.get("queue_empty"))
    actionable_now = summary.get("actionable_now") if isinstance(summary.get("actionable_now"), list) else []

    if queue_empty:
        headline = "QUEUE EMPTY: no runnable initiatives in authoritative core+expanded queues"
        immediate = [
            {
                "step": 1,
                "action": "Treat system as quiet-state; do not infer hidden backlog from stale support artifacts.",
            },
            {
                "step": 2,
                "action": "Before roadmap mutation, run source-of-truth map guard.",
                "command": "python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root . --map-path reports/openclaw_system_source_of_truth_map_2026-03-20.md --json",
            },
            {
                "step": 3,
                "action": "When a new bounded slice is added, regenerate this graph snapshot.",
                "command": "python scripts/roadmap_initiative_graph_runtime.py --repo-root /home/yeqiuqiu/clawd-architect --json",
            },
        ]
    else:
        headline = "ACTION REQUIRED: runnable initiatives exist in authoritative queues"
        immediate = [
            {
                "step": 1,
                "action": "Execute first actionable initiatives in canonical queue order.",
                "initiative_ids": actionable_now[:10],
            },
            {
                "step": 2,
                "action": "For blocked initiatives, resolve unresolved dependencies before retry.",
            },
        ]

    return {
        "headline": headline,
        "immediate_next_actions": immediate,
        "stale_signal_notices": stale_notices,
    }


def _operator_brief_markdown(graph: Dict[str, Any]) -> str:
    summary = graph.get("summary") if isinstance(graph.get("summary"), dict) else {}
    operator = graph.get("operator_surface") if isinstance(graph.get("operator_surface"), dict) else {}
    lane_rollup = graph.get("lane_rollup") if isinstance(graph.get("lane_rollup"), list) else []

    lines: List[str] = []
    lines.append("# Roadmap Initiative Graph Operator Brief")
    lines.append("")
    lines.append(f"Generated: `{graph.get('generated_at', '')}`")
    lines.append("")
    lines.append(f"**Headline:** {operator.get('headline', '')}")
    lines.append("")

    lines.append("## Queue summary")
    lines.append("")
    lines.append(f"- Total initiatives: **{summary.get('total_initiatives', 0)}**")
    lines.append(f"- Done: **{(summary.get('state_counts') or {}).get('done', 0)}**")
    lines.append(f"- Ready: **{(summary.get('state_counts') or {}).get('ready', 0)}**")
    lines.append(f"- Running: **{(summary.get('state_counts') or {}).get('running', 0)}**")
    lines.append(f"- Dependency blocked: **{(summary.get('state_counts') or {}).get('dependency_blocked', 0)}**")
    lines.append(f"- Queued: **{(summary.get('state_counts') or {}).get('queued', 0)}**")
    lines.append(f"- Queue empty: **{summary.get('queue_empty', False)}**")
    lines.append("")

    lines.append("## Immediate next actions")
    lines.append("")
    for row in operator.get("immediate_next_actions") if isinstance(operator.get("immediate_next_actions"), list) else []:
        if not isinstance(row, dict):
            continue
        step = row.get("step")
        action = row.get("action")
        command = row.get("command")
        lines.append(f"{step}. {action}")
        if command:
            lines.append(f"   - Command: `{command}`")
    lines.append("")

    stale_notices = operator.get("stale_signal_notices") if isinstance(operator.get("stale_signal_notices"), list) else []
    if stale_notices:
        lines.append("## Stale/non-authoritative signal notices")
        lines.append("")
        for row in stale_notices:
            if not isinstance(row, dict):
                continue
            lines.append(f"- `{row.get('path', '')}`: {row.get('detail', '')}")
        lines.append("")

    lines.append("## Lane rollup (open initiatives)")
    lines.append("")
    any_open = False
    for lane in lane_rollup:
        if not isinstance(lane, dict):
            continue
        open_count = int(lane.get("open_count") or 0)
        if open_count <= 0:
            continue
        any_open = True
        lines.append(f"- `{lane.get('lane_id', '')}`: open={open_count}, actionable={len(lane.get('actionable_now') or [])}, blocked={len(lane.get('blocked') or [])}")
    if not any_open:
        lines.append("- None (all lanes closed in authoritative queue views).")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    core_queue_path = Path(args.core_queue_path).resolve()
    core_layer_path = Path(args.core_layer_path).resolve()
    expanded_queue_path = Path(args.expanded_queue_path).resolve()
    graph_output_path = Path(args.graph_output_path).resolve()
    operator_brief_path = Path(args.operator_brief_path).resolve()

    core_queue = load_json(core_queue_path)
    core_layer = load_json(core_layer_path)
    expanded_queue = load_json(expanded_queue_path)

    generated_at = utc_now_iso()
    initiatives = _collect_initiatives(core_queue=core_queue, expanded_queue=expanded_queue)
    initiatives, graph_bits = _augment_graph(initiatives)
    summary = graph_bits["summary"]

    lane_rollup = _lane_rollup(initiatives)
    stale_notices = _detect_non_authoritative_stale_signals(
        repo_root=repo_root,
        core_layer=core_layer,
        expanded_queue=expanded_queue,
    )
    operator_surface = _operator_surface(summary=summary, stale_notices=stale_notices)

    graph_payload = {
        "schema": "clawd.roadmap_initiative_graph_snapshot.v1",
        "generated_at": generated_at,
        "scope": "openclaw_system_upgrade",
        "authoritative_sources": {
            "core_queue": {
                "path": relpath(core_queue_path, repo_root),
                "schema": str(core_queue.get("schema") or ""),
                "sha256": sha256(core_queue_path),
            },
            "core_queue_layer": {
                "path": relpath(core_layer_path, repo_root),
                "schema": str(core_layer.get("schema") or ""),
                "sha256": sha256(core_layer_path),
            },
            "expanded_queue": {
                "path": relpath(expanded_queue_path, repo_root),
                "schema": str(expanded_queue.get("schema") or ""),
                "sha256": sha256(expanded_queue_path),
            },
        },
        "summary": summary,
        "adjacency": graph_bits["adjacency"],
        "lane_rollup": lane_rollup,
        "initiatives": initiatives,
        "operator_surface": operator_surface,
    }

    write_json(graph_output_path, graph_payload)
    write_text(operator_brief_path, _operator_brief_markdown(graph_payload))

    if args.json:
        manifest = {
            "status": "PASS",
            "schema": "clawd.roadmap_initiative_graph_manifest.v1",
            "generated_at": generated_at,
            "outputs": {
                "graph": relpath(graph_output_path, repo_root),
                "operator_brief": relpath(operator_brief_path, repo_root),
            },
            "summary": summary,
        }
        print(json.dumps(manifest, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
