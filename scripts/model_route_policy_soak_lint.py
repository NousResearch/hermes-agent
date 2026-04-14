#!/usr/bin/env python3
"""Deterministic route-policy soak/lint snapshot (Wave 5)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from model_pool_policy_contract import load_pool_policy, policy_allowed_models, policy_route_entry


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_POOL_POLICY_PATH = Path("docs/ops/model_pool_policy_v1.json")
DEFAULT_POOL_POLICY_SCHEMA = Path("docs/ops/schemas/model_pool_policy.schema.json")
DEFAULT_GATE_DECISIONS = Path("state/continuity/model_rollout_gate_runner/decisions.jsonl")
DEFAULT_ROUTING_DECISIONS = Path("state/continuity/session_topology_router/decisions.jsonl")
DEFAULT_OUT_PATH = Path("state/continuity/model_route_policy_soak/latest.json")
DEFAULT_WINDOW_HOURS = 168


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_iso(raw: Any) -> Optional[dt.datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return (repo_root / path).resolve()
    return path.resolve()


def load_jsonl(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    if not path.exists():
        return []
    rows: List[Tuple[int, Dict[str, Any]]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append((idx, payload))
        else:
            raise ValueError(f"jsonl_non_object_line:{idx}")
    return rows


def _in_window(ts: Optional[dt.datetime], start: dt.datetime, now_ts: dt.datetime) -> bool:
    if ts is None:
        return False
    return start <= ts <= now_ts


def _normalize_allowed_stages(row: Mapping[str, Any]) -> set[str]:
    stages: set[str] = set()
    rollout = row.get("rollout") if isinstance(row.get("rollout"), Mapping) else {}
    for stage in rollout.get("allowed_stages") if isinstance(rollout.get("allowed_stages"), list) else []:
        text = str(stage or "").strip()
        if text:
            stages.add(text)
    approved_stage = str(rollout.get("approved_stage") or rollout.get("target_stage") or "").strip()
    if approved_stage:
        stages.add(approved_stage)
    return stages


def _stage_allowed(required_stage: str, allowed_stages: set[str]) -> bool:
    if required_stage == "canary":
        return "canary" in allowed_stages or "active" in allowed_stages
    if required_stage == "active":
        return "active" in allowed_stages
    return False


def build_snapshot(
    *,
    pool_policy: Mapping[str, Any],
    policy_meta: Mapping[str, Any],
    gate_rows: List[Tuple[int, Mapping[str, Any]]],
    routing_rows: List[Tuple[int, Mapping[str, Any]]],
    now_ts: dt.datetime,
    window_hours: int,
) -> Dict[str, Any]:
    window_start = now_ts - dt.timedelta(hours=window_hours)

    qualified: Dict[str, Dict[str, Any]] = {}
    gate_scanned = 0
    gate_in_window = 0
    for line_no, row in gate_rows:
        gate_scanned += 1
        ts = parse_iso(row.get("evaluated_at"))
        if not _in_window(ts, window_start, now_ts):
            continue
        gate_in_window += 1
        if str(row.get("schema") or "") != "clawd.model_rollout_gate.decision.v1":
            continue
        if str(row.get("decision") or "") != "PASS":
            continue
        model = row.get("model") if isinstance(row.get("model"), Mapping) else {}
        model_key = str(model.get("model_key") or model.get("model_ref") or "").strip()
        if not model_key:
            continue
        route_class = str(model.get("route_class") or model.get("model_family") or "").strip()
        allowed_stages = _normalize_allowed_stages(row)
        if not allowed_stages:
            continue
        current = qualified.get(model_key)
        if not current:
            qualified[model_key] = {
                "route_class": route_class,
                "allowed_stages": set(allowed_stages),
                "first_seen_line": line_no,
            }
        else:
            current["allowed_stages"] = set(current.get("allowed_stages") or set()).union(allowed_stages)
            if route_class and not current.get("route_class"):
                current["route_class"] = route_class

    violations: List[Dict[str, Any]] = []
    route_counts: Dict[str, int] = {"NO_LLM": 0, "SPARK": 0, "HEAVY": 0, "unknown": 0}
    pass_count = 0
    block_count = 0
    routing_scanned = 0
    routing_in_window = 0

    for line_no, row in routing_rows:
        routing_scanned += 1
        ts = parse_iso(row.get("evaluated_at"))
        if not _in_window(ts, window_start, now_ts):
            continue
        routing_in_window += 1

        if str(row.get("schema") or "") != "clawd.session_topology_routing.decision.v1":
            continue

        decision = str(row.get("decision") or "").strip().upper()
        route = row.get("route") if isinstance(row.get("route"), Mapping) else {}
        route_class = str(route.get("route_class") or "").strip()
        selected_model = str(route.get("selected_model") or "").strip() or None
        required_stage = str(route.get("required_rollout_stage") or "").strip()

        if route_class in route_counts:
            route_counts[route_class] += 1
        else:
            route_counts["unknown"] += 1

        if decision == "PASS":
            pass_count += 1
        else:
            block_count += 1

        if decision != "PASS":
            continue

        if policy_route_entry(pool_policy, route_class) is None:
            violations.append(
                {
                    "line": line_no,
                    "evaluated_at": row.get("evaluated_at"),
                    "violation": "route_class_not_in_policy",
                    "route_class": route_class or None,
                }
            )
            continue

        if route_class == "NO_LLM":
            if selected_model is not None:
                violations.append(
                    {
                        "line": line_no,
                        "evaluated_at": row.get("evaluated_at"),
                        "violation": "no_llm_selected_model_present",
                        "selected_model": selected_model,
                    }
                )
            continue

        if selected_model is None:
            violations.append(
                {
                    "line": line_no,
                    "evaluated_at": row.get("evaluated_at"),
                    "violation": "selected_model_missing",
                    "route_class": route_class,
                }
            )
            continue

        allowed_models = policy_allowed_models(pool_policy, route_class)
        if selected_model not in allowed_models:
            violations.append(
                {
                    "line": line_no,
                    "evaluated_at": row.get("evaluated_at"),
                    "violation": "selected_model_not_in_policy_route_pool",
                    "route_class": route_class,
                    "selected_model": selected_model,
                    "allowed_models": sorted(allowed_models),
                }
            )
            continue

        gate_meta = qualified.get(selected_model)
        if not gate_meta:
            violations.append(
                {
                    "line": line_no,
                    "evaluated_at": row.get("evaluated_at"),
                    "violation": "selected_model_not_qualified_in_window",
                    "route_class": route_class,
                    "selected_model": selected_model,
                    "required_rollout_stage": required_stage or None,
                }
            )
            continue

        gate_route = str(gate_meta.get("route_class") or "").strip()
        if gate_route and gate_route != route_class:
            violations.append(
                {
                    "line": line_no,
                    "evaluated_at": row.get("evaluated_at"),
                    "violation": "selected_model_route_class_mismatch",
                    "route_class": route_class,
                    "selected_model": selected_model,
                    "gate_route_class": gate_route,
                }
            )

        allowed_stages = set(str(x) for x in (gate_meta.get("allowed_stages") or set()))
        if required_stage and not _stage_allowed(required_stage, allowed_stages):
            violations.append(
                {
                    "line": line_no,
                    "evaluated_at": row.get("evaluated_at"),
                    "violation": "selected_model_stage_unqualified",
                    "route_class": route_class,
                    "selected_model": selected_model,
                    "required_rollout_stage": required_stage,
                    "qualified_allowed_stages": sorted(allowed_stages),
                }
            )

    status = "ok" if not violations else "policy_violations"

    return {
        "schema": "clawd.model_route_policy_soak_snapshot.v1",
        "generated_at": now_ts.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "window_hours": window_hours,
        "window_start": window_start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "policy": {
            "policy_id": pool_policy.get("policy_id"),
            "policy_path": policy_meta.get("path"),
            "policy_schema_path": policy_meta.get("schema_path"),
        },
        "counts": {
            "gate_rows_scanned": gate_scanned,
            "gate_rows_in_window": gate_in_window,
            "routing_rows_scanned": routing_scanned,
            "routing_rows_in_window": routing_in_window,
            "routing_pass": pass_count,
            "routing_block": block_count,
            "violations": len(violations),
            "route_class_counts": route_counts,
        },
        "violations": violations,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic route-policy soak/lint snapshot")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--pool-policy", default=str(DEFAULT_POOL_POLICY_PATH), help="Unified model pool policy JSON path")
    ap.add_argument("--pool-policy-schema", default=str(DEFAULT_POOL_POLICY_SCHEMA), help="Unified model pool policy schema path")
    ap.add_argument("--gate-decisions", default=str(DEFAULT_GATE_DECISIONS), help="Model rollout gate decisions JSONL")
    ap.add_argument("--routing-decisions", default=str(DEFAULT_ROUTING_DECISIONS), help="Session routing decisions JSONL")
    ap.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS, help="Replay/lint window in hours")
    ap.add_argument("--now", default="", help="Optional deterministic now timestamp (UTC ISO8601)")
    ap.add_argument("--out", default=str(DEFAULT_OUT_PATH), help="Output snapshot JSON path")
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    pool_policy_path = resolve_path(repo_root, args.pool_policy)
    pool_policy_schema_path = resolve_path(repo_root, args.pool_policy_schema)
    gate_decisions_path = resolve_path(repo_root, args.gate_decisions)
    routing_decisions_path = resolve_path(repo_root, args.routing_decisions)
    out_path = resolve_path(repo_root, args.out)

    now_ts = parse_iso(args.now) if str(args.now or "").strip() else dt.datetime.now(dt.timezone.utc)
    if now_ts is None:
        result = {
            "schema": "clawd.model_route_policy_soak_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": "invalid_now_timestamp",
            "raw_now": args.now,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    if not isinstance(args.window_hours, int) or args.window_hours <= 0:
        result = {
            "schema": "clawd.model_route_policy_soak_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": "invalid_window_hours",
            "window_hours": args.window_hours,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    policy_ok, reason, policy_meta, pool_policy = load_pool_policy(pool_policy_path, pool_policy_schema_path)
    if not policy_ok:
        result = {
            "schema": "clawd.model_route_policy_soak_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": reason,
            "details": policy_meta,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    try:
        gate_rows = load_jsonl(gate_decisions_path)
        routing_rows = load_jsonl(routing_decisions_path)
    except Exception as exc:
        result = {
            "schema": "clawd.model_route_policy_soak_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": "input_read_failed",
            "detail": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    snapshot = build_snapshot(
        pool_policy=pool_policy if isinstance(pool_policy, Mapping) else {},
        policy_meta=policy_meta,
        gate_rows=gate_rows,
        routing_rows=routing_rows,
        now_ts=now_ts,
        window_hours=int(args.window_hours),
    )

    if not is_within(repo_root, out_path):
        result = {
            "schema": "clawd.model_route_policy_soak_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": "unsafe_output_path",
            "path": str(out_path),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload = dict(snapshot)
    payload["written_path"] = str(out_path)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(payload))

    return 0 if snapshot.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
