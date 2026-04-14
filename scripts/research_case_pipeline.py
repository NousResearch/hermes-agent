#!/usr/bin/env python3
"""Research Case pipeline MVP (governed extraction -> understanding -> implementation).

This script implements a minimal local-first substrate for durable research promotion:
- canonical unit: Research Case folder under memory/research_cases/<case_id>
- explicit lifecycle + promotion metadata in rc.json
- head pointers + append-only ledger + checkpoint artifacts
- synthesis and candidate promotion contracts

It is intentionally narrow: first concrete slice, no broad framework rewrite.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent.parent
CASES_ROOT = ROOT / "memory" / "research_cases"
REGISTRY_PATH = ROOT / "state" / "continuity" / "latest" / "research_case_registry.json"
PROMOTION_GATE_RUNNER = ROOT / "scripts" / "promotion_gate_runner.py"
PROMOTION_SCHEMA_PATH = ROOT / "docs" / "ops" / "schemas" / "promotion_candidate.schema.json"
IMPLEMENTATION_QUEUE_SCHEMA_PATH = ROOT / "docs" / "ops" / "schemas" / "research_implementation_queue_item.schema.json"
DEPENDENCY_POLICY_PACK_PATH = ROOT / "state" / "continuity" / "latest" / "core_roadmap_dependency_unblock_policy_pack_v1.json"

PRIMARY_STATES = {
    "captured",
    "extracted",
    "triaged",
    "understanding_partial",
    "synthesis_partial",
    "synthesis_complete",
    "promotion_ready",
    "closed",
}

READING_STATES = {"unread", "skimmed", "partial", "complete"}
SYNTHESIS_STATES = {"none", "in_progress", "partial", "complete"}
PROMOTION_STATES = {"none", "mapped", "spec_draft", "gated", "promoted"}
DISPOSITIONS = {"active", "deferred", "abandoned", "archived", "superseded"}
UNDERSTANDING_LEVELS = {"exploratory", "partial", "substantial", "complete"}
WORK_STATUS = {"active", "paused", "blocked", "abandoned"}
FRESHNESS = {"current", "stale"}
GATE_DECISIONS = {"approved", "rejected", "needs_work"}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def rel_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_slug(raw: str, *, lower: bool = False, default: str = "id") -> str:
    text = str(raw or "").strip()
    if lower:
        text = text.lower()
    text = re.sub(r"[^a-z0-9A-Z._-]", "_", text)
    text = text.strip("._-")
    if not text:
        return default
    return text


def _build_source_ref(ref_id: str, path: Path, *, locator: str = "") -> Dict[str, Any]:
    rel = rel_to_root(path)
    row: Dict[str, Any] = {
        "ref_id": ref_id,
        "path": rel,
        "content_hash": f"sha256:{sha256_file(path)}",
    }
    if locator:
        row["locator"] = locator
    return row


def _run_promotion_gate(candidate_path: Path, *, decision_log_path: Path, publish_note_path: Path) -> Dict[str, Any]:
    if not PROMOTION_GATE_RUNNER.exists():
        return {
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "schema",
            "block_reason": "gate_unavailable",
            "gates": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "gate_unavailable",
                    "details": {"error": f"runner_missing:{PROMOTION_GATE_RUNNER}"},
                }
            ],
            "decision_record": {
                "enabled": False,
                "appended": False,
                "reason": "runner_missing",
            },
        }

    cmd = [
        sys.executable,
        str(PROMOTION_GATE_RUNNER),
        "--candidate",
        str(candidate_path),
        "--repo-root",
        str(ROOT),
        "--schema-path",
        str(PROMOTION_SCHEMA_PATH),
        "--decision-log",
        str(decision_log_path),
        "--publish-note-path",
        str(publish_note_path),
        "--json",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        return {
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "schema",
            "block_reason": "gate_unavailable",
            "gates": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "gate_unavailable",
                    "details": {"error": f"runner_exec_failed:{exc}"},
                }
            ],
            "decision_record": {
                "enabled": False,
                "appended": False,
                "reason": "runner_exec_failed",
            },
        }

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return {
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "schema",
            "block_reason": "gate_unavailable",
            "gates": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "gate_unavailable",
                    "details": {
                        "error": "runner_empty_output",
                        "return_code": proc.returncode,
                        "stderr": (proc.stderr or "").strip(),
                    },
                }
            ],
            "decision_record": {
                "enabled": False,
                "appended": False,
                "reason": "runner_empty_output",
            },
        }

    try:
        payload = json.loads(stdout)
    except Exception:
        return {
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "schema",
            "block_reason": "gate_unavailable",
            "gates": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "gate_unavailable",
                    "details": {
                        "error": "runner_output_not_json",
                        "stdout": stdout,
                        "stderr": (proc.stderr or "").strip(),
                        "return_code": proc.returncode,
                    },
                }
            ],
            "decision_record": {
                "enabled": False,
                "appended": False,
                "reason": "runner_output_not_json",
            },
        }

    if not isinstance(payload, dict):
        return {
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "schema",
            "block_reason": "gate_unavailable",
            "gates": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "gate_unavailable",
                    "details": {
                        "error": "runner_output_not_object",
                        "return_code": proc.returncode,
                    },
                }
            ],
            "decision_record": {
                "enabled": False,
                "appended": False,
                "reason": "runner_output_not_object",
            },
        }

    payload.setdefault("runner", {})
    if isinstance(payload["runner"], dict):
        payload["runner"]["return_code"] = proc.returncode
        stderr = (proc.stderr or "").strip()
        if stderr:
            payload["runner"]["stderr"] = stderr
    return payload


def _load_b2_capacity_policy() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "policy_id": "b2_capacity_starvation_concurrency_v1",
        "scheduler_policy": {
            "algorithm": "weighted_round_robin_with_aging",
            "global_hard_cap_active_cases": 8,
            "per_node_class_max_active_cases": {"small": 3, "medium": 5, "large": 8},
            "per_node_class_max_parallel_batch_replays": {"small": 2, "medium": 4, "large": 6},
            "node_class_rules": {
                "small": "<=8 vCPU OR <=16 GiB RAM",
                "medium": "9-16 vCPU OR 17-32 GiB RAM",
                "large": ">16 vCPU AND >32 GiB RAM",
            },
        },
        "starvation_policy": {
            "max_runnable_wait_seconds": 900,
            "starvation_incident_wait_seconds": 1200,
            "max_consecutive_scheduler_skips": 3,
            "preemption_trigger_runtime_seconds": 1800,
            "required_fairness_metric": "oldest_runnable_case_wait_seconds",
            "required_alerts": ["b2_starvation_threshold_breach", "b2_concurrency_cap_reached"],
        },
    }
    payload = read_json(DEPENDENCY_POLICY_PACK_PATH, default={}) or {}
    b2 = ((payload.get("slices") or {}).get("22") or {}) if isinstance(payload, dict) else {}
    if not isinstance(b2, dict) or not b2:
        return defaults

    scheduler = b2.get("scheduler_policy") if isinstance(b2.get("scheduler_policy"), dict) else {}
    starvation = b2.get("starvation_policy") if isinstance(b2.get("starvation_policy"), dict) else {}

    merged_scheduler = dict(defaults["scheduler_policy"])
    merged_scheduler.update(scheduler)

    merged_starvation = dict(defaults["starvation_policy"])
    merged_starvation.update(starvation)

    return {
        "policy_id": str(b2.get("policy_id") or defaults["policy_id"]),
        "scheduler_policy": merged_scheduler,
        "starvation_policy": merged_starvation,
    }


def _classify_node(*, cpu_count: int, memory_gib: float) -> str:
    if cpu_count <= 8 or memory_gib <= 16:
        return "small"
    if cpu_count > 16 and memory_gib > 32:
        return "large"
    return "medium"


def _parse_iso_to_utc(value: str) -> Optional[dt.datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _queue_wait_seconds(now: dt.datetime, queue_item: Dict[str, Any]) -> int:
    orchestr = queue_item.get("orchestration") if isinstance(queue_item.get("orchestration"), dict) else {}
    queue_entered = (
        orchestr.get("queue_entered_at")
        or queue_item.get("generated_at")
        or queue_item.get("created_at")
    )
    parsed = _parse_iso_to_utc(str(queue_entered or ""))
    if parsed is None:
        return 0
    return max(0, int((now - parsed).total_seconds()))


def _collect_implementation_items(*, queue_states: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not CASES_ROOT.exists():
        return rows

    allowed_states = {str(state).upper() for state in (queue_states or set()) if str(state).strip()}

    for rc_json in sorted(CASES_ROOT.glob("*/rc.json")):
        rc_payload = read_json(rc_json, default={}) or {}
        case_id = str(rc_payload.get("case_id") or rc_json.parent.name)
        for queue_path in sorted(rc_json.parent.glob("CANDIDATE/candidates/*/implementation_queue_item.json")):
            payload = read_json(queue_path, default={}) or {}
            if not isinstance(payload, dict) or not payload:
                continue
            queue_state = str(payload.get("queue_state") or "").upper()
            if allowed_states and queue_state not in allowed_states:
                continue
            payload["_queue_item_path"] = queue_path
            payload["_queue_item_path_rel"] = rel_to_root(queue_path)
            payload["case_id"] = str(payload.get("case_id") or case_id)
            payload["candidate_id"] = str(payload.get("candidate_id") or queue_path.parent.name)
            payload["_queue_state"] = queue_state
            rows.append(payload)
    return rows


def _collect_ready_implementation_items() -> List[Dict[str, Any]]:
    return _collect_implementation_items(queue_states={"READY_FOR_EXECUTION", "IN_EXECUTION"})


def _fair_order_with_round_robin(candidates: List[Dict[str, Any]], *, last_selected_case_id: Optional[str]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in candidates:
        cid = str(row.get("case_id") or "")
        grouped.setdefault(cid, []).append(row)

    for case_rows in grouped.values():
        case_rows.sort(
            key=lambda r: (
                -int(r.get("_starvation_flag") or 0),
                -float(r.get("_priority_score") or 0.0),
                -int(r.get("_wait_seconds") or 0),
                str(r.get("queue_item_id") or ""),
            )
        )

    case_order: List[str] = []
    for row in candidates:
        cid = str(row.get("case_id") or "")
        if cid and cid not in case_order:
            case_order.append(cid)
    if last_selected_case_id and last_selected_case_id in case_order:
        idx = case_order.index(last_selected_case_id)
        case_order = case_order[idx + 1 :] + case_order[: idx + 1]

    ordered: List[Dict[str, Any]] = []
    while True:
        emitted = False
        for case_id in case_order:
            bucket = grouped.get(case_id) or []
            if bucket:
                ordered.append(bucket.pop(0))
                emitted = True
        if not emitted:
            break
    return ordered


def _plan_capacity_orchestration(
    *,
    now: dt.datetime,
    node_class: str,
    policy: Dict[str, Any],
    active_case_ids: List[str],
    ready_items: List[Dict[str, Any]],
    previous_runtime: Dict[str, Any],
) -> Dict[str, Any]:
    scheduler = policy.get("scheduler_policy") if isinstance(policy.get("scheduler_policy"), dict) else {}
    starvation = policy.get("starvation_policy") if isinstance(policy.get("starvation_policy"), dict) else {}

    per_node_max = scheduler.get("per_node_class_max_active_cases") if isinstance(scheduler.get("per_node_class_max_active_cases"), dict) else {}
    node_cap = int(per_node_max.get(node_class) or scheduler.get("global_hard_cap_active_cases") or 8)
    global_cap = int(scheduler.get("global_hard_cap_active_cases") or node_cap)
    cap = max(1, min(node_cap, global_cap))

    max_wait = int(starvation.get("max_runnable_wait_seconds") or 900)
    incident_wait = int(starvation.get("starvation_incident_wait_seconds") or max_wait)
    max_skips = int(starvation.get("max_consecutive_scheduler_skips") or 3)

    skip_state = previous_runtime.get("skip_state") if isinstance(previous_runtime.get("skip_state"), dict) else {}
    prev_skip_by_item = skip_state.get("by_queue_item") if isinstance(skip_state.get("by_queue_item"), dict) else {}

    active_set = {str(x).strip() for x in active_case_ids if str(x).strip()}
    active_count = len(active_set)
    available_slots = max(0, cap - active_count)

    candidate_pool: List[Dict[str, Any]] = []
    oldest_wait_seconds = 0
    for item in ready_items:
        case_id = str(item.get("case_id") or "")
        if not case_id or case_id in active_set:
            continue
        queue_item_id = str(item.get("queue_item_id") or "")
        if not queue_item_id:
            continue

        wait_seconds = _queue_wait_seconds(now, item)
        oldest_wait_seconds = max(oldest_wait_seconds, wait_seconds)

        previous_skip = int(prev_skip_by_item.get(queue_item_id) or 0)
        weight = 1.0
        orch = item.get("orchestration") if isinstance(item.get("orchestration"), dict) else {}
        try:
            weight = float(orch.get("priority_weight") or 1.0)
        except Exception:
            weight = 1.0

        starvation_flag = int(wait_seconds >= max_wait or previous_skip >= max_skips)
        priority_score = weight + (wait_seconds / max(1, max_wait)) + (3.0 if starvation_flag else 0.0)

        row = dict(item)
        row["_wait_seconds"] = wait_seconds
        row["_priority_score"] = round(priority_score, 6)
        row["_starvation_flag"] = starvation_flag
        row["_previous_skips"] = previous_skip
        candidate_pool.append(row)

    candidate_pool.sort(
        key=lambda r: (
            -int(r.get("_starvation_flag") or 0),
            -float(r.get("_priority_score") or 0.0),
            -int(r.get("_wait_seconds") or 0),
            str(r.get("case_id") or ""),
            str(r.get("queue_item_id") or ""),
        )
    )

    last_selected = str(previous_runtime.get("last_selected_case_id") or "").strip() or None
    rr_ordered = _fair_order_with_round_robin(candidate_pool, last_selected_case_id=last_selected)

    selected = rr_ordered[:available_slots]
    selected_ids = {str(row.get("queue_item_id") or "") for row in selected}

    alerts: List[str] = []
    if rr_ordered and available_slots == 0:
        alerts.append("b2_concurrency_cap_reached")

    if any(int(row.get("_wait_seconds") or 0) >= incident_wait for row in rr_ordered):
        alerts.append("b2_starvation_threshold_breach")
    if any(int(row.get("_previous_skips") or 0) >= max_skips for row in rr_ordered):
        if "b2_starvation_threshold_breach" not in alerts:
            alerts.append("b2_starvation_threshold_breach")

    new_skip_by_item: Dict[str, int] = {}
    for row in rr_ordered:
        item_id = str(row.get("queue_item_id") or "")
        if not item_id:
            continue
        if item_id in selected_ids:
            new_skip_by_item[item_id] = 0
        else:
            new_skip_by_item[item_id] = int(row.get("_previous_skips") or 0) + 1

    selected_case_ids: List[str] = []
    for row in selected:
        case_id = str(row.get("case_id") or "")
        if case_id and case_id not in selected_case_ids:
            selected_case_ids.append(case_id)

    runtime_payload: Dict[str, Any] = {
        "schema": "clawd.research_case.capacity_orchestration_runtime.v1",
        "generated_at": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "policy_id": str(policy.get("policy_id") or "b2_capacity_starvation_concurrency_v1"),
        "scheduler_algorithm": str(scheduler.get("algorithm") or "weighted_round_robin_with_aging"),
        "node_class": node_class,
        "limits": {
            "global_hard_cap_active_cases": global_cap,
            "node_class_cap_active_cases": cap,
            "active_case_count": active_count,
            "available_slots": available_slots,
            "max_consecutive_scheduler_skips": max_skips,
            "max_runnable_wait_seconds": max_wait,
            "starvation_incident_wait_seconds": incident_wait,
        },
        "fairness": {
            "metric": str(starvation.get("required_fairness_metric") or "oldest_runnable_case_wait_seconds"),
            "oldest_runnable_case_wait_seconds": oldest_wait_seconds,
            "starvation_threshold_breached": "b2_starvation_threshold_breach" in alerts,
        },
        "alerts": alerts,
        "selected": [
            {
                "queue_item_id": str(row.get("queue_item_id") or ""),
                "case_id": str(row.get("case_id") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "wait_seconds": int(row.get("_wait_seconds") or 0),
                "priority_score": float(row.get("_priority_score") or 0.0),
                "starvation_flag": bool(row.get("_starvation_flag") or 0),
                "queue_item_path": str(row.get("_queue_item_path_rel") or ""),
            }
            for row in selected
        ],
        "runnable_pool": [
            {
                "queue_item_id": str(row.get("queue_item_id") or ""),
                "case_id": str(row.get("case_id") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "wait_seconds": int(row.get("_wait_seconds") or 0),
                "priority_score": float(row.get("_priority_score") or 0.0),
                "starvation_flag": bool(row.get("_starvation_flag") or 0),
                "previous_skips": int(row.get("_previous_skips") or 0),
                "next_skip_count": int(new_skip_by_item.get(str(row.get("queue_item_id") or "")) or 0),
                "queue_item_path": str(row.get("_queue_item_path_rel") or ""),
            }
            for row in rr_ordered
        ],
        "skip_state": {
            "by_queue_item": new_skip_by_item,
        },
        "active_case_ids": sorted(active_set),
        "selected_case_ids": selected_case_ids,
        "last_selected_case_id": selected_case_ids[-1] if selected_case_ids else last_selected,
    }

    return runtime_payload


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _deterministic_replay_scores(
    *,
    queue_item_id: str,
    case_id: str,
    candidate_id: str,
    replay_seed: str,
    priority_weight: float,
    wait_seconds: int,
    tuning_bias: float,
) -> Tuple[float, float, float, str]:
    seed_material = "|".join([replay_seed, queue_item_id, case_id, candidate_id])
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    sample = int(digest[:8], 16) / 0xFFFFFFFF

    baseline_quality = round(0.55 + (0.35 * sample), 4)
    priority_bonus = _clamp((priority_weight - 1.0) * 0.03, -0.06, 0.09)
    wait_bonus = _clamp(wait_seconds / 7200.0, 0.0, 0.08)
    candidate_quality = round(_clamp(baseline_quality + tuning_bias + priority_bonus + wait_bonus, 0.0, 1.0), 4)
    quality_delta = round(candidate_quality - baseline_quality, 4)
    return baseline_quality, candidate_quality, quality_delta, digest


def _plan_batch_replay(
    *,
    now: dt.datetime,
    node_class: str,
    policy: Dict[str, Any],
    replay_candidates: List[Dict[str, Any]],
    replay_seed: str,
    max_replays_override: int,
    quality_gate: float,
    min_quality_delta: float,
    tuning_bias: float,
    replay_target_state: str,
) -> Dict[str, Any]:
    scheduler = policy.get("scheduler_policy") if isinstance(policy.get("scheduler_policy"), dict) else {}
    starvation = policy.get("starvation_policy") if isinstance(policy.get("starvation_policy"), dict) else {}
    replay_caps = scheduler.get("per_node_class_max_parallel_batch_replays") if isinstance(scheduler.get("per_node_class_max_parallel_batch_replays"), dict) else {}

    node_cap = int(replay_caps.get(node_class) or 1)
    bounded_cap = max(1, node_cap)
    effective_cap = min(bounded_cap, max_replays_override) if max_replays_override > 0 else bounded_cap

    ordered: List[Dict[str, Any]] = []
    for row in replay_candidates:
        queue_item_id = str(row.get("queue_item_id") or "")
        case_id = str(row.get("case_id") or "")
        candidate_id = str(row.get("candidate_id") or "")
        if not queue_item_id or not case_id or not candidate_id:
            continue

        wait_seconds = _queue_wait_seconds(now, row)
        orchestration = row.get("orchestration") if isinstance(row.get("orchestration"), dict) else {}
        try:
            priority_weight = float(orchestration.get("priority_weight") or 1.0)
        except Exception:
            priority_weight = 1.0

        replay_row = dict(row)
        replay_row["_wait_seconds"] = wait_seconds
        replay_row["_priority_weight"] = round(priority_weight, 4)
        ordered.append(replay_row)

    ordered.sort(
        key=lambda r: (
            -int(r.get("_wait_seconds") or 0),
            -float(r.get("_priority_weight") or 1.0),
            str(r.get("case_id") or ""),
            str(r.get("queue_item_id") or ""),
        )
    )

    selected = ordered[:effective_cap]
    results: List[Dict[str, Any]] = []
    pass_count = 0
    quality_deltas: List[float] = []
    starvation_threshold = int(starvation.get("starvation_incident_wait_seconds") or starvation.get("max_runnable_wait_seconds") or 1200)

    for row in selected:
        queue_item_id = str(row.get("queue_item_id") or "")
        case_id = str(row.get("case_id") or "")
        candidate_id = str(row.get("candidate_id") or "")
        wait_seconds = int(row.get("_wait_seconds") or 0)
        priority_weight = float(row.get("_priority_weight") or 1.0)

        baseline_quality, candidate_quality, quality_delta, run_hash = _deterministic_replay_scores(
            queue_item_id=queue_item_id,
            case_id=case_id,
            candidate_id=candidate_id,
            replay_seed=replay_seed,
            priority_weight=priority_weight,
            wait_seconds=wait_seconds,
            tuning_bias=tuning_bias,
        )

        replay_pass = candidate_quality >= quality_gate and quality_delta >= min_quality_delta
        if replay_pass:
            pass_count += 1
        quality_deltas.append(quality_delta)

        lifecycle = [
            {
                "stage": "intake_replay",
                "status": "PASS",
                "details": {
                    "target_state": replay_target_state,
                    "queue_state": str(row.get("_queue_state") or "UNKNOWN"),
                },
            },
            {
                "stage": "synthesis_replay",
                "status": "PASS",
                "details": {
                    "policy_id": str((policy.get("policy_id") or "b2_capacity_starvation_concurrency_v1")),
                    "priority_weight": priority_weight,
                },
            },
            {
                "stage": "promotion_gate_replay",
                "status": "PASS" if replay_pass else "FAIL",
                "details": {
                    "quality_gate": quality_gate,
                    "min_quality_delta": min_quality_delta,
                },
            },
            {
                "stage": "implementation_queue_handoff_replay",
                "status": "PASS" if replay_pass else "FAIL",
                "details": {
                    "queue_item_path": str(row.get("_queue_item_path_rel") or ""),
                    "state_transition": f"{row.get('_queue_state') or 'UNKNOWN'}->{replay_target_state}",
                },
            },
        ]

        results.append(
            {
                "queue_item_id": queue_item_id,
                "case_id": case_id,
                "candidate_id": candidate_id,
                "queue_state": str(row.get("_queue_state") or "UNKNOWN"),
                "queue_item_path": str(row.get("_queue_item_path_rel") or ""),
                "wait_seconds": wait_seconds,
                "priority_weight": priority_weight,
                "run_hash": run_hash,
                "metrics": {
                    "baseline_quality_score": baseline_quality,
                    "candidate_quality_score": candidate_quality,
                    "quality_delta": quality_delta,
                    "quality_gate": quality_gate,
                    "min_quality_delta": min_quality_delta,
                    "replay_pass": replay_pass,
                },
                "lifecycle": lifecycle,
            }
        )

    fail_count = len(results) - pass_count
    avg_delta = round(sum(quality_deltas) / len(quality_deltas), 4) if quality_deltas else 0.0

    cohort_digest = hashlib.sha256(
        "|".join(f"{row['queue_item_id']}:{row['metrics']['quality_delta']}" for row in results).encode("utf-8")
    ).hexdigest()

    alerts: List[str] = []
    if not results:
        alerts.append("b2_replay_cohort_empty")
    if fail_count > 0:
        alerts.append("b2_replay_quality_regression_detected")
    if any(int(row.get("wait_seconds") or 0) >= starvation_threshold for row in results):
        alerts.append("b2_starvation_threshold_breach")

    payload: Dict[str, Any] = {
        "schema": "clawd.research_case.batch_replay_runtime.v1",
        "generated_at": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "policy_id": str(policy.get("policy_id") or "b2_capacity_starvation_concurrency_v1"),
        "scheduler_algorithm": str(scheduler.get("algorithm") or "weighted_round_robin_with_aging"),
        "replay_seed": replay_seed,
        "node_class": node_class,
        "limits": {
            "node_class_max_parallel_batch_replays": bounded_cap,
            "requested_max_replays": max_replays_override if max_replays_override > 0 else None,
            "effective_parallel_replay_cap": effective_cap,
            "cohort_size": len(ordered),
        },
        "targets": {
            "replay_target_state": replay_target_state,
            "quality_gate": quality_gate,
            "min_quality_delta": min_quality_delta,
            "tuning_bias": tuning_bias,
        },
        "summary": {
            "selected_count": len(results),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": round((pass_count / len(results)), 4) if results else 0.0,
            "average_quality_delta": avg_delta,
            "cohort_digest": cohort_digest,
        },
        "alerts": alerts,
        "results": results,
    }
    return payload


def directory_fingerprint(path: Path) -> Dict[str, Any]:
    files: List[Path] = [p for p in path.rglob("*") if p.is_file()]
    files_sorted = sorted(files)
    h = hashlib.sha256()
    total_bytes = 0
    for file_path in files_sorted:
        rel = file_path.relative_to(path).as_posix()
        st = file_path.stat()
        total_bytes += int(st.st_size)
        h.update(rel.encode("utf-8"))
        h.update(str(int(st.st_size)).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
    return {
        "file_count": len(files_sorted),
        "total_bytes": total_bytes,
        "fingerprint_sha256": h.hexdigest(),
    }


def detect_batch_tags(text: str) -> List[str]:
    tags = re.findall(r"batch_\d+_\d{4}-\d{2}-\d{2}", text)
    uniq: List[str] = []
    for tag in tags:
        if tag not in uniq:
            uniq.append(tag)
    return uniq


def case_dir(case_id: str) -> Path:
    return CASES_ROOT / case_id


def load_case(case_id: str) -> Dict[str, Any]:
    rc_path = case_dir(case_id) / "rc.json"
    rc = read_json(rc_path)
    if not isinstance(rc, dict):
        raise SystemExit(f"missing_or_invalid_case: {rc_path}")
    return rc


def write_case(case_id: str, rc: Dict[str, Any]) -> None:
    rc["updated_at"] = now_iso()
    atomic_write_json(case_dir(case_id) / "rc.json", rc)


def checkpoint_payload(rc: Dict[str, Any], next_actions: Optional[List[str]], do_not_do: Optional[List[str]], notes: str) -> Dict[str, Any]:
    return {
        "schema": "clawd.research_case.checkpoint.v1",
        "generated_at": now_iso(),
        "case_id": rc.get("case_id"),
        "primary_state": rc.get("primary_state"),
        "lifecycle": rc.get("lifecycle"),
        "reliability_flags": rc.get("reliability_flags"),
        "heads": rc.get("heads"),
        "promotion": rc.get("promotion"),
        "next_actions": next_actions or [],
        "do_not_do": do_not_do or [],
        "notes": notes,
    }


def write_checkpoint(case_id: str, rc: Dict[str, Any], next_actions: Optional[List[str]] = None, do_not_do: Optional[List[str]] = None, notes: str = "") -> Dict[str, Any]:
    cdir = case_dir(case_id)
    payload = checkpoint_payload(rc, next_actions, do_not_do, notes)

    cp_json = cdir / "CHECKPOINT" / "latest.json"
    cp_md = cdir / "CHECKPOINT" / "latest.md"
    atomic_write_json(cp_json, payload)

    lines = [
        f"# Research Case Checkpoint — {case_id}",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- primary_state: {payload.get('primary_state')}",
        f"- understanding_level: {(payload.get('reliability_flags') or {}).get('understanding_level')}",
        f"- promotion_gate_status: {(payload.get('promotion') or {}).get('gate_status')}",
        "",
        "## Active heads",
        "",
    ]
    heads = payload.get("heads") or {}
    lines.append(f"- synthesis_head_id: `{heads.get('synthesis_head_id')}`")
    lines.append(f"- candidate_head_id: `{heads.get('candidate_head_id')}`")
    lines.append("")

    lines.append("## Next actions")
    if payload["next_actions"]:
        for row in payload["next_actions"]:
            lines.append(f"- {row}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## DO NOT DO")
    if payload["do_not_do"]:
        for row in payload["do_not_do"]:
            lines.append(f"- {row}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Notes")
    lines.append(payload["notes"] or "-")
    lines.append("")

    write_text(cp_md, "\n".join(lines) + "\n")

    report_path = ROOT / "reports" / f"research_case_{case_id}_handover_latest.md"
    write_text(report_path, "\n".join(lines) + "\n")

    return {
        "checkpoint_json": rel_to_root(cp_json),
        "checkpoint_md": rel_to_root(cp_md),
        "handover_report": rel_to_root(report_path),
    }


def emit(data: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    status = data.get("status", "ok")
    print(f"research_case_pipeline: {status}")
    summary = data.get("summary")
    if isinstance(summary, dict):
        for k in sorted(summary.keys()):
            print(f"- {k}: {summary[k]}")


def cmd_init(args: argparse.Namespace) -> None:
    cid = args.case_id.strip()
    cdir = case_dir(cid)
    rc_path = cdir / "rc.json"

    if rc_path.exists() and not args.force:
        raise SystemExit(f"case_exists: {rc_path} (use --force to overwrite)")

    now = now_iso()
    ensure_dir(cdir)
    for rel in [
        "RAW/blobs",
        "EXTRACT/extract_runs",
        "ANALYZE/analyses",
        "SYNTH/synths",
        "CANDIDATE/candidates",
        "IMPLEMENT/slices",
        "LEDGER",
        "CHECKPOINT",
    ]:
        ensure_dir(cdir / rel)

    source_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    batch_tags: List[str] = []

    for idx, raw in enumerate(args.source or [], start=1):
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if not p.exists():
            missing.append(str(p))
            continue

        path_text = str(p)
        for tag in detect_batch_tags(path_text):
            if tag not in batch_tags:
                batch_tags.append(tag)

        row: Dict[str, Any] = {
            "source_id": f"sd_{idx:03d}",
            "path": str(p.resolve()),
            "path_rel": rel_to_root(p),
            "registered_at": now,
        }

        if p.is_file():
            st = p.stat()
            row.update(
                {
                    "kind": "file",
                    "sha256": sha256_file(p),
                    "bytes": int(st.st_size),
                    "mtime": dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                }
            )
        elif p.is_dir():
            fp = directory_fingerprint(p)
            row.update({"kind": "directory", **fp})
        else:
            row.update({"kind": "other"})

        source_rows.append(row)

    if missing:
        raise SystemExit("missing_sources:\n" + "\n".join(f"- {x}" for x in missing))

    raw_manifest = {
        "schema": "clawd.research_case.raw_manifest.v1",
        "case_id": cid,
        "generated_at": now,
        "source_count": len(source_rows),
        "sources": source_rows,
    }
    atomic_write_json(cdir / "RAW" / "raw_manifest.json", raw_manifest)

    head = {
        "schema": "clawd.research_case.synthesis_head.v1",
        "case_id": cid,
        "active_synth_id": None,
        "active_synth_path": None,
        "updated_at": now,
    }
    atomic_write_json(cdir / "SYNTH" / "SYNTHESIS_HEAD.json", head)

    rc = {
        "schema": "clawd.research_case.v1",
        "case_id": cid,
        "title": args.title,
        "intent": args.intent,
        "created_at": now,
        "updated_at": now,
        "primary_state": "captured",
        "lifecycle": {
            "intake_state": "captured",
            "reading_state": "unread",
            "synthesis_state": "none",
            "promotion_state": "none",
            "disposition": "active",
        },
        "reliability_flags": {
            "understanding_level": "exploratory",
            "work_status": "active",
            "freshness": "current",
            "partial": False,
            "partial_reason_code": None,
        },
        "promotion": {
            "gate_status": "blocked",
            "last_decision": "none",
            "last_gatecheck_path": None,
            "last_candidate_id": None,
        },
        "heads": {
            "extract_run_id": None,
            "synthesis_head_id": None,
            "candidate_head_id": None,
        },
        "contracts": {
            "raw_manifest": "RAW/raw_manifest.json",
            "synthesis_head": "SYNTH/SYNTHESIS_HEAD.json",
            "events": "LEDGER/events.jsonl",
            "checkpoint": "CHECKPOINT/latest.json",
        },
        "source_overview": {
            "source_count": len(source_rows),
            "batch_arc": batch_tags,
        },
    }

    write_case(cid, rc)
    append_jsonl(
        cdir / "LEDGER" / "events.jsonl",
        {
            "ts": now_iso(),
            "event": "rc_initialized",
            "case_id": cid,
            "source_count": len(source_rows),
            "batch_arc": batch_tags,
        },
    )

    cp_paths = write_checkpoint(
        cid,
        rc,
        next_actions=["Record synthesis from governed extraction + batch 1-4 arc", "Promote first implementation candidate via gatecheck"],
        do_not_do=["Do not mark promotion_state=promoted without gatecheck artifact"],
        notes="Research Case scaffold initialized.",
    )

    emit(
        {
            "status": "ok",
            "case_id": cid,
            "paths": {
                "case_dir": rel_to_root(cdir),
                "rc_json": rel_to_root(rc_path),
                "raw_manifest": rel_to_root(cdir / "RAW" / "raw_manifest.json"),
                **cp_paths,
            },
            "summary": {
                "batch_arc_tags": len(batch_tags),
                "source_count": len(source_rows),
            },
        },
        as_json=args.json,
    )


def _ensure_enum(value: str, allowed: set[str], field: str) -> str:
    if value not in allowed:
        raise SystemExit(f"invalid_{field}: {value} (allowed: {sorted(allowed)})")
    return value


def cmd_record_synth(args: argparse.Namespace) -> None:
    cid = args.case_id
    cdir = case_dir(cid)
    rc = load_case(cid)

    synth_id = args.synth_id.strip()
    if not synth_id:
        raise SystemExit("missing_synth_id")

    source_manifest = read_json(cdir / "RAW" / "raw_manifest.json", default={}) or {}
    known_source_ids = {str(row.get("source_id")) for row in source_manifest.get("sources", []) if isinstance(row, dict)}

    evidence_source_ids = args.evidence_source_id or []
    if not evidence_source_ids:
        # default to first known source for minimum traceability path
        evidence_source_ids = sorted(known_source_ids)[:1]

    invalid_source_ids = [sid for sid in evidence_source_ids if sid not in known_source_ids]
    if invalid_source_ids:
        raise SystemExit(f"unknown_evidence_source_ids: {invalid_source_ids}")

    takeaways = args.takeaway or []
    if not takeaways:
        raise SystemExit("at_least_one_takeaway_required")

    prev_head_id = (rc.get("heads") or {}).get("synthesis_head_id")
    if prev_head_id and prev_head_id != synth_id:
        prev_json = cdir / "SYNTH" / "synths" / f"{prev_head_id}.json"
        prev_data = read_json(prev_json, default={}) or {}
        if isinstance(prev_data, dict) and prev_data:
            prev_data["status"] = "superseded"
            prev_data["superseded_at"] = now_iso()
            prev_data["superseded_by"] = synth_id
            atomic_write_json(prev_json, prev_data)

    takeaway_rows: List[Dict[str, Any]] = []
    for idx, text in enumerate(takeaways, start=1):
        takeaway_rows.append(
            {
                "takeaway_id": f"tw_{idx:03d}",
                "summary": text,
                "status": "active",
                "evidence": [
                    {
                        "source_id": sid,
                        "ref_type": "source_document",
                        "locator": None,
                        "quote": None,
                    }
                    for sid in evidence_source_ids
                ],
            }
        )

    summary_md = args.summary_text or ""
    if args.summary_file:
        summary_md = Path(args.summary_file).expanduser().read_text(encoding="utf-8")
    if not summary_md.strip():
        bullets = "\n".join(f"- {row['summary']}" for row in takeaway_rows)
        summary_md = (
            f"# Synthesis — {synth_id}\n\n"
            f"## Summary\n{args.summary_text or 'Implementation-oriented synthesis recorded.'}\n\n"
            f"## Takeaways\n{bullets}\n"
        )

    synth_payload = {
        "schema": "clawd.research_case.synthesis.v1",
        "case_id": cid,
        "synth_id": synth_id,
        "created_at": now_iso(),
        "created_by": args.created_by,
        "status": "active",
        "supersedes": prev_head_id,
        "understanding_level": _ensure_enum(args.understanding_level, UNDERSTANDING_LEVELS, "understanding_level"),
        "open_questions": args.open_question or [],
        "takeaways": takeaway_rows,
    }

    synth_dir = cdir / "SYNTH" / "synths"
    atomic_write_json(synth_dir / f"{synth_id}.json", synth_payload)
    write_text(synth_dir / f"{synth_id}.md", summary_md if summary_md.endswith("\n") else summary_md + "\n")

    head_payload = {
        "schema": "clawd.research_case.synthesis_head.v1",
        "case_id": cid,
        "active_synth_id": synth_id,
        "active_synth_path": f"SYNTH/synths/{synth_id}.json",
        "supersedes": prev_head_id,
        "updated_at": now_iso(),
    }
    atomic_write_json(cdir / "SYNTH" / "SYNTHESIS_HEAD.json", head_payload)

    lifecycle = rc.get("lifecycle") or {}
    lifecycle["reading_state"] = _ensure_enum(args.reading_state, READING_STATES, "reading_state")
    lifecycle["synthesis_state"] = "complete" if args.complete else "partial"

    rc["primary_state"] = "synthesis_complete" if args.complete else "synthesis_partial"
    rc["lifecycle"] = lifecycle

    flags = rc.get("reliability_flags") or {}
    flags["understanding_level"] = _ensure_enum(args.understanding_level, UNDERSTANDING_LEVELS, "understanding_level")
    flags["partial"] = not args.complete
    flags["partial_reason_code"] = args.partial_reason_code if not args.complete else None
    rc["reliability_flags"] = flags

    heads = rc.get("heads") or {}
    heads["synthesis_head_id"] = synth_id
    rc["heads"] = heads

    write_case(cid, rc)

    append_jsonl(
        cdir / "LEDGER" / "events.jsonl",
        {
            "ts": now_iso(),
            "event": "synthesis_recorded",
            "case_id": cid,
            "synth_id": synth_id,
            "takeaway_count": len(takeaway_rows),
            "complete": bool(args.complete),
            "supersedes": prev_head_id,
        },
    )

    cp_paths = write_checkpoint(
        cid,
        rc,
        next_actions=["Map synthesis takeaways into candidate requirements", "Run promotion gatecheck"],
        do_not_do=["Do not open implementation slice before approved gatecheck"],
        notes="Synthesis recorded and set as active head.",
    )

    emit(
        {
            "status": "ok",
            "case_id": cid,
            "synth_id": synth_id,
            "paths": {
                "synth_json": rel_to_root(synth_dir / f"{synth_id}.json"),
                "synth_md": rel_to_root(synth_dir / f"{synth_id}.md"),
                "synthesis_head": rel_to_root(cdir / "SYNTH" / "SYNTHESIS_HEAD.json"),
                **cp_paths,
            },
            "summary": {
                "takeaway_count": len(takeaway_rows),
                "state": rc.get("primary_state"),
            },
        },
        as_json=args.json,
    )


def _emit_implementation_queue_item(
    *,
    case_id: str,
    candidate_id: str,
    promotion_id: str,
    candidate_json_path: Path,
    gatecheck_path: Path,
    promotion_candidate_path: Path,
    promotion_gate_decision_path: Path,
    promotion_gate_decision_log_path: Path,
    source_refs: List[Dict[str, Any]],
    requirements: List[Dict[str, Any]],
    acceptance_criteria: List[Dict[str, Any]],
    verification_plan: List[str],
    risk_assessment: str,
) -> Tuple[Path, Dict[str, Any]]:
    queue_item_id = f"rci_{_safe_slug(case_id, lower=True, default='case')}_{_safe_slug(candidate_id, lower=True, default='candidate')}"

    payload: Dict[str, Any] = {
        "schema_version": "clawd.research_case.implementation_queue_item.v1",
        "queue_item_id": queue_item_id,
        "generated_at": now_iso(),
        "queue_state": "READY_FOR_EXECUTION",
        "case_id": case_id,
        "candidate_id": candidate_id,
        "promotion_id": promotion_id,
        "traceability": {
            "candidate_path": rel_to_root(candidate_json_path),
            "gatecheck_path": rel_to_root(gatecheck_path),
            "promotion_candidate_path": rel_to_root(promotion_candidate_path),
            "promotion_gate_decision_path": rel_to_root(promotion_gate_decision_path),
            "promotion_gate_decision_log_path": rel_to_root(promotion_gate_decision_log_path),
            "source_refs": source_refs,
        },
        "implementation_contract": {
            "requirements": requirements,
            "acceptance_criteria": acceptance_criteria,
            "verification_plan": verification_plan,
            "risk_assessment": risk_assessment,
        },
        "orchestration": {
            "policy_id": "b2_capacity_starvation_concurrency_v1",
            "scheduler_algorithm": "weighted_round_robin_with_aging",
            "queue_entered_at": now_iso(),
            "priority_weight": 1.0,
            "starvation_controls": {
                "max_runnable_wait_seconds": 900,
                "starvation_incident_wait_seconds": 1200,
                "max_consecutive_scheduler_skips": 3,
                "preemption_trigger_runtime_seconds": 1800,
                "required_fairness_metric": "oldest_runnable_case_wait_seconds",
            },
        },
    }

    b2_policy = _load_b2_capacity_policy()
    starvation_controls = (b2_policy.get("starvation_policy") if isinstance(b2_policy.get("starvation_policy"), dict) else {})
    payload_orchestration = payload.get("orchestration") if isinstance(payload.get("orchestration"), dict) else {}
    payload_orchestration["policy_id"] = str(b2_policy.get("policy_id") or payload_orchestration.get("policy_id") or "b2_capacity_starvation_concurrency_v1")
    scheduler_policy = b2_policy.get("scheduler_policy") if isinstance(b2_policy.get("scheduler_policy"), dict) else {}
    if scheduler_policy.get("algorithm"):
        payload_orchestration["scheduler_algorithm"] = str(scheduler_policy.get("algorithm"))
    payload_orchestration["starvation_controls"] = {
        "max_runnable_wait_seconds": int(starvation_controls.get("max_runnable_wait_seconds") or 900),
        "starvation_incident_wait_seconds": int(starvation_controls.get("starvation_incident_wait_seconds") or 1200),
        "max_consecutive_scheduler_skips": int(starvation_controls.get("max_consecutive_scheduler_skips") or 3),
        "preemption_trigger_runtime_seconds": int(starvation_controls.get("preemption_trigger_runtime_seconds") or 1800),
        "required_fairness_metric": str(starvation_controls.get("required_fairness_metric") or "oldest_runnable_case_wait_seconds"),
    }
    payload["orchestration"] = payload_orchestration

    queue_path = candidate_json_path.parent / "implementation_queue_item.json"
    atomic_write_json(queue_path, payload)

    # Optional schema self-check when available; fail closed only when schema exists and is malformed.
    if IMPLEMENTATION_QUEUE_SCHEMA_PATH.exists():
        try:
            import jsonschema  # type: ignore

            schema_obj = read_json(IMPLEMENTATION_QUEUE_SCHEMA_PATH, default={}) or {}
            if isinstance(schema_obj, dict) and schema_obj:
                jsonschema.validate(payload, schema_obj)
        except Exception as exc:
            raise SystemExit(f"implementation_queue_schema_validation_failed: {exc}")

    return queue_path, payload


def cmd_promote(args: argparse.Namespace) -> None:
    cid = args.case_id
    cdir = case_dir(cid)
    rc = load_case(cid)

    decision = _ensure_enum(args.decision, GATE_DECISIONS, "decision")

    synth_id = args.synth_id or ((rc.get("heads") or {}).get("synthesis_head_id"))
    if not synth_id:
        raise SystemExit("missing_synth_head: run record-synth first")

    synth_json_path = cdir / "SYNTH" / "synths" / f"{synth_id}.json"
    synth_payload = read_json(synth_json_path, default={}) or {}
    if not isinstance(synth_payload, dict) or not synth_payload:
        raise SystemExit(f"missing_synth_payload: {synth_json_path}")

    candidate_id = args.candidate_id.strip()
    if not candidate_id:
        raise SystemExit("missing_candidate_id")

    requirements = args.requirement or []
    if not requirements:
        raise SystemExit("at_least_one_requirement_required")

    takeaway_ids = [str(row.get("takeaway_id")) for row in (synth_payload.get("takeaways") or []) if isinstance(row, dict)]

    req_rows = []
    crit_rows = []
    for idx, text in enumerate(requirements, start=1):
        req_id = f"req_{idx:03d}"
        req_rows.append(
            {
                "req_id": req_id,
                "text": text,
                "takeaway_refs": takeaway_ids,
                "priority": "P1" if idx > 1 else "P0",
            }
        )
        crit_rows.append(
            {
                "criterion_id": f"ac_{idx:03d}",
                "statement": f"Requirement {req_id} is reflected in governed artifact + validation output.",
                "covers_req_ids": [req_id],
            }
        )

    verification_plan = [
        "Run research case lint in strict mode",
        "Confirm gatecheck decision + traceability links",
        "Generate checkpoint + handover artifact",
    ]

    candidate_payload = {
        "schema": "clawd.research_case.candidate.v1",
        "case_id": cid,
        "candidate_id": candidate_id,
        "derived_from_synthesis": synth_id,
        "status": "gated" if decision == "approved" else "draft",
        "readiness": args.readiness,
        "created_at": now_iso(),
        "created_by": args.created_by,
        "problem_statement": args.problem_statement,
        "proposed_change": args.proposed_change,
        "requirements": req_rows,
        "acceptance_criteria": crit_rows,
        "verification_plan": verification_plan,
        "risk_assessment": args.risk_assessment,
        "non_goals": args.non_goal or [],
    }

    cpath = cdir / "CANDIDATE" / "candidates" / candidate_id
    ensure_dir(cpath)
    candidate_json_path = cpath / "candidate.json"
    candidate_md_path = cpath / "candidate.md"
    gatecheck_path = cpath / "gatecheck.json"

    atomic_write_json(candidate_json_path, candidate_payload)

    md_lines = [
        f"# Candidate — {candidate_id}",
        "",
        f"- case_id: `{cid}`",
        f"- derived_from_synthesis: `{synth_id}`",
        f"- readiness: `{args.readiness}`",
        f"- decision_target: `{decision}`",
        "",
        "## Problem statement",
        args.problem_statement,
        "",
        "## Proposed change",
        args.proposed_change,
        "",
        "## Requirements",
    ]
    for row in req_rows:
        md_lines.append(f"- `{row['req_id']}` {row['text']}")
    write_text(candidate_md_path, "\n".join(md_lines) + "\n")

    promotion_id = (args.promotion_id or "").strip() or f"prom_{_safe_slug(candidate_id, lower=True, default='candidate')}"
    if not re.fullmatch(r"prom_[a-z0-9._-]+", promotion_id):
        raise SystemExit(f"invalid_promotion_id: {promotion_id}")

    review_state_map = {
        "approved": "approved",
        "rejected": "rejected",
        "needs_work": "pending",
    }
    promotion_state_map = {
        "approved": "APPROVED",
        "rejected": "REJECTED",
        "needs_work": "PENDING_REVIEW",
    }

    review_state = review_state_map.get(decision, "pending")
    review_role: Optional[str] = args.reviewer_role if review_state == "approved" else None
    review_id: Optional[str] = args.reviewer_id if review_state == "approved" else None
    reviewed_at: Optional[str] = now_iso() if review_state == "approved" else None

    source_refs = [
        _build_source_ref(f"src_{_safe_slug(cid, default='case')}_synth", synth_json_path, locator=f"$.synth_id:{synth_id}"),
        _build_source_ref(
            f"src_{_safe_slug(cid, default='case')}_candidate",
            candidate_json_path,
            locator=f"$.candidate_id:{candidate_id}",
        ),
    ]

    promotion_candidate_payload = {
        "promotion_id": promotion_id,
        "created_at": now_iso(),
        "promotion_state": promotion_state_map.get(decision, "PENDING_REVIEW"),
        "source_lane": {
            "lane_id": args.source_lane_id,
            "work_item_id": args.source_work_item_id or f"wi_research_case_{_safe_slug(cid, lower=True, default='case')}",
            "producer_role": args.source_producer_role,
            "session_key": args.source_session_key,
        },
        "insight": {
            "title": args.insight_title or f"Research case promotion candidate {candidate_id}",
            "statement": args.insight_statement or args.proposed_change,
            "kind": args.insight_kind,
        },
        "provenance": {
            "capture_method": args.capture_method,
            "captured_at": now_iso(),
            "tool_trace_refs": [f"research_case:{cid}", f"candidate:{candidate_id}"],
        },
        "confidence": {
            "score": float(args.confidence_score),
            "method": args.confidence_method,
            "notes": args.confidence_notes,
        },
        "source_refs": source_refs,
        "review": {
            "state": review_state,
            "reviewer_role": review_role,
            "reviewer_id": review_id,
            "reviewed_at": reviewed_at,
            "rationale": args.review_rationale,
        },
        "target": {
            "surface": args.target_surface,
            "target_path": args.target_path,
            "merge_mode": args.target_merge_mode,
        },
        "safety": {
            "classification": args.safety_classification,
            "leakage_check": args.safety_leakage_check,
            "redaction_applied": bool(args.safety_redaction_applied),
            "notes": args.safety_notes,
        },
        "decision_refs": [
            f"research_case:{cid}:promotion:{promotion_id}",
            f"candidate:{candidate_id}",
        ],
    }

    promotion_candidate_path = cpath / "promotion_candidate.json"
    promotion_gate_decision_path = cpath / "promotion_gate_decision.json"
    promotion_gate_decision_log_path = cpath / "promotion_gate_decisions.jsonl"
    publish_note_path = cpath / "promotion_publish_note.md"

    atomic_write_json(promotion_candidate_path, promotion_candidate_payload)
    write_text(
        publish_note_path,
        "\n".join(
            [
                f"# Promotion publish note — {promotion_id}",
                "",
                f"- case_id: `{cid}`",
                f"- candidate_id: `{candidate_id}`",
                f"- promotion_id: `{promotion_id}`",
                f"- review_state: `{review_state}`",
                "",
                "Traceability: promotion_id must be present for publish-gate validation.",
                "",
            ]
        ),
    )

    promotion_gate = _run_promotion_gate(
        promotion_candidate_path,
        decision_log_path=promotion_gate_decision_log_path,
        publish_note_path=publish_note_path,
    )
    atomic_write_json(promotion_gate_decision_path, promotion_gate)

    gate_decision = "needs_work"
    if str(promotion_gate.get("decision") or "").upper() == "PASS":
        gate_decision = "approved"
    elif decision == "rejected":
        gate_decision = "rejected"

    traceability_pass = bool(req_rows and takeaway_ids and synth_id == ((rc.get("heads") or {}).get("synthesis_head_id")))
    checks = [
        {
            "name": "evidence_traceability",
            "pass": traceability_pass,
            "findings": "requirements reference synthesis takeaway IDs",
        },
        {
            "name": "supersession_head_alignment",
            "pass": synth_id == ((rc.get("heads") or {}).get("synthesis_head_id")),
            "findings": "candidate derived from active synthesis head",
        },
        {
            "name": "acceptance_criteria_present",
            "pass": len(crit_rows) > 0,
            "findings": "acceptance criteria generated",
        },
        {
            "name": "promotion_contract_gate",
            "pass": gate_decision == "approved",
            "findings": f"promotion_gate_runner decision={promotion_gate.get('decision')} block_reason={promotion_gate.get('block_reason')}",
        },
    ]

    gate_payload = {
        "schema": "clawd.research_case.gatecheck.v1",
        "case_id": cid,
        "candidate_id": candidate_id,
        "gate_id": f"gate_{candidate_id}",
        "stage_transition": "synthesis_complete->promotion_ready",
        "run_at": now_iso(),
        "run_by": args.created_by,
        "requested_review_decision": decision,
        "checks": checks,
        "decision": gate_decision,
        "promotion_contract": {
            "promotion_id": promotion_id,
            "promotion_candidate_path": rel_to_root(promotion_candidate_path),
            "publish_note_path": rel_to_root(publish_note_path),
            "gate_decision_path": rel_to_root(promotion_gate_decision_path),
            "gate_decision_log_path": rel_to_root(promotion_gate_decision_log_path),
            "gate_decision": promotion_gate.get("decision"),
            "gate_final_state": promotion_gate.get("final_state"),
            "gate_block_gate": promotion_gate.get("block_gate"),
            "gate_block_reason": promotion_gate.get("block_reason"),
        },
    }
    atomic_write_json(gatecheck_path, gate_payload)

    implementation_queue_path: Optional[Path] = None
    implementation_queue_payload: Optional[Dict[str, Any]] = None
    if gate_decision == "approved":
        implementation_queue_path, implementation_queue_payload = _emit_implementation_queue_item(
            case_id=cid,
            candidate_id=candidate_id,
            promotion_id=promotion_id,
            candidate_json_path=candidate_json_path,
            gatecheck_path=gatecheck_path,
            promotion_candidate_path=promotion_candidate_path,
            promotion_gate_decision_path=promotion_gate_decision_path,
            promotion_gate_decision_log_path=promotion_gate_decision_log_path,
            source_refs=source_refs,
            requirements=req_rows,
            acceptance_criteria=crit_rows,
            verification_plan=verification_plan,
            risk_assessment=args.risk_assessment,
        )

    heads = rc.get("heads") or {}
    heads["candidate_head_id"] = candidate_id
    rc["heads"] = heads

    lifecycle = rc.get("lifecycle") or {}
    lifecycle["promotion_state"] = "promoted" if gate_decision == "approved" else "gated"
    rc["lifecycle"] = lifecycle

    rc["primary_state"] = "promotion_ready" if gate_decision == "approved" else "synthesis_complete"

    promotion = rc.get("promotion") or {}
    promotion["gate_status"] = gate_decision
    promotion["last_decision"] = gate_decision
    promotion["last_requested_review_decision"] = decision
    promotion["last_gatecheck_path"] = f"CANDIDATE/candidates/{candidate_id}/gatecheck.json"
    promotion["last_candidate_id"] = candidate_id
    promotion["last_promotion_id"] = promotion_id
    promotion["last_promotion_candidate_path"] = f"CANDIDATE/candidates/{candidate_id}/promotion_candidate.json"
    promotion["last_promotion_gate_decision_path"] = f"CANDIDATE/candidates/{candidate_id}/promotion_gate_decision.json"
    promotion["last_promotion_gate_result"] = str(promotion_gate.get("decision") or "")
    promotion["last_implementation_queue_item_path"] = (
        f"CANDIDATE/candidates/{candidate_id}/implementation_queue_item.json" if implementation_queue_path else None
    )
    rc["promotion"] = promotion

    write_case(cid, rc)

    append_jsonl(
        cdir / "LEDGER" / "events.jsonl",
        {
            "ts": now_iso(),
            "event": "candidate_promoted",
            "case_id": cid,
            "candidate_id": candidate_id,
            "decision": gate_decision,
            "requested_review_decision": decision,
            "promotion_gate_decision": promotion_gate.get("decision"),
            "promotion_id": promotion_id,
            "synth_id": synth_id,
            "requirement_count": len(req_rows),
            "implementation_queue_item_path": rel_to_root(implementation_queue_path) if implementation_queue_path else None,
        },
    )

    if implementation_queue_path is not None:
        append_jsonl(
            cdir / "LEDGER" / "events.jsonl",
            {
                "ts": now_iso(),
                "event": "implementation_queue_item_emitted",
                "case_id": cid,
                "candidate_id": candidate_id,
                "queue_item_path": rel_to_root(implementation_queue_path),
                "queue_state": (implementation_queue_payload or {}).get("queue_state"),
            },
        )

    if gate_decision == "approved":
        next_actions = [
            "Consume implementation_queue_item.json in IMPLEMENT/slices execution lane",
            "Open first bounded implementation task with linked traceability refs",
        ]
        notes = "Candidate gatecheck generated with promotion contract gate PASS and implementation queue item emitted."
    else:
        next_actions = ["Fix promotion packet blockers", "Re-run promote command after review/provenance updates"]
        notes = "Candidate gatecheck generated with promotion contract gate BLOCK (fail-closed)."

    cp_paths = write_checkpoint(
        cid,
        rc,
        next_actions=next_actions,
        do_not_do=["Do not mark shipped/closed before verification artifacts exist"],
        notes=notes,
    )

    emit(
        {
            "status": "ok",
            "case_id": cid,
            "candidate_id": candidate_id,
            "decision": gate_decision,
            "requested_review_decision": decision,
            "paths": {
                "candidate_json": rel_to_root(candidate_json_path),
                "candidate_md": rel_to_root(candidate_md_path),
                "gatecheck_json": rel_to_root(gatecheck_path),
                "promotion_candidate_json": rel_to_root(promotion_candidate_path),
                "promotion_gate_decision_json": rel_to_root(promotion_gate_decision_path),
                "promotion_gate_decisions_jsonl": rel_to_root(promotion_gate_decision_log_path),
                "promotion_publish_note_md": rel_to_root(publish_note_path),
                "implementation_queue_item_json": rel_to_root(implementation_queue_path) if implementation_queue_path else None,
                **cp_paths,
            },
            "summary": {
                "requirements": len(req_rows),
                "promotion_gate_decision": promotion_gate.get("decision"),
                "implementation_queue_state": (implementation_queue_payload or {}).get("queue_state") if implementation_queue_payload else None,
                "state": rc.get("primary_state"),
            },
        },
        as_json=args.json,
    )


def cmd_orchestrate_capacity(args: argparse.Namespace) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    cpu_count = int(args.cpu_count or os.cpu_count() or 1)
    memory_gib = float(args.memory_gib)

    node_class = _classify_node(cpu_count=cpu_count, memory_gib=memory_gib)
    policy = _load_b2_capacity_policy()

    runtime_path = Path(args.runtime_path).expanduser()
    if not runtime_path.is_absolute():
        runtime_path = (ROOT / runtime_path).resolve()

    previous_runtime = read_json(runtime_path, default={}) or {}
    ready_items = _collect_ready_implementation_items()

    payload = _plan_capacity_orchestration(
        now=now,
        node_class=node_class,
        policy=policy,
        active_case_ids=args.active_case_id or [],
        ready_items=ready_items,
        previous_runtime=previous_runtime if isinstance(previous_runtime, dict) else {},
    )

    atomic_write_json(runtime_path, payload)

    if args.history_path:
        history_path = Path(args.history_path).expanduser()
        if not history_path.is_absolute():
            history_path = (ROOT / history_path).resolve()
        append_jsonl(history_path, payload)

    emit(
        {
            "status": "ok",
            "runtime_path": rel_to_root(runtime_path),
            "summary": {
                "node_class": payload.get("node_class"),
                "selected": len(payload.get("selected") or []),
                "runnable_pool": len(payload.get("runnable_pool") or []),
                "alerts": ",".join(payload.get("alerts") or []),
            },
            "payload": payload,
        },
        as_json=args.json,
    )


def cmd_replay_batch(args: argparse.Namespace) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    cpu_count = int(args.cpu_count or os.cpu_count() or 1)
    memory_gib = float(args.memory_gib)

    node_class = _classify_node(cpu_count=cpu_count, memory_gib=memory_gib)
    policy = _load_b2_capacity_policy()

    runtime_path = Path(args.runtime_path).expanduser()
    if not runtime_path.is_absolute():
        runtime_path = (ROOT / runtime_path).resolve()

    queue_states = {str(state).strip().upper() for state in (args.queue_state or []) if str(state).strip()}
    if not queue_states:
        queue_states = {"READY_FOR_EXECUTION", "IN_EXECUTION", "DONE", "BLOCKED"}

    candidates = _collect_implementation_items(queue_states=queue_states)

    queue_item_filter = {str(x).strip() for x in (args.queue_item_id or []) if str(x).strip()}
    case_filter = {str(x).strip() for x in (args.case_id or []) if str(x).strip()}
    candidate_filter = {str(x).strip() for x in (args.candidate_id or []) if str(x).strip()}

    filtered: List[Dict[str, Any]] = []
    for row in candidates:
        if queue_item_filter and str(row.get("queue_item_id") or "") not in queue_item_filter:
            continue
        if case_filter and str(row.get("case_id") or "") not in case_filter:
            continue
        if candidate_filter and str(row.get("candidate_id") or "") not in candidate_filter:
            continue
        filtered.append(row)

    replay_payload = _plan_batch_replay(
        now=now,
        node_class=node_class,
        policy=policy,
        replay_candidates=filtered,
        replay_seed=str(args.replay_seed),
        max_replays_override=int(args.max_replays or 0),
        quality_gate=float(args.quality_gate),
        min_quality_delta=float(args.min_quality_delta),
        tuning_bias=float(args.tuning_bias),
        replay_target_state=str(args.replay_target_state),
    )

    atomic_write_json(runtime_path, replay_payload)

    if args.history_path:
        history_path = Path(args.history_path).expanduser()
        if not history_path.is_absolute():
            history_path = (ROOT / history_path).resolve()
        append_jsonl(history_path, replay_payload)

    emit(
        {
            "status": "ok",
            "runtime_path": rel_to_root(runtime_path),
            "summary": {
                "cohort_size": replay_payload.get("limits", {}).get("cohort_size"),
                "selected_count": replay_payload.get("summary", {}).get("selected_count"),
                "pass_count": replay_payload.get("summary", {}).get("pass_count"),
                "fail_count": replay_payload.get("summary", {}).get("fail_count"),
                "alerts": ",".join(replay_payload.get("alerts") or []),
            },
            "payload": replay_payload,
        },
        as_json=args.json,
    )


def cmd_checkpoint(args: argparse.Namespace) -> None:
    rc = load_case(args.case_id)
    paths = write_checkpoint(
        args.case_id,
        rc,
        next_actions=args.next_action,
        do_not_do=args.do_not_do,
        notes=args.notes or "manual checkpoint",
    )
    emit(
        {
            "status": "ok",
            "case_id": args.case_id,
            "paths": paths,
            "summary": {
                "state": rc.get("primary_state"),
            },
        },
        as_json=args.json,
    )


def _lint_case(rc: Dict[str, Any], cdir: Path) -> Dict[str, Any]:
    violations: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if rc.get("primary_state") not in PRIMARY_STATES:
        violations.append({"kind": "invalid_primary_state", "value": rc.get("primary_state")})

    lifecycle = rc.get("lifecycle") or {}
    if lifecycle.get("reading_state") not in READING_STATES:
        violations.append({"kind": "invalid_reading_state", "value": lifecycle.get("reading_state")})
    if lifecycle.get("synthesis_state") not in SYNTHESIS_STATES:
        violations.append({"kind": "invalid_synthesis_state", "value": lifecycle.get("synthesis_state")})
    if lifecycle.get("promotion_state") not in PROMOTION_STATES:
        violations.append({"kind": "invalid_promotion_state", "value": lifecycle.get("promotion_state")})
    if lifecycle.get("disposition") not in DISPOSITIONS:
        violations.append({"kind": "invalid_disposition", "value": lifecycle.get("disposition")})

    flags = rc.get("reliability_flags") or {}
    if flags.get("understanding_level") not in UNDERSTANDING_LEVELS:
        violations.append({"kind": "invalid_understanding_level", "value": flags.get("understanding_level")})
    if flags.get("work_status") not in WORK_STATUS:
        violations.append({"kind": "invalid_work_status", "value": flags.get("work_status")})
    if flags.get("freshness") not in FRESHNESS:
        violations.append({"kind": "invalid_freshness", "value": flags.get("freshness")})

    synth_head_id = ((rc.get("heads") or {}).get("synthesis_head_id"))
    if synth_head_id:
        synth_path = cdir / "SYNTH" / "synths" / f"{synth_head_id}.json"
        if not synth_path.exists():
            violations.append({"kind": "missing_synthesis_head_payload", "path": rel_to_root(synth_path)})

    candidate_head_id = ((rc.get("heads") or {}).get("candidate_head_id"))
    if candidate_head_id:
        cpath = cdir / "CANDIDATE" / "candidates" / candidate_head_id
        for rel_name in ["candidate.json", "gatecheck.json", "promotion_candidate.json", "promotion_gate_decision.json"]:
            fp = cpath / rel_name
            if not fp.exists():
                violations.append({"kind": "missing_candidate_contract", "path": rel_to_root(fp)})

    if (rc.get("lifecycle") or {}).get("promotion_state") == "promoted":
        gate_status = (rc.get("promotion") or {}).get("gate_status")
        if gate_status != "approved":
            violations.append({"kind": "promoted_without_approved_gate", "gate_status": gate_status})

    if not (cdir / "RAW" / "raw_manifest.json").exists():
        violations.append({"kind": "missing_raw_manifest"})
    if not (cdir / "LEDGER" / "events.jsonl").exists():
        warnings.append({"kind": "missing_event_ledger"})

    status = "pass"
    if violations:
        status = "fail"
    elif warnings:
        status = "warn"

    return {
        "status": status,
        "ok": status != "fail",
        "summary": {
            "violations": len(violations),
            "warnings": len(warnings),
        },
        "violations": violations,
        "warnings": warnings,
    }


def cmd_lint(args: argparse.Namespace) -> None:
    cid = args.case_id
    cdir = case_dir(cid)
    rc = load_case(cid)
    out = _lint_case(rc, cdir)
    out["case_id"] = cid
    if args.strict and out["status"] != "pass":
        emit(out, as_json=args.json)
        raise SystemExit(1)
    emit(out, as_json=args.json)


def cmd_status(args: argparse.Namespace) -> None:
    cid = args.case_id
    cdir = case_dir(cid)
    rc = load_case(cid)

    lint = _lint_case(rc, cdir)

    payload = {
        "schema": "clawd.research_case.status.v1",
        "generated_at": now_iso(),
        "case_id": cid,
        "case_path": rel_to_root(cdir),
        "primary_state": rc.get("primary_state"),
        "lifecycle": rc.get("lifecycle"),
        "reliability_flags": rc.get("reliability_flags"),
        "heads": rc.get("heads"),
        "promotion": rc.get("promotion"),
        "lint": lint,
    }

    if args.publish:
        latest_dir = ROOT / "state" / "continuity" / "latest"
        ensure_dir(latest_dir)
        case_latest_path = latest_dir / f"research_case_{cid}.json"
        atomic_write_json(case_latest_path, payload)

        registry = read_json(REGISTRY_PATH, default={}) or {}
        entries = registry.get("entries") if isinstance(registry.get("entries"), list) else []

        updated = False
        for row in entries:
            if str(row.get("case_id")) == cid:
                row.update(
                    {
                        "case_id": cid,
                        "primary_state": payload["primary_state"],
                        "updated_at": payload["generated_at"],
                        "case_latest_path": rel_to_root(case_latest_path),
                    }
                )
                updated = True
                break

        if not updated:
            entries.append(
                {
                    "case_id": cid,
                    "primary_state": payload["primary_state"],
                    "updated_at": payload["generated_at"],
                    "case_latest_path": rel_to_root(case_latest_path),
                }
            )

        entries = sorted(entries, key=lambda r: str(r.get("updated_at", "")), reverse=True)

        registry_payload = {
            "schema": "clawd.research_case.registry.v1",
            "generated_at": payload["generated_at"],
            "entry_count": len(entries),
            "entries": entries,
        }
        atomic_write_json(REGISTRY_PATH, registry_payload)
        payload["published"] = {
            "case_latest_path": rel_to_root(case_latest_path),
            "registry_path": rel_to_root(REGISTRY_PATH),
        }

    emit(payload, as_json=args.json)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Research Case pipeline MVP")
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="initialize a Research Case scaffold")
    p_init.add_argument("--case-id", required=True)
    p_init.add_argument("--title", required=True)
    p_init.add_argument("--intent", required=True)
    p_init.add_argument("--source", action="append", help="source file/dir path", default=[])
    p_init.add_argument("--force", action="store_true")
    p_init.add_argument("--json", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_synth = sub.add_parser("record-synth", help="record synthesis artifact and advance state")
    p_synth.add_argument("--case-id", required=True)
    p_synth.add_argument("--synth-id", required=True)
    p_synth.add_argument("--takeaway", action="append", default=[])
    p_synth.add_argument("--evidence-source-id", action="append", default=[])
    p_synth.add_argument("--open-question", action="append", default=[])
    p_synth.add_argument("--summary-file")
    p_synth.add_argument("--summary-text", default="")
    p_synth.add_argument("--created-by", default="architect")
    p_synth.add_argument("--reading-state", default="partial")
    p_synth.add_argument("--understanding-level", default="substantial")
    p_synth.add_argument("--partial-reason-code", default="timebox_remaining")
    p_synth.add_argument("--complete", action="store_true")
    p_synth.add_argument("--json", action="store_true")
    p_synth.set_defaults(func=cmd_record_synth)

    p_promote = sub.add_parser("promote", help="promote synthesis into candidate + gatecheck")
    p_promote.add_argument("--case-id", required=True)
    p_promote.add_argument("--candidate-id", required=True)
    p_promote.add_argument("--synth-id")
    p_promote.add_argument("--decision", default="approved")
    p_promote.add_argument("--requirement", action="append", default=[])
    p_promote.add_argument("--problem-statement", default="Need explicit governed promotion substrate for research-driven implementation work.")
    p_promote.add_argument("--proposed-change", default="Adopt Research Case contract + lifecycle metadata + gatechecked candidate promotion.")
    p_promote.add_argument("--risk-assessment", default="Low-to-medium: additive artifacts + metadata discipline, no broad runtime mutation.")
    p_promote.add_argument("--non-goal", action="append", default=[])
    p_promote.add_argument("--readiness", default="ready_for_implementation")
    p_promote.add_argument("--created-by", default="architect")
    p_promote.add_argument("--promotion-id", default="")
    p_promote.add_argument("--source-lane-id", default="lane.column_b.swarm_orchestration")
    p_promote.add_argument("--source-work-item-id", default="")
    p_promote.add_argument(
        "--source-producer-role",
        default="EXECUTOR",
        choices=["PLANNER", "EXECUTOR", "VALIDATOR", "RESEARCHER", "SRE", "LIBRARIAN"],
    )
    p_promote.add_argument("--source-session-key", default="research_case_pipeline")
    p_promote.add_argument("--insight-title", default="")
    p_promote.add_argument("--insight-statement", default="")
    p_promote.add_argument("--insight-kind", default="procedure", choices=["fact", "rule", "procedure", "heuristic"])
    p_promote.add_argument("--capture-method", default="research_case_pipeline_promote")
    p_promote.add_argument("--confidence-score", type=float, default=0.82)
    p_promote.add_argument("--confidence-method", default="research_case_synthesis_traceability")
    p_promote.add_argument("--confidence-notes", default="meets doctrine threshold by default workflow policy")
    p_promote.add_argument(
        "--reviewer-role",
        default="VALIDATOR",
        choices=["PLANNER", "EXECUTOR", "VALIDATOR", "RESEARCHER", "SRE", "LIBRARIAN"],
    )
    p_promote.add_argument("--reviewer-id", default="validator_research_case_pipeline")
    p_promote.add_argument("--review-rationale", default="Research-case promotion packet reviewed under Wave 4 contract gate policy.")
    p_promote.add_argument("--target-surface", default="doctrine", choices=["doctrine", "memory", "playbook"])
    p_promote.add_argument("--target-path", default="docs/ops/unified_operating_doctrine_v1.md")
    p_promote.add_argument("--target-merge-mode", default="append", choices=["append", "patch", "replace_section"])
    p_promote.add_argument("--safety-classification", default="internal", choices=["public", "internal", "restricted", "secret"])
    p_promote.add_argument("--safety-leakage-check", default="pass", choices=["pass", "fail"])
    p_promote.add_argument("--safety-redaction-applied", action="store_true")
    p_promote.add_argument("--safety-notes", default="research case packet contains no secret-class content")
    p_promote.add_argument("--json", action="store_true")
    p_promote.set_defaults(func=cmd_promote)

    p_sched = sub.add_parser("orchestrate-capacity", help="plan deterministic multi-case capacity orchestration for B2 implementation queue")
    p_sched.add_argument("--cpu-count", type=int, default=0, help="node CPU count (defaults to local os.cpu_count)")
    p_sched.add_argument("--memory-gib", type=float, default=16.0, help="node memory in GiB for node-class cap selection")
    p_sched.add_argument("--active-case-id", action="append", default=[], help="case_id currently in execution (repeatable)")
    p_sched.add_argument(
        "--runtime-path",
        default="state/continuity/latest/research_case_capacity_orchestration_runtime.json",
        help="where to persist latest orchestration runtime payload",
    )
    p_sched.add_argument(
        "--history-path",
        default="state/continuity/history/research_case_capacity_orchestration_runtime.jsonl",
        help="append-only history jsonl path (set empty string to disable)",
    )
    p_sched.add_argument("--json", action="store_true")
    p_sched.set_defaults(func=cmd_orchestrate_capacity)

    p_replay = sub.add_parser("replay-batch", help="run bounded deterministic B2 batch-level replay automation for queue cohorts")
    p_replay.add_argument("--cpu-count", type=int, default=0, help="node CPU count (defaults to local os.cpu_count)")
    p_replay.add_argument("--memory-gib", type=float, default=16.0, help="node memory in GiB for node-class cap selection")
    p_replay.add_argument(
        "--queue-state",
        action="append",
        default=[],
        help="queue states to include in replay cohort (repeatable; defaults to READY_FOR_EXECUTION,IN_EXECUTION,DONE,BLOCKED)",
    )
    p_replay.add_argument("--queue-item-id", action="append", default=[], help="queue_item_id filter (repeatable)")
    p_replay.add_argument("--case-id", action="append", default=[], help="case_id filter (repeatable)")
    p_replay.add_argument("--candidate-id", action="append", default=[], help="candidate_id filter (repeatable)")
    p_replay.add_argument("--max-replays", type=int, default=0, help="optional hard bound override for selected replays")
    p_replay.add_argument("--replay-seed", default="b2_batch_replay_seed_v1", help="deterministic replay seed")
    p_replay.add_argument("--quality-gate", type=float, default=0.72, help="minimum candidate quality score to pass")
    p_replay.add_argument("--min-quality-delta", type=float, default=0.01, help="minimum quality delta over baseline to pass")
    p_replay.add_argument("--tuning-bias", type=float, default=0.02, help="candidate tuning bias applied during replay scoring")
    p_replay.add_argument("--replay-target-state", default="REPLAY_VALIDATED", help="target queue state projection for successful replay")
    p_replay.add_argument(
        "--runtime-path",
        default="state/continuity/latest/research_case_batch_replay_runtime.json",
        help="where to persist latest replay runtime payload",
    )
    p_replay.add_argument(
        "--history-path",
        default="state/continuity/history/research_case_batch_replay_runtime.jsonl",
        help="append-only history jsonl path (set empty string to disable)",
    )
    p_replay.add_argument("--json", action="store_true")
    p_replay.set_defaults(func=cmd_replay_batch)

    p_cp = sub.add_parser("checkpoint", help="write checkpoint/handover artifacts")
    p_cp.add_argument("--case-id", required=True)
    p_cp.add_argument("--next-action", action="append", default=[])
    p_cp.add_argument("--do-not-do", action="append", default=[])
    p_cp.add_argument("--notes", default="")
    p_cp.add_argument("--json", action="store_true")
    p_cp.set_defaults(func=cmd_checkpoint)

    p_lint = sub.add_parser("lint", help="lint contract and lifecycle invariants")
    p_lint.add_argument("--case-id", required=True)
    p_lint.add_argument("--strict", action="store_true")
    p_lint.add_argument("--json", action="store_true")
    p_lint.set_defaults(func=cmd_lint)

    p_status = sub.add_parser("status", help="show status and optionally publish latest registry")
    p_status.add_argument("--case-id", required=True)
    p_status.add_argument("--publish", action="store_true")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_dir(CASES_ROOT)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
