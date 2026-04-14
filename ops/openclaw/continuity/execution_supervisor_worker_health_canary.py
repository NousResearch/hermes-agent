#!/usr/bin/env python3
"""Build execution-supervisor worker-health canary evidence artifact.

This producer closes the gap between dispatch-qualification health gates and
runtime evidence availability by materializing:
- state/continuity/latest/execution_supervisor_worker_health_canary_latest.json
- state/continuity/history/execution_supervisor_worker_health_canary_history.jsonl

Primary signal source is codex_dispatch_health.py. Route workers not covered by
that source are projected explicitly with deterministic fallback posture.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCHEMA = "clawd.execution_supervisor_worker_health_canary.v1"

DEFAULT_ROUTE_WORKERS: List[str] = [
    "codex-worker-plus-11",
    "gemini-pro",
    "deepseek-reasoner",
    "gemini-flash-lite",
    "gemini-flash",
    "deepseek-chat",
    "kimi-k2",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(value: Any) -> Optional[dt.datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except Exception:
        return None


def to_rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def ensure_within(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(root.resolve())
    except Exception as exc:
        raise ValueError(f"path_outside_repo:{raw_path}") from exc
    return candidate


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("json_top_level_not_object")
    return obj


def run_dispatch_health(repo_root: Path, lookback_hours: float, *, include_orchestrator: bool = False) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(repo_root / "ops" / "openclaw" / "codex_dispatch_health.py"),
        "--lookback-hours",
        str(max(0.0, lookback_hours)),
    ]
    if include_orchestrator:
        cmd.append("--include-orchestrator")
    cp = subprocess.run(cmd, text=True, capture_output=True, check=False, cwd=str(repo_root))
    if cp.returncode != 0:
        raise RuntimeError((cp.stderr or cp.stdout or "dispatch_health_failed").strip())
    payload = json.loads(cp.stdout)
    if not isinstance(payload, dict):
        raise ValueError("dispatch_health_payload_not_object")
    return payload


def discover_route_workers(qualification_payload: Dict[str, Any]) -> Set[str]:
    discovered: Set[str] = set()

    def add_worker(raw: Any) -> None:
        worker = str(raw or "").strip()
        if worker:
            discovered.add(worker)

    for key in ("evaluated_candidates", "qualified_candidates", "blocked_candidates", "ready_candidates"):
        rows = qualification_payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            add_worker(row.get("target_worker"))
            route_workers = row.get("route_workers")
            if isinstance(route_workers, list):
                for worker in route_workers:
                    add_worker(worker)
            allocation_order = row.get("allocation_order")
            if isinstance(allocation_order, list):
                for worker in allocation_order:
                    add_worker(worker)

    schedule = qualification_payload.get("canary_probe_schedule")
    if isinstance(schedule, dict):
        workers = schedule.get("workers")
        if isinstance(workers, list):
            for row in workers:
                if isinstance(row, dict):
                    add_worker(row.get("worker"))

    return discovered


def discover_orchestrator_agents(*, repo_root: Path, dispatch_health_payload: Dict[str, Any]) -> Set[str]:
    raw_manifest_path = str(dispatch_health_payload.get("manifestPath") or "").strip()
    if not raw_manifest_path:
        return set()

    manifest_path = Path(raw_manifest_path).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    else:
        manifest_path = manifest_path.resolve()

    try:
        manifest = load_json(manifest_path)
    except Exception:
        return set()

    orchestrator = manifest.get("orchestrator")
    if not isinstance(orchestrator, dict):
        return set()
    agent = str(orchestrator.get("agent") or "").strip()
    if not agent:
        return set()
    return {agent}


def normalize_reason(row: Dict[str, Any]) -> Optional[str]:
    for key in ("quarantineReasons", "probationReasons"):
        values = row.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            token = str(value or "").strip()
            if token:
                return token
    return None


def _normalize_reason_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for value in values:
        token = str(value or "").strip()
        if token:
            out.append(token)
    return out


def _dispatch_status_counts_from_lanes(dispatch_lanes: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"healthy": 0, "probationary": 0, "quarantined": 0}
    for lane in dispatch_lanes:
        status = str(lane.get("status") or "").strip().lower()
        if status in counts:
            counts[status] += 1
    return counts


def _classify_active_pool_posture(status_counts: Dict[str, int]) -> Tuple[str, str]:
    healthy = max(0, int(status_counts.get("healthy") or 0))
    probationary = max(0, int(status_counts.get("probationary") or 0))
    quarantined = max(0, int(status_counts.get("quarantined") or 0))
    total = healthy + probationary + quarantined
    if total <= 0:
        return "red", "active_pool_missing"
    if quarantined > 0:
        return "red", "active_pool_quarantined_workers_present"
    if probationary > 0:
        return "yellow", "active_pool_probationary_workers_present"
    return "green", "active_pool_all_workers_healthy"


def _build_dispatch_health_matrix(
    *,
    generated_at: str,
    dispatch_generated_at: str,
    dispatch_health_source: str,
    dispatch_health_error: Optional[str],
    dispatch_artifact_rel: str,
    dispatch_lanes: List[Dict[str, Any]],
    routine_dispatch_lane_disable_statuses: Optional[List[str]] = None,
    quota_exhaustion_reason_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    normalized_disable_statuses: List[str] = []
    for raw in routine_dispatch_lane_disable_statuses or []:
        token = str(raw or "").strip().lower()
        if token and token not in normalized_disable_statuses:
            normalized_disable_statuses.append(token)
    if not normalized_disable_statuses:
        normalized_disable_statuses = ["probationary", "quarantined"]

    normalized_quota_prefixes: List[str] = []
    for raw in quota_exhaustion_reason_prefixes or []:
        token = str(raw or "").strip().lower()
        if token and token not in normalized_quota_prefixes:
            normalized_quota_prefixes.append(token)
    if not normalized_quota_prefixes:
        normalized_quota_prefixes = [
            "quota_exhausted",
            "quota_exhausted_additional",
            "runtime_bodyless_usage_limit",
        ]

    workers: Dict[str, Dict[str, Any]] = {}
    routine_dispatch_disabled_workers: List[Dict[str, Any]] = []
    for lane in dispatch_lanes:
        if not isinstance(lane, dict):
            continue
        worker = str(lane.get("agent") or "").strip()
        if not worker:
            continue
        status_token = str(lane.get("status") or "unknown").strip().lower() or "unknown"
        quarantine_reasons = _normalize_reason_list(lane.get("quarantineReasons"))
        probation_reasons = _normalize_reason_list(lane.get("probationReasons"))
        disable_reasons = []
        for reason in quarantine_reasons + probation_reasons:
            if reason not in disable_reasons:
                disable_reasons.append(reason)

        workers[worker] = {
            "status": status_token,
            "quarantine_reasons": quarantine_reasons,
            "probation_reasons": probation_reasons,
            "ready": lane.get("ready") if isinstance(lane.get("ready"), bool) else None,
            "binding_ambiguous": bool(lane.get("bindingAmbiguous") is True),
            "effective_ambiguity_reasons": _normalize_reason_list(lane.get("effectiveAmbiguityReasons")),
            "expected_email": lane.get("expectedEmail"),
            "desired_account": lane.get("desiredAccount") if isinstance(lane.get("desiredAccount"), dict) else None,
            "shared_account_allowlisted": bool(lane.get("sharedAccountAllowlisted") is True),
            "recommended_actions": lane.get("recommendedActions") if isinstance(lane.get("recommendedActions"), list) else [],
        }

        if status_token in normalized_disable_statuses:
            quota_exhausted_signal = any(
                any(reason.startswith(prefix) for prefix in normalized_quota_prefixes)
                for reason in disable_reasons
            )
            routine_dispatch_disabled_workers.append(
                {
                    "worker": worker,
                    "status": status_token,
                    "disable_reasons": disable_reasons,
                    "quota_exhausted_signal": quota_exhausted_signal,
                }
            )

    counts = _dispatch_status_counts_from_lanes(dispatch_lanes)
    return {
        "schema": "clawd.dispatch_health_matrix.v1",
        "generated_at": generated_at,
        "dispatch_generated_at": dispatch_generated_at,
        "source": {
            "dispatch_health_path": dispatch_artifact_rel,
            "dispatch_health_source": dispatch_health_source,
            "dispatch_health_source_error": dispatch_health_error,
            "dispatch_health_lane_count": len(workers),
        },
        "counts": counts,
        "routine_dispatch_lane_disable_policy": {
            "lane_disable_statuses": normalized_disable_statuses,
            "quota_exhaustion_reason_prefixes": normalized_quota_prefixes,
        },
        "routine_dispatch_disabled_worker_count": len(routine_dispatch_disabled_workers),
        "routine_dispatch_disabled_workers": routine_dispatch_disabled_workers,
        "workers": workers,
    }


def worker_row_from_dispatch_lane(
    *,
    worker: str,
    lane_row: Dict[str, Any],
    checked_at: str,
    dispatch_health_artifact_rel: str,
) -> Dict[str, Any]:
    status = str(lane_row.get("status") or "").strip().lower()
    reason = normalize_reason(lane_row)

    if status == "healthy":
        return {
            "worker": worker,
            "health_status": "healthy",
            "canary_status": "pass",
            "reason": reason or "dispatch_health_healthy",
            "checked_at": checked_at,
            "canary_reason": reason or "dispatch_health_healthy",
            "canary_checked_at": checked_at,
            "canary_artifact_path": dispatch_health_artifact_rel,
            "probe_status": "pass",
            "probe_reason": reason or "dispatch_health_healthy",
            "probe_checked_at": checked_at,
            "probe_artifact_path": dispatch_health_artifact_rel,
        }

    if status == "probationary":
        return {
            "worker": worker,
            "health_status": "probationary",
            "canary_status": "required",
            "reason": reason or "dispatch_health_probationary",
            "checked_at": checked_at,
            "canary_reason": reason or "dispatch_health_probationary",
            "canary_checked_at": checked_at,
            "canary_artifact_path": None,
            "probe_status": "required",
            "probe_reason": reason or "dispatch_health_probationary",
            "probe_checked_at": checked_at,
            "probe_artifact_path": None,
        }

    return {
        "worker": worker,
        "health_status": "quarantined",
        "canary_status": "fail",
        "reason": reason or "dispatch_health_quarantined",
        "checked_at": checked_at,
        "canary_reason": reason or "dispatch_health_quarantined",
        "canary_checked_at": checked_at,
        "canary_artifact_path": None,
        "probe_status": "required",
        "probe_reason": reason or "dispatch_health_quarantined",
        "probe_checked_at": checked_at,
        "probe_artifact_path": None,
    }


def fallback_worker_row(
    *,
    worker: str,
    checked_at: str,
    canary_artifact_rel: str,
    dispatch_health_ready: bool,
) -> Tuple[Dict[str, Any], str]:
    if dispatch_health_ready:
        reason = "worker_health_unmanaged_route_default_assumed_healthy"
        return (
            {
                "worker": worker,
                "health_status": "healthy",
                "canary_status": "pass",
                "reason": reason,
                "checked_at": checked_at,
                "canary_reason": reason,
                "canary_checked_at": checked_at,
                "canary_artifact_path": canary_artifact_rel,
                "probe_status": "pass",
                "probe_reason": reason,
                "probe_checked_at": checked_at,
                "probe_artifact_path": canary_artifact_rel,
            },
            "assumed_healthy",
        )

    reason = "dispatch_health_unavailable"
    return (
        {
            "worker": worker,
            "health_status": "probationary",
            "canary_status": "required",
            "reason": reason,
            "checked_at": checked_at,
            "canary_reason": reason,
            "canary_checked_at": checked_at,
            "canary_artifact_path": None,
            "probe_status": "required",
            "probe_reason": reason,
            "probe_checked_at": checked_at,
            "probe_artifact_path": None,
        },
        "fail_closed_probationary",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build execution-supervisor worker-health canary evidence artifact")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]), help="Repository root")
    ap.add_argument(
        "--latest",
        default="state/continuity/latest/execution_supervisor_worker_health_canary_latest.json",
        help="Latest output JSON path",
    )
    ap.add_argument(
        "--history",
        default="state/continuity/history/execution_supervisor_worker_health_canary_history.jsonl",
        help="History output JSONL path",
    )
    ap.add_argument(
        "--dispatch-health-latest",
        default="state/continuity/latest/execution_supervisor_dispatch_health_latest.json",
        help="Persisted dispatch-health latest snapshot path",
    )
    ap.add_argument(
        "--dispatch-health-history",
        default="state/continuity/history/execution_supervisor_dispatch_health_history.jsonl",
        help="Persisted dispatch-health history JSONL path",
    )
    ap.add_argument(
        "--dispatch-health-matrix-latest",
        default="state/continuity/latest/dispatch_health_matrix.json",
        help="Persisted dispatch-health matrix latest snapshot path",
    )
    ap.add_argument(
        "--dispatch-health-json",
        help="Read dispatch-health payload from file instead of running codex_dispatch_health.py",
    )
    ap.add_argument(
        "--dispatch-qualification",
        default="state/continuity/latest/execution_supervisor_dispatch_qualification_latest.json",
        help="Optional dispatch-qualification source for route worker discovery",
    )
    ap.add_argument("--worker", action="append", help="Additional worker id(s) to include")
    ap.add_argument("--lookback-hours", type=float, default=24.0)
    ap.add_argument("--json", action="store_true", help="Emit JSON payload")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()

    latest_path = ensure_within(repo_root, args.latest)
    history_path = ensure_within(repo_root, args.history)
    dispatch_latest_path = ensure_within(repo_root, args.dispatch_health_latest)
    dispatch_history_path = ensure_within(repo_root, args.dispatch_health_history)
    dispatch_matrix_latest_path = ensure_within(repo_root, args.dispatch_health_matrix_latest)
    dispatch_qualification_path = ensure_within(repo_root, args.dispatch_qualification)

    generated_at = now_iso()

    dispatch_health_source = "command"
    dispatch_health_error: Optional[str] = None
    dispatch_health_payload: Dict[str, Any] = {}

    try:
        if args.dispatch_health_json:
            dispatch_health_source = "file"
            dispatch_health_payload = load_json(ensure_within(repo_root, args.dispatch_health_json))
        else:
            dispatch_health_payload = run_dispatch_health(
                repo_root,
                lookback_hours=args.lookback_hours,
                include_orchestrator=True,
            )
    except Exception as exc:
        dispatch_health_error = str(exc)
        dispatch_health_payload = {
            "generatedAt": generated_at,
            "statusCounts": {"healthy": 0, "probationary": 0, "quarantined": 0},
            "lanes": [],
            "error": dispatch_health_error,
        }

    dispatch_lanes = dispatch_health_payload.get("lanes")
    if not isinstance(dispatch_lanes, list):
        dispatch_lanes = []

    orchestrator_agents = discover_orchestrator_agents(repo_root=repo_root, dispatch_health_payload=dispatch_health_payload)
    dispatch_worker_lanes = [
        lane
        for lane in dispatch_lanes
        if isinstance(lane, dict) and str(lane.get("agent") or "").strip() not in orchestrator_agents
    ]

    dispatch_generated_at = str(dispatch_health_payload.get("generatedAt") or "").strip() or generated_at
    if parse_iso(dispatch_generated_at) is None:
        dispatch_generated_at = generated_at

    dispatch_health_artifact = {
        "schema": "clawd.execution_supervisor_dispatch_health.v1",
        "generated_at": generated_at,
        "source": dispatch_health_source,
        "lookback_hours": max(0.0, float(args.lookback_hours)),
        "source_error": dispatch_health_error,
        "dispatch_health": dispatch_health_payload,
    }
    write_json(dispatch_latest_path, dispatch_health_artifact)
    append_jsonl(dispatch_history_path, dispatch_health_artifact)

    dispatch_artifact_rel = to_rel(repo_root, dispatch_latest_path)
    dispatch_disable_policy = (
        dispatch_health_payload.get("routineDispatchLaneDisablePolicy")
        if isinstance(dispatch_health_payload.get("routineDispatchLaneDisablePolicy"), dict)
        else {}
    )
    dispatch_health_matrix_payload = _build_dispatch_health_matrix(
        generated_at=generated_at,
        dispatch_generated_at=dispatch_generated_at,
        dispatch_health_source=dispatch_health_source,
        dispatch_health_error=dispatch_health_error,
        dispatch_artifact_rel=dispatch_artifact_rel,
        dispatch_lanes=dispatch_worker_lanes,
        routine_dispatch_lane_disable_statuses=(
            dispatch_disable_policy.get("laneDisableStatuses")
            if isinstance(dispatch_disable_policy.get("laneDisableStatuses"), list)
            else None
        ),
        quota_exhaustion_reason_prefixes=(
            dispatch_disable_policy.get("quotaExhaustionReasonPrefixes")
            if isinstance(dispatch_disable_policy.get("quotaExhaustionReasonPrefixes"), list)
            else None
        ),
    )
    write_json(dispatch_matrix_latest_path, dispatch_health_matrix_payload)

    discovered_route_workers: Set[str] = set()
    if dispatch_qualification_path.exists():
        try:
            discovered_route_workers = discover_route_workers(load_json(dispatch_qualification_path))
        except Exception:
            discovered_route_workers = set()

    explicit_workers: Set[str] = set()
    for worker in args.worker or []:
        token = str(worker or "").strip()
        if token:
            explicit_workers.add(token)

    lane_by_worker: Dict[str, Dict[str, Any]] = {}
    for lane in dispatch_worker_lanes:
        if not isinstance(lane, dict):
            continue
        worker = str(lane.get("agent") or "").strip()
        if worker:
            lane_by_worker[worker] = lane

    worker_universe: Set[str] = set(DEFAULT_ROUTE_WORKERS)
    worker_universe.update(discovered_route_workers)
    worker_universe.update(explicit_workers)
    worker_universe.update(lane_by_worker.keys())

    dispatch_health_ready = dispatch_health_error is None

    worker_rows: List[Dict[str, Any]] = []
    assumed_healthy_workers: List[str] = []
    fail_closed_fallback_workers: List[str] = []

    canary_artifact_rel = to_rel(repo_root, latest_path)

    for worker in sorted(worker_universe):
        lane = lane_by_worker.get(worker)
        if isinstance(lane, dict):
            row = worker_row_from_dispatch_lane(
                worker=worker,
                lane_row=lane,
                checked_at=dispatch_generated_at,
                dispatch_health_artifact_rel=dispatch_artifact_rel,
            )
            worker_rows.append(row)
            continue

        fallback_row, fallback_mode = fallback_worker_row(
            worker=worker,
            checked_at=generated_at,
            canary_artifact_rel=canary_artifact_rel,
            dispatch_health_ready=dispatch_health_ready,
        )
        worker_rows.append(fallback_row)
        if fallback_mode == "assumed_healthy":
            assumed_healthy_workers.append(worker)
        else:
            fail_closed_fallback_workers.append(worker)

    status_counts = {"healthy": 0, "probationary": 0, "quarantined": 0}
    for row in worker_rows:
        status = str(row.get("health_status") or "").strip().lower()
        if status in status_counts:
            status_counts[status] += 1

    row_by_worker: Dict[str, Dict[str, Any]] = {
        str(row.get("worker") or "").strip(): row for row in worker_rows if isinstance(row, dict)
    }
    active_pool_workers: List[Dict[str, Any]] = []
    for worker in sorted(lane_by_worker.keys()):
        row = row_by_worker.get(worker) or {}
        active_pool_workers.append(
            {
                "worker": worker,
                "health_status": row.get("health_status"),
                "canary_status": row.get("canary_status"),
                "probe_status": row.get("probe_status"),
                "reason": row.get("reason"),
                "checked_at": row.get("checked_at"),
            }
        )

    active_pool_status_counts = _dispatch_status_counts_from_lanes(dispatch_worker_lanes)
    active_pool_posture, active_pool_posture_reason = _classify_active_pool_posture(active_pool_status_counts)
    active_pool_payload = {
        "worker_count": len(active_pool_workers),
        "status_counts": active_pool_status_counts,
        "green_worker_count": int(active_pool_status_counts.get("healthy") or 0),
        "yellow_worker_count": int(active_pool_status_counts.get("probationary") or 0),
        "red_worker_count": int(active_pool_status_counts.get("quarantined") or 0),
        "posture": active_pool_posture,
        "posture_reason": active_pool_posture_reason,
        "workers": active_pool_workers,
    }

    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "source": {
            "dispatch_health_path": dispatch_artifact_rel,
            "dispatch_health_source": dispatch_health_source,
            "dispatch_health_generated_at": dispatch_generated_at,
            "dispatch_health_source_error": dispatch_health_error,
            "dispatch_health_status_counts": dispatch_health_payload.get("statusCounts"),
            "dispatch_health_lane_count": len(dispatch_lanes),
            "dispatch_health_worker_lane_count": len(dispatch_worker_lanes),
            "excluded_orchestrator_agents": sorted(orchestrator_agents),
            "dispatch_health_matrix_path": to_rel(repo_root, dispatch_matrix_latest_path),
            "dispatch_health_matrix_generated_at": dispatch_health_matrix_payload.get("generated_at"),
            "dispatch_health_matrix_counts": dispatch_health_matrix_payload.get("counts"),
            "dispatch_qualification_path": to_rel(repo_root, dispatch_qualification_path),
            "dispatch_qualification_present": dispatch_qualification_path.exists(),
            "default_route_workers": list(DEFAULT_ROUTE_WORKERS),
            "discovered_route_worker_count": len(discovered_route_workers),
            "discovered_route_workers": sorted(discovered_route_workers),
            "assumed_healthy_worker_count": len(assumed_healthy_workers),
            "assumed_healthy_workers": assumed_healthy_workers,
            "fail_closed_fallback_worker_count": len(fail_closed_fallback_workers),
            "fail_closed_fallback_workers": fail_closed_fallback_workers,
        },
        "worker_count": len(worker_rows),
        "status_counts": status_counts,
        "active_pool": active_pool_payload,
        "workers": worker_rows,
    }

    write_json(latest_path, payload)
    append_jsonl(history_path, payload)

    result = {
        "ok": True,
        "schema": SCHEMA,
        "generated_at": generated_at,
        "latest_path": to_rel(repo_root, latest_path),
        "history_path": to_rel(repo_root, history_path),
        "dispatch_health_latest_path": dispatch_artifact_rel,
        "dispatch_health_history_path": to_rel(repo_root, dispatch_history_path),
        "dispatch_health_matrix_latest_path": to_rel(repo_root, dispatch_matrix_latest_path),
        "worker_count": len(worker_rows),
        "status_counts": status_counts,
        "assumed_healthy_worker_count": len(assumed_healthy_workers),
        "fail_closed_fallback_worker_count": len(fail_closed_fallback_workers),
        "dispatch_health_source_error": dispatch_health_error,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
