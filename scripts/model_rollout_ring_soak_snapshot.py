#!/usr/bin/env python3
"""Deterministic rollout ring-soak automation snapshot (Wave 5)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from model_pool_policy_contract import load_pool_policy


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_POOL_POLICY_PATH = Path("docs/ops/model_pool_policy_v1.json")
DEFAULT_POOL_POLICY_SCHEMA = Path("docs/ops/schemas/model_pool_policy.schema.json")
DEFAULT_LEDGER_PATH = Path("state/continuity/model_rollout_ledger/ledger.jsonl")
DEFAULT_HEALTH_PATH = Path("state/continuity/model_rollout_health/latest.json")
DEFAULT_OUT_PATH = Path("state/continuity/model_rollout_soak/latest.json")

RING_NEXT_STATE = {
    "CANARY": "RING_1",
    "RING_1": "RING_2",
    "RING_2": "FULL",
}


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


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            raise ValueError(f"jsonl_non_object_line:{idx}")
    return rows


def _load_health(health_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not health_path.exists():
        return None, "health_missing"
    if not health_path.is_file():
        return None, "health_unavailable"
    try:
        payload = load_json_file(health_path)
    except Exception:
        return None, "health_unreadable"
    if not isinstance(payload, dict):
        return None, "health_invalid"
    if payload.get("schema_version") != "clawd.model_rollout_health.v1":
        return None, "health_schema_mismatch"
    return payload, None


def build_snapshot(
    *,
    pool_policy: Mapping[str, Any],
    policy_meta: Mapping[str, Any],
    ledger_rows: List[Mapping[str, Any]],
    health_snapshot: Optional[Mapping[str, Any]],
    health_error: Optional[str],
    now_ts: dt.datetime,
) -> Dict[str, Any]:
    rollout_policy = pool_policy.get("rollout_policy") if isinstance(pool_policy.get("rollout_policy"), Mapping) else {}
    dwell_cfg = rollout_policy.get("dwell_seconds") if isinstance(rollout_policy.get("dwell_seconds"), Mapping) else {}
    health_max_age = rollout_policy.get("health_max_age_seconds") if isinstance(rollout_policy.get("health_max_age_seconds"), int) else 3600

    dwell_seconds: Dict[str, int] = {}
    for state in ("CANARY", "RING_1", "RING_2"):
        raw = dwell_cfg.get(state)
        dwell_seconds[state] = int(raw) if isinstance(raw, int) and raw > 0 else {"CANARY": 3600, "RING_1": 7200, "RING_2": 14400}[state]

    latest_ring_rows: Dict[str, Dict[str, Any]] = {}
    for row in ledger_rows:
        if str(row.get("schema") or "") != "clawd.model_rollout_ledger.entry.v1":
            continue
        qualification_id = str(row.get("qualification_id") or "").strip()
        result_state = str(row.get("result_state") or "").strip()
        if not qualification_id or result_state not in dwell_seconds:
            continue

        ts = parse_iso(row.get("recorded_at"))
        if ts is None:
            continue

        current = latest_ring_rows.get(qualification_id)
        if not current:
            latest_ring_rows[qualification_id] = {"row": dict(row), "recorded_at": ts}
            continue
        if ts >= current["recorded_at"]:
            latest_ring_rows[qualification_id] = {"row": dict(row), "recorded_at": ts}

    health_generated_at = parse_iso(health_snapshot.get("generated_at")) if isinstance(health_snapshot, Mapping) else None
    health_age_sec = int(max(0.0, (now_ts - health_generated_at).total_seconds())) if health_generated_at else None
    health_fresh = bool(health_age_sec is not None and health_age_sec <= health_max_age)
    health_ok_global = bool(
        isinstance(health_snapshot, Mapping)
        and str(health_snapshot.get("overall_status") or "").lower() == "healthy"
        and health_fresh
    )

    rings = health_snapshot.get("rings") if isinstance(health_snapshot, Mapping) and isinstance(health_snapshot.get("rings"), Mapping) else {}

    active: List[Dict[str, Any]] = []
    ready_count = 0
    soaking_count = 0
    attention_count = 0

    for qualification_id, info in sorted(latest_ring_rows.items()):
        row = info["row"]
        entered_at = info["recorded_at"]
        state = str(row.get("result_state") or "")
        required = int(dwell_seconds.get(state, 0))
        age_sec = int(max(0.0, (now_ts - entered_at).total_seconds()))
        dwell_met = age_sec >= required

        ring_health = rings.get(state) if isinstance(rings.get(state), Mapping) else {}
        ring_slo_ok = ring_health.get("slo_ok") is True
        health_ok = bool(health_ok_global and ring_slo_ok)

        if health_error:
            local_status = "attention"
            local_reason = health_error
        elif not health_ok:
            local_status = "attention"
            local_reason = "health_not_ok"
        elif dwell_met:
            local_status = "ready"
            local_reason = "dwell_met"
        else:
            local_status = "soaking"
            local_reason = "dwell_pending"

        if local_status == "ready":
            ready_count += 1
        elif local_status == "soaking":
            soaking_count += 1
        else:
            attention_count += 1

        active.append(
            {
                "qualification_id": qualification_id,
                "state": state,
                "entered_at": entered_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "age_sec": age_sec,
                "required_dwell_sec": required,
                "dwell_met": dwell_met,
                "next_state": RING_NEXT_STATE.get(state),
                "health": {
                    "overall_healthy": health_ok_global,
                    "ring_slo_ok": ring_slo_ok,
                    "health_generated_at": health_snapshot.get("generated_at") if isinstance(health_snapshot, Mapping) else None,
                    "health_age_sec": health_age_sec,
                    "health_max_age_sec": health_max_age,
                    "health_fresh": health_fresh,
                },
                "status": local_status,
                "reason": local_reason,
            }
        )

    if not active:
        status = "idle"
    elif attention_count > 0:
        status = "attention"
    elif soaking_count > 0:
        status = "soaking"
    else:
        status = "ready"

    return {
        "schema": "clawd.model_rollout_ring_soak_snapshot.v1",
        "generated_at": now_ts.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "policy": {
            "policy_id": pool_policy.get("policy_id"),
            "policy_path": policy_meta.get("path"),
            "policy_schema_path": policy_meta.get("schema_path"),
            "dwell_seconds": dwell_seconds,
            "health_max_age_seconds": health_max_age,
        },
        "health": {
            "status": "ok" if health_error is None else "error",
            "error": health_error,
            "generated_at": health_snapshot.get("generated_at") if isinstance(health_snapshot, Mapping) else None,
            "overall_status": health_snapshot.get("overall_status") if isinstance(health_snapshot, Mapping) else None,
            "age_sec": health_age_sec,
            "fresh": health_fresh,
        },
        "counts": {
            "active_rollouts": len(active),
            "ready": ready_count,
            "soaking": soaking_count,
            "attention": attention_count,
        },
        "active": active,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic ring-soak automation snapshot")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--pool-policy", default=str(DEFAULT_POOL_POLICY_PATH), help="Unified model pool policy JSON path")
    ap.add_argument("--pool-policy-schema", default=str(DEFAULT_POOL_POLICY_SCHEMA), help="Unified model pool policy schema path")
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), help="Rollout ledger JSONL path")
    ap.add_argument("--health", default=str(DEFAULT_HEALTH_PATH), help="Rollout health snapshot JSON path")
    ap.add_argument("--out", default=str(DEFAULT_OUT_PATH), help="Output snapshot JSON path")
    ap.add_argument("--now", default="", help="Optional deterministic now timestamp (UTC ISO8601)")
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    pool_policy_path = resolve_path(repo_root, args.pool_policy)
    pool_policy_schema_path = resolve_path(repo_root, args.pool_policy_schema)
    ledger_path = resolve_path(repo_root, args.ledger)
    health_path = resolve_path(repo_root, args.health)
    out_path = resolve_path(repo_root, args.out)

    now_ts = parse_iso(args.now) if str(args.now or "").strip() else dt.datetime.now(dt.timezone.utc)
    if now_ts is None:
        result = {
            "schema": "clawd.model_rollout_ring_soak_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": "invalid_now_timestamp",
            "raw_now": args.now,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    policy_ok, reason, policy_meta, pool_policy = load_pool_policy(pool_policy_path, pool_policy_schema_path)
    if not policy_ok:
        result = {
            "schema": "clawd.model_rollout_ring_soak_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": reason,
            "details": policy_meta,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    try:
        ledger_rows = load_jsonl(ledger_path)
    except Exception as exc:
        result = {
            "schema": "clawd.model_rollout_ring_soak_snapshot.v1",
            "generated_at": now_iso(),
            "status": "error",
            "error": "ledger_read_failed",
            "detail": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else stable_json_dumps(result))
        return 2

    health_snapshot, health_error = _load_health(health_path)

    snapshot = build_snapshot(
        pool_policy=pool_policy if isinstance(pool_policy, Mapping) else {},
        policy_meta=policy_meta,
        ledger_rows=ledger_rows,
        health_snapshot=health_snapshot,
        health_error=health_error,
        now_ts=now_ts,
    )

    if not is_within(repo_root, out_path):
        result = {
            "schema": "clawd.model_rollout_ring_soak_snapshot.v1",
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

    return 0 if snapshot.get("status") in {"idle", "soaking", "ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
