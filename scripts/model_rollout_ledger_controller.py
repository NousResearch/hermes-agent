#!/usr/bin/env python3
"""Queue-integrated model rollout ledger/controller (v1).

Consumes PASS decisions from the model rollout gate decision log, enforces
dwell-time + health SLO checks for ring promotions, and appends deterministic
rollout ledger/events entries.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_DECISION_LOG = Path("state/continuity/model_rollout_gate_runner/decisions.jsonl")
DEFAULT_LEDGER_PATH = Path("state/continuity/model_rollout_ledger/ledger.jsonl")
DEFAULT_STATE_PATH = Path("state/continuity/model_rollout_ledger/state.json")
DEFAULT_EVENTS_LOG = Path("state/continuity/model_rollout_ledger/events.jsonl")
DEFAULT_HEALTH_PATH = Path("state/continuity/model_rollout_health/latest.json")

STATE_SCHEMA = "clawd.model_rollout_ledger.state.v1"
LEDGER_SCHEMA = "clawd.model_rollout_ledger.entry.v1"
EVENT_SCHEMA = "clawd.model_rollout_event.v1"
WRAPPER_REQUIRED_SCHEMA = "clawd.model_rollout_ledger.wrapper_contract.v1"
DEFAULT_ALLOWED_MUTATION_CALLSITES = {
    "continuity.sh:model-rollout-controller",
}

DWELL_SECONDS = {
    "CANARY": 3600,
    "RING_1": 7200,
    "RING_2": 14400,
}
HEALTH_REQUIRED_STATES = set(DWELL_SECONDS.keys())
HEALTH_MAX_AGE_SECONDS = 3600

ENTRY_EVENT_TYPES = {
    "promote": "PROMOTION_APPLIED",
    "rollback": "ROLLBACK_APPLIED",
    "kill": "KILL_APPLIED",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


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


def resolve_repo_path(repo_root: Path, raw_path: Path | str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def safe_repo_path(repo_root: Path, raw_path: Path | str) -> Tuple[bool, Path, Optional[str]]:
    try:
        resolved = resolve_repo_path(repo_root, raw_path)
    except Exception as exc:
        return False, repo_root, f"path_resolve_failed:{exc}"
    if not is_within(repo_root, resolved):
        return False, resolved, "path_outside_repo"
    return True, resolved, None


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema": STATE_SCHEMA,
            "updated_at": None,
            "decision_log": None,
            "last_processed_line": 0,
        }
    payload = load_json_file(path)
    if not isinstance(payload, dict):
        raise RuntimeError("state_not_object")
    if payload.get("schema") != STATE_SCHEMA:
        raise RuntimeError("state_schema_mismatch")
    last_processed = payload.get("last_processed_line")
    if not isinstance(last_processed, int) or last_processed < 0:
        raise RuntimeError("state_cursor_invalid")
    return dict(payload)


def load_ledger(ledger_path: Path) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
    if not ledger_path.exists():
        return True, None, []
    if not ledger_path.is_file():
        return False, "ledger_unavailable", []
    rows: List[Dict[str, Any]] = []
    try:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except Exception:
        return False, "ledger_unreadable", []
    return True, None, rows


def load_event_ids(events_path: Path) -> Tuple[bool, Optional[str], set[str]]:
    if not events_path.exists():
        return True, None, set()
    if not events_path.is_file():
        return False, "events_unavailable", set()
    event_ids: set[str] = set()
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                event_id = payload.get("event_id")
                if isinstance(event_id, str):
                    event_ids.add(event_id)
    except Exception:
        return False, "events_unreadable", set()
    return True, None, event_ids


def append_jsonl(path: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if path.exists() and not path.is_file():
            return {"written": False, "reason": "path_not_file", "path": str(path)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(row) + "\n")
        return {"written": True, "path": str(path)}
    except Exception as exc:
        return {"written": False, "reason": "append_failed", "error": str(exc), "path": str(path)}


def extract_rollout_transition(decision: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    gates = decision.get("gates")
    if not isinstance(gates, list):
        return None, None, None
    for row in gates:
        if not isinstance(row, dict):
            continue
        if row.get("gate") != "rollout_transition":
            continue
        if row.get("status") != "pass":
            continue
        details = row.get("details")
        if not isinstance(details, dict):
            continue
        action = details.get("action")
        current_state = details.get("current_state")
        requested_state = details.get("requested_state")
        return (
            action if isinstance(action, str) else None,
            current_state if isinstance(current_state, str) else None,
            requested_state if isinstance(requested_state, str) else None,
        )
    return None, None, None


def build_event_id(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def index_ledger(
    rows: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]], set[str], Dict[str, Dict[str, Any]]]:
    latest_by_id: Dict[str, Dict[str, Any]] = {}
    last_state_by_id: Dict[str, Dict[str, Dict[str, Any]]] = {}
    seen_event_ids: set[str] = set()
    event_by_id: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        event_id = row.get("event_id")
        if isinstance(event_id, str):
            seen_event_ids.add(event_id)
            event_by_id[event_id] = row
        qualification_id = row.get("qualification_id")
        result_state = row.get("result_state")
        if not isinstance(qualification_id, str):
            continue
        latest_by_id[qualification_id] = row
        if isinstance(result_state, str):
            last_state_by_id.setdefault(qualification_id, {})[result_state] = row
    return latest_by_id, last_state_by_id, seen_event_ids, event_by_id


def check_state_match(
    *,
    latest_entry: Optional[Dict[str, Any]],
    current_state: Optional[str],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if not isinstance(current_state, str):
        return False, "current_state_missing", {"error": "current_state_missing"}
    if latest_entry is None:
        if current_state in {"DRAFT", "QUALIFIED"}:
            return True, None, {"current_state": current_state, "ledger_state": None}
        return False, "ledger_state_missing", {"current_state": current_state, "ledger_state": None}
    ledger_state = latest_entry.get("result_state")
    if ledger_state != current_state:
        return False, "ledger_state_mismatch", {"current_state": current_state, "ledger_state": ledger_state}
    return True, None, {"current_state": current_state, "ledger_state": ledger_state}


def check_dwell(
    *,
    current_state: Optional[str],
    decision_time: Optional[dt.datetime],
    last_state_entry: Optional[Dict[str, Any]],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if not isinstance(current_state, str) or current_state not in DWELL_SECONDS:
        return True, None, {"required": False, "state": current_state}
    if decision_time is None:
        return False, "decision_time_invalid", {"error": "decision_time_invalid"}
    if last_state_entry is None:
        return False, "dwell_state_missing", {"state": current_state}
    entered_at = parse_iso(last_state_entry.get("recorded_at"))
    if entered_at is None:
        return False, "dwell_state_time_invalid", {"state": current_state}
    age_sec = int(max(0.0, (decision_time - entered_at).total_seconds()))
    required = DWELL_SECONDS[current_state]
    if age_sec < required:
        return False, "dwell_not_met", {"state": current_state, "age_sec": age_sec, "required_sec": required}
    return True, None, {"required": True, "state": current_state, "age_sec": age_sec, "required_sec": required}


def load_health_snapshot(path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if not path.exists():
        return False, "health_missing", {"error": "health_snapshot_missing", "path": str(path)}
    if not path.is_file():
        return False, "health_unavailable", {"error": "health_snapshot_not_file", "path": str(path)}
    try:
        payload = load_json_file(path)
    except Exception as exc:
        return False, "health_unreadable", {"error": "health_snapshot_unreadable", "detail": str(exc), "path": str(path)}
    if not isinstance(payload, dict):
        return False, "health_invalid", {"error": "health_snapshot_not_object", "path": str(path)}
    if payload.get("schema_version") != "clawd.model_rollout_health.v1":
        return False, "health_invalid", {"error": "schema_version_mismatch", "schema_version": payload.get("schema_version")}
    return True, None, payload


def check_health(
    *,
    current_state: Optional[str],
    decision_time: Optional[dt.datetime],
    health_snapshot: Optional[Dict[str, Any]],
    health_path: Path,
    max_age_seconds: int,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if not isinstance(current_state, str) or current_state not in HEALTH_REQUIRED_STATES:
        return True, None, {"required": False, "state": current_state}
    if decision_time is None:
        return False, "decision_time_invalid", {"error": "decision_time_invalid"}
    if health_snapshot is None:
        return False, "health_missing", {"error": "health_snapshot_missing", "path": str(health_path)}
    generated_at = parse_iso(health_snapshot.get("generated_at"))
    if generated_at is None:
        return False, "health_invalid", {"error": "generated_at_invalid", "path": str(health_path)}
    age_sec = int(max(0.0, (decision_time - generated_at).total_seconds()))
    if age_sec > max_age_seconds:
        return False, "health_stale", {"error": "health_snapshot_stale", "age_sec": age_sec, "max_age_sec": max_age_seconds}
    overall_status = str(health_snapshot.get("overall_status") or "").lower()
    if overall_status != "healthy":
        return False, "health_unhealthy", {"error": "overall_status_unhealthy", "overall_status": overall_status or None}
    rings = health_snapshot.get("rings") if isinstance(health_snapshot.get("rings"), dict) else {}
    ring = rings.get(current_state)
    if not isinstance(ring, dict):
        return False, "health_invalid", {"error": "ring_status_missing", "state": current_state}
    if ring.get("slo_ok") is not True:
        return False, "health_unhealthy", {"error": "ring_slo_not_ok", "state": current_state}
    return True, None, {"required": True, "state": current_state, "age_sec": age_sec, "max_age_sec": max_age_seconds}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Queue-integrated model rollout ledger/controller (v1)")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--decision-log", default=str(DEFAULT_DECISION_LOG), help="Gate decision log JSONL path")
    ap.add_argument("--ledger-path", default=str(DEFAULT_LEDGER_PATH), help="Append-only rollout ledger JSONL path")
    ap.add_argument("--state-path", default=str(DEFAULT_STATE_PATH), help="Controller state JSON path")
    ap.add_argument("--events-log", default=str(DEFAULT_EVENTS_LOG), help="Append-only rollout events JSONL path")
    ap.add_argument("--health-path", default=str(DEFAULT_HEALTH_PATH), help="Rollout health snapshot JSON path")
    ap.add_argument("--health-max-age-sec", default=HEALTH_MAX_AGE_SECONDS, type=int, help="Max age for health snapshot")
    ap.add_argument("--max-events", default=50, type=int, help="Max decision log lines to consume")
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON output")
    return ap.parse_args(argv)


def _resolve_allowed_callsites() -> List[str]:
    allowed = set(DEFAULT_ALLOWED_MUTATION_CALLSITES)
    raw = str(os.environ.get("OPENCLAW_MODEL_ROLLOUT_CONTROLLER_ALLOWED_CALLSITES") or "").strip()
    if raw:
        for token in raw.split(","):
            value = token.strip()
            if value:
                allowed.add(value)
    return sorted(allowed)


def enforce_wrapper_only_contract() -> Optional[Dict[str, Any]]:
    internal_mutation = str(os.environ.get("OPENCLAW_INTERNAL_MUTATION") or "").strip()
    callsite = str(os.environ.get("OPENCLAW_INTERNAL_MUTATION_CALLSITE") or "").strip()
    allowed_callsites = _resolve_allowed_callsites()

    if internal_mutation != "1":
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_env_missing",
            "required_env": ["OPENCLAW_INTERNAL_MUTATION=1", "OPENCLAW_INTERNAL_MUTATION_CALLSITE=<allowlisted>"],
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh model-rollout-controller --json",
        }

    if not callsite:
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_callsite_missing",
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh model-rollout-controller --json",
        }

    if callsite not in allowed_callsites:
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_callsite_not_allowlisted",
            "callsite": callsite,
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh model-rollout-controller --json",
        }

    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    wrapper_guard_error = enforce_wrapper_only_contract()
    if wrapper_guard_error is not None:
        print(json.dumps(wrapper_guard_error, ensure_ascii=False, indent=2 if args.json else None))
        return 2

    repo_root = Path(args.repo_root).expanduser().resolve()
    ok, decision_log_path, reason = safe_repo_path(repo_root, args.decision_log)
    if not ok:
        print(json.dumps({"ok": False, "error": reason, "path": str(decision_log_path)}))
        return 2
    ok, ledger_path, reason = safe_repo_path(repo_root, args.ledger_path)
    if not ok:
        print(json.dumps({"ok": False, "error": reason, "path": str(ledger_path)}))
        return 2
    ok, state_path, reason = safe_repo_path(repo_root, args.state_path)
    if not ok:
        print(json.dumps({"ok": False, "error": reason, "path": str(state_path)}))
        return 2
    ok, events_log_path, reason = safe_repo_path(repo_root, args.events_log)
    if not ok:
        print(json.dumps({"ok": False, "error": reason, "path": str(events_log_path)}))
        return 2
    ok, health_path, reason = safe_repo_path(repo_root, args.health_path)
    if not ok:
        print(json.dumps({"ok": False, "error": reason, "path": str(health_path)}))
        return 2

    try:
        state = load_state(state_path)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": "state_load_failed", "detail": str(exc)}))
        return 2

    if not decision_log_path.exists():
        result = {"ok": False, "error": "decision_log_missing", "path": str(decision_log_path)}
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
        return 2
    if not decision_log_path.is_file():
        result = {"ok": False, "error": "decision_log_not_file", "path": str(decision_log_path)}
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
        return 2

    ok, reason, ledger_rows = load_ledger(ledger_path)
    if not ok:
        result = {"ok": False, "error": reason, "path": str(ledger_path)}
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
        return 2

    ok, reason, event_ids = load_event_ids(events_log_path)
    if not ok:
        result = {"ok": False, "error": reason, "path": str(events_log_path)}
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
        return 2

    latest_by_id, last_state_by_id, seen_event_ids, ledger_by_event = index_ledger(ledger_rows)
    health_snapshot: Optional[Dict[str, Any]] = None
    health_loaded = False
    health_error: Optional[Dict[str, Any]] = None

    last_processed = int(state.get("last_processed_line") or 0)
    lines = decision_log_path.read_text(encoding="utf-8").splitlines()
    upper = min(len(lines), last_processed + max(0, int(args.max_events)))

    processed = 0
    consumed = 0
    applied = 0
    blocked = 0
    skipped = 0
    events: List[Dict[str, Any]] = []

    for idx in range(last_processed, upper):
        processed += 1
        raw = lines[idx].strip()
        if not raw:
            skipped += 1
            continue
        try:
            decision = json.loads(raw)
        except Exception as exc:
            result = {
                "ok": False,
                "error": "decision_log_unreadable",
                "line": idx + 1,
                "detail": str(exc),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
            return 2

        if not isinstance(decision, dict):
            result = {"ok": False, "error": "decision_not_object", "line": idx + 1}
            print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
            return 2

        if decision.get("schema") != "clawd.model_rollout_gate.decision.v1":
            result = {
                "ok": False,
                "error": "decision_schema_mismatch",
                "line": idx + 1,
                "schema": decision.get("schema"),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
            return 2

        consumed += 1
        if decision.get("decision") != "PASS":
            continue

        qualification_id = decision.get("qualification_id")
        if not isinstance(qualification_id, str) or not qualification_id:
            blocked += 1
            continue

        action, current_state, requested_state = extract_rollout_transition(decision)
        if action is None or current_state is None or requested_state is None:
            blocked += 1
            continue

        evaluated_at = decision.get("evaluated_at")
        decision_time = parse_iso(evaluated_at)
        if decision_time is None:
            blocked += 1
            continue

        decision_ref = decision.get("candidate") if isinstance(decision.get("candidate"), dict) else {}
        event_fingerprint = {
            "qualification_id": qualification_id,
            "evaluated_at": evaluated_at,
            "candidate_sha256": decision_ref.get("sha256"),
            "action": action,
            "current_state": current_state,
            "requested_state": requested_state,
        }
        event_id = build_event_id(event_fingerprint)
        if event_id in event_ids:
            skipped += 1
            continue

        if event_id in seen_event_ids and event_id not in event_ids:
            ledger_entry = ledger_by_event.get(event_id) or {}
            event_payload = {
                "schema": EVENT_SCHEMA,
                "event_id": event_id,
                "recorded_at": ledger_entry.get("recorded_at"),
                "qualification_id": ledger_entry.get("qualification_id"),
                "action": ledger_entry.get("action"),
                "event_type": (ledger_entry.get("controller") or {}).get("event_type"),
                "current_state": ledger_entry.get("current_state"),
                "requested_state": ledger_entry.get("requested_state"),
                "result_state": ledger_entry.get("result_state"),
                "decision_ref": ledger_entry.get("decision_ref"),
                "checks": (ledger_entry.get("controller") or {}).get("checks"),
            }
            event_result = append_jsonl(events_log_path, event_payload)
            if not event_result.get("written"):
                result = {"ok": False, "error": "events_append_failed", "detail": event_result}
                print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
                return 2
            event_ids.add(event_id)
            events.append(event_payload)
            skipped += 1
            continue

        latest_entry = latest_by_id.get(qualification_id)
        last_state_entry = last_state_by_id.get(qualification_id, {}).get(current_state)
        state_ok, state_reason, state_details = check_state_match(
            latest_entry=latest_entry,
            current_state=current_state,
        )

        dwell_ok, dwell_reason, dwell_details = check_dwell(
            current_state=current_state,
            decision_time=decision_time,
            last_state_entry=last_state_entry,
        )

        health_ok = True
        health_reason = None
        health_details: Dict[str, Any] = {"required": False}
        if action == "promote" and current_state in HEALTH_REQUIRED_STATES:
            if not health_loaded:
                health_loaded = True
                ok, reason, snapshot = load_health_snapshot(health_path)
                if ok:
                    health_snapshot = snapshot
                else:
                    health_error = {"error": reason, "detail": snapshot}
            if health_error:
                health_ok = False
                health_reason = str(health_error.get("error") or "health_missing")
                health_details = dict(health_error.get("detail") or {})
            else:
                health_ok, health_reason, health_details = check_health(
                    current_state=current_state,
                    decision_time=decision_time,
                    health_snapshot=health_snapshot,
                    health_path=health_path,
                    max_age_seconds=max(1, int(args.health_max_age_sec)),
                )

        event_type = ENTRY_EVENT_TYPES.get(action, "ACTION_APPLIED")
        result_state = requested_state
        apply_change = True

        if not state_ok:
            event_type = "ACTION_BLOCKED"
            result_state = current_state
            apply_change = False
        elif action == "promote" and (not dwell_ok or not health_ok):
            event_type = "PROMOTION_BLOCKED"
            result_state = current_state
            apply_change = False

        event_payload = {
            "schema": EVENT_SCHEMA,
            "event_id": event_id,
            "recorded_at": evaluated_at,
            "qualification_id": qualification_id,
            "action": action,
            "event_type": event_type,
            "current_state": current_state,
            "requested_state": requested_state,
            "result_state": result_state,
            "decision_ref": {
                "evaluated_at": evaluated_at,
                "decision": decision.get("decision"),
                "final_state": decision.get("final_state"),
                "candidate_path": decision_ref.get("path"),
                "candidate_sha256": decision_ref.get("sha256"),
            },
            "checks": {
                "state_match": {"ok": state_ok, "reason": state_reason, "details": state_details},
                "dwell": {"ok": dwell_ok, "reason": dwell_reason, "details": dwell_details},
                "health": {"ok": health_ok, "reason": health_reason, "details": health_details},
            },
        }

        if apply_change:
            ledger_entry = {
                "schema": LEDGER_SCHEMA,
                "recorded_at": evaluated_at,
                "qualification_id": qualification_id,
                "event_id": event_id,
                "action": action,
                "current_state": current_state,
                "requested_state": requested_state,
                "result_state": result_state,
                "decision_ref": event_payload.get("decision_ref"),
                "controller": {
                    "event_type": event_type,
                    "checks": event_payload.get("checks"),
                },
            }
            if event_id not in seen_event_ids:
                ledger_result = append_jsonl(ledger_path, ledger_entry)
                if not ledger_result.get("written"):
                    result = {"ok": False, "error": "ledger_append_failed", "detail": ledger_result}
                    print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
                    return 2
                seen_event_ids.add(event_id)
                latest_by_id[qualification_id] = ledger_entry
                last_state_by_id.setdefault(qualification_id, {})[result_state] = ledger_entry

            applied += 1
        else:
            blocked += 1

        if event_id not in event_ids:
            event_result = append_jsonl(events_log_path, event_payload)
            if not event_result.get("written"):
                result = {"ok": False, "error": "events_append_failed", "detail": event_result}
                print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
                return 2
            event_ids.add(event_id)

        events.append(event_payload)

    state_update = {
        "schema": STATE_SCHEMA,
        "updated_at": now_iso(),
        "decision_log": str(decision_log_path.relative_to(repo_root)),
        "last_processed_line": upper,
    }
    try:
        atomic_write_json(state_path, state_update)
    except Exception as exc:
        result = {"ok": False, "error": "state_write_failed", "detail": str(exc)}
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
        return 2

    result = {
        "ok": True,
        "consumed_lines": processed,
        "consumed_decisions": consumed,
        "applied": applied,
        "blocked": blocked,
        "skipped": skipped,
        "state": state_update,
        "events": events,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
