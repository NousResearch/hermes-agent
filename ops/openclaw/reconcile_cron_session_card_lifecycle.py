#!/usr/bin/env python3
"""Apply truthful lifecycle reconciliation writes for historical failed cron session cards.

This utility consumes:
  1) a no_llm_watchdog_cron_authority_guard --json summary payload
  2) an openclaw sessions --json payload (for authoritative per-agent store paths)

For rows projected as resolved historical residues, it mutates the authoritative
session store entry so card/status surfaces can converge from stale `failed`
into resolved historical lifecycle state, while preserving fail-closed behavior
for active/running rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FAILED_STATUSES = {
    "failed",
    "error",
    "aborted",
    "cancelled",
    "canceled",
    "timeout",
    "timed_out",
    "crashed",
}

ACTIVE_STATUSES = {"running"}

RESOLVED_PROJECTION_STATES = {
    "resolved_historical_projected",
    "resolved_historical_inferred",
}

RESOLVED_PROJECTED_STATUSES = {
    "historical_failed_reconciled_now",
    "historical_failed_reconciled",
}

AGENT_KEY_RE = re.compile(r"^agent:([^:]+):")


@dataclass(frozen=True)
class ReconcileRow:
    session_key: str
    projected_status: str
    projection_state: str
    projection_reason: str
    observed_status: str
    retirement_state: str
    updated_at_ms: Optional[int] = None


@dataclass
class MutationRecord:
    session_key: str
    store_path: str
    expected_projected_status: str
    before_status: str
    before_aborted_last_run: bool
    after_status: str
    after_aborted_last_run: bool


def _normalize_status(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_optional_epoch_ms(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except Exception:  # noqa: BLE001
        return None
    return parsed if parsed >= 0 else None


def _extract_reconcile_rows(summary: Dict[str, Any]) -> List[ReconcileRow]:
    reconciliation = summary.get("session_surface_reconciliation")
    if not isinstance(reconciliation, dict):
        return []
    rows = reconciliation.get("historical_failed_session_rows")
    if not isinstance(rows, list):
        return []

    out: List[ReconcileRow] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        session_key = str(row.get("session_key") or "").strip()
        if not session_key or session_key in seen:
            continue

        projected_status = _normalize_status(row.get("status"))
        projection_state = _normalize_status(row.get("status_projection_state"))

        if projection_state not in RESOLVED_PROJECTION_STATES:
            continue
        if projected_status not in RESOLVED_PROJECTED_STATUSES:
            continue

        seen.add(session_key)
        out.append(
            ReconcileRow(
                session_key=session_key,
                projected_status=projected_status,
                projection_state=projection_state,
                projection_reason=str(row.get("status_projection_reason") or "").strip(),
                observed_status=_normalize_status(row.get("status_observed")),
                retirement_state=_normalize_status(row.get("status_retirement_state")),
                updated_at_ms=_parse_optional_epoch_ms(row.get("updated_at_ms")),
            )
        )

    return out


def _extract_session_store_paths_from_sessions_list(payload: Dict[str, Any]) -> Dict[str, str]:
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        return {}

    out: Dict[str, str] = {}
    for agent_id, raw_path in paths.items():
        key = str(agent_id or "").strip()
        path = str(raw_path or "").strip()
        if not key or not path:
            continue
        out[key] = path
    return out


def _agent_id_from_session_key(session_key: str) -> Optional[str]:
    m = AGENT_KEY_RE.match(session_key)
    if not m:
        return None
    agent_id = str(m.group(1) or "").strip()
    return agent_id or None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _load_store(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"session store is not an object: {path}")
    return data


def _is_failed_like(status: str, aborted_last_run: bool) -> bool:
    return status in FAILED_STATUSES or aborted_last_run


def _build_lifecycle_metadata(
    *,
    previous_status: str,
    previous_aborted_last_run: bool,
    row: ReconcileRow,
    now_ms: int,
) -> Dict[str, Any]:
    return {
        "state": "resolved_historical",
        "source": "ops/openclaw/reconcile_cron_session_card_lifecycle.py",
        "updatedAt": now_ms,
        "projectedStatus": row.projected_status,
        "projectionState": row.projection_state,
        "projectionReason": row.projection_reason,
        "retirementState": row.retirement_state or "projected",
        "observedStatusBeforeWrite": previous_status or "unknown",
        "observedAbortedLastRunBeforeWrite": previous_aborted_last_run,
    }


def _verify_post_apply_mutations(mutated_records: List[MutationRecord]) -> Dict[str, Any]:
    if not mutated_records:
        return {
            "verification_state": "verified",
            "verified_mutation_count": 0,
            "verification_mismatches": [],
            "verification_store_load_errors": [],
        }

    verification_store_load_errors: List[Dict[str, str]] = []
    loaded_stores: Dict[str, Dict[str, Any]] = {}
    for raw_path in dict.fromkeys(record.store_path for record in mutated_records):
        path = Path(raw_path)
        try:
            loaded_stores[raw_path] = _load_store(path)
        except Exception as exc:  # noqa: BLE001
            verification_store_load_errors.append({"path": raw_path, "error": str(exc)})

    verified_mutation_count = 0
    verification_mismatches: List[Dict[str, Any]] = []
    for record in mutated_records:
        store = loaded_stores.get(record.store_path)
        if store is None:
            verification_mismatches.append(
                {
                    "session_key": record.session_key,
                    "store_path": record.store_path,
                    "reasons": ["store_unavailable_for_verification"],
                }
            )
            continue

        entry = store.get(record.session_key)
        if not isinstance(entry, dict):
            verification_mismatches.append(
                {
                    "session_key": record.session_key,
                    "store_path": record.store_path,
                    "reasons": ["session_missing_after_apply"],
                }
            )
            continue

        observed_status = _normalize_status(entry.get("status"))
        observed_aborted_last_run = bool(entry.get("abortedLastRun") is True)
        lifecycle = entry.get("sessionCardLifecycle")
        lifecycle_state = _normalize_status(lifecycle.get("state")) if isinstance(lifecycle, dict) else ""
        lifecycle_projected_status = (
            _normalize_status(lifecycle.get("projectedStatus")) if isinstance(lifecycle, dict) else ""
        )

        reasons: List[str] = []
        if observed_status != "done":
            reasons.append("status_not_done")
        if observed_aborted_last_run:
            reasons.append("aborted_last_run_not_cleared")
        if lifecycle_state != "resolved_historical":
            reasons.append("lifecycle_state_not_resolved_historical")
        if lifecycle_projected_status != record.expected_projected_status:
            reasons.append("lifecycle_projected_status_mismatch")

        if reasons:
            verification_mismatches.append(
                {
                    "session_key": record.session_key,
                    "store_path": record.store_path,
                    "reasons": reasons,
                    "expected": {
                        "status": "done",
                        "abortedLastRun": False,
                        "sessionCardLifecycle.state": "resolved_historical",
                        "sessionCardLifecycle.projectedStatus": record.expected_projected_status,
                    },
                    "observed": {
                        "status": observed_status,
                        "abortedLastRun": observed_aborted_last_run,
                        "sessionCardLifecycle.state": lifecycle_state,
                        "sessionCardLifecycle.projectedStatus": lifecycle_projected_status,
                    },
                }
            )
            continue

        verified_mutation_count += 1

    verification_state = (
        "verified" if not verification_store_load_errors and not verification_mismatches else "verification_failed"
    )
    return {
        "verification_state": verification_state,
        "verified_mutation_count": verified_mutation_count,
        "verification_mismatches": verification_mismatches,
        "verification_store_load_errors": verification_store_load_errors,
    }


def reconcile(
    *,
    rows: List[ReconcileRow],
    store_paths: Iterable[Path],
    store_path_by_agent: Dict[str, Path],
    apply_changes: bool,
) -> Dict[str, Any]:
    now_ms = int(time.time() * 1000)

    stores: Dict[Path, Dict[str, Any]] = {}
    store_load_errors: List[Dict[str, str]] = []
    for path in dict.fromkeys(store_paths):
        try:
            stores[path] = _load_store(path)
        except Exception as exc:  # noqa: BLE001
            store_load_errors.append({"path": str(path), "error": str(exc)})

    mutated_records: List[MutationRecord] = []
    skipped_missing: List[str] = []
    skipped_non_failed_like: List[str] = []
    skipped_active_running: List[str] = []
    skipped_no_store: List[str] = []
    stores_touched: set[Path] = set()

    for row in rows:
        preferred_path: Optional[Path] = None
        agent_id = _agent_id_from_session_key(row.session_key)
        if agent_id and agent_id in store_path_by_agent:
            preferred_path = store_path_by_agent[agent_id]

        target_store_path: Optional[Path] = None
        target_store: Optional[Dict[str, Any]] = None

        if preferred_path is not None and preferred_path in stores:
            candidate = stores[preferred_path]
            if row.session_key in candidate:
                target_store_path = preferred_path
                target_store = candidate

        if target_store_path is None:
            for path, store in stores.items():
                if row.session_key in store:
                    target_store_path = path
                    target_store = store
                    break

        if target_store_path is None or target_store is None:
            if preferred_path is not None and preferred_path not in stores:
                skipped_no_store.append(row.session_key)
            else:
                skipped_missing.append(row.session_key)
            continue

        entry = target_store.get(row.session_key)
        if not isinstance(entry, dict):
            skipped_missing.append(row.session_key)
            continue

        current_status = _normalize_status(entry.get("status"))
        current_aborted = bool(entry.get("abortedLastRun") is True)

        if current_status in ACTIVE_STATUSES:
            skipped_active_running.append(row.session_key)
            continue

        if not _is_failed_like(current_status, current_aborted):
            skipped_non_failed_like.append(row.session_key)
            continue

        before_status = current_status
        before_aborted = current_aborted

        entry["status"] = "done"
        entry["abortedLastRun"] = False
        entry["sessionCardLifecycle"] = _build_lifecycle_metadata(
            previous_status=before_status,
            previous_aborted_last_run=before_aborted,
            row=row,
            now_ms=now_ms,
        )
        entry["updatedAt"] = max(int(entry.get("updatedAt") or 0), now_ms)

        mutated_records.append(
            MutationRecord(
                session_key=row.session_key,
                store_path=str(target_store_path),
                expected_projected_status=row.projected_status,
                before_status=before_status,
                before_aborted_last_run=before_aborted,
                after_status="done",
                after_aborted_last_run=False,
            )
        )
        stores_touched.add(target_store_path)

    verification_state = "not_applicable_dry_run"
    verified_mutation_count = 0
    verification_mismatches: List[Dict[str, Any]] = []
    verification_store_load_errors: List[Dict[str, str]] = []
    if apply_changes:
        for path in stores_touched:
            _atomic_write_json(path, stores[path])
        verification_result = _verify_post_apply_mutations(mutated_records)
        verification_state = str(verification_result.get("verification_state") or "verification_failed")
        verified_mutation_count = int(verification_result.get("verified_mutation_count") or 0)
        verification_mismatches = list(verification_result.get("verification_mismatches") or [])
        verification_store_load_errors = list(verification_result.get("verification_store_load_errors") or [])

    return {
        "ok": (verification_state == "verified") if apply_changes else True,
        "mode": "apply" if apply_changes else "dry_run",
        "candidate_count": len(rows),
        "mutated_count": len(mutated_records),
        "verification_state": verification_state,
        "verified_mutation_count": verified_mutation_count,
        "verification_mismatch_count": len(verification_mismatches),
        "verification_store_load_error_count": len(verification_store_load_errors),
        "verification_mismatches": verification_mismatches,
        "verification_store_load_errors": verification_store_load_errors,
        "mutated": [
            {
                "session_key": r.session_key,
                "store_path": r.store_path,
                "before_status": r.before_status,
                "before_aborted_last_run": r.before_aborted_last_run,
                "after_status": r.after_status,
                "after_aborted_last_run": r.after_aborted_last_run,
            }
            for r in mutated_records
        ],
        "store_paths_touched": [str(p) for p in sorted(stores_touched)],
        "skipped_missing_session": skipped_missing,
        "skipped_missing_store": skipped_no_store,
        "skipped_non_failed_like": skipped_non_failed_like,
        "skipped_active_running": skipped_active_running,
        "store_load_errors": store_load_errors,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile resolved-historical cron session cards into authoritative session store lifecycle writes."
    )
    parser.add_argument("--summary-json", required=True, help="Path to no_llm watchdog guard --json summary payload")
    parser.add_argument(
        "--sessions-list-json",
        required=True,
        help="Path to openclaw sessions --json payload containing per-agent session store paths",
    )
    parser.add_argument(
        "--store-path",
        action="append",
        default=[],
        help="Additional explicit session store path(s) to scan",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply writes in-place (default is dry-run)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    summary_path = Path(args.summary_json)
    sessions_list_path = Path(args.sessions_list_json)
    extra_store_paths = [Path(p) for p in args.store_path]

    if not summary_path.is_file():
        print(json.dumps({"ok": False, "error": f"summary_json_not_found:{summary_path}"}))
        return 2
    if not sessions_list_path.is_file():
        print(json.dumps({"ok": False, "error": f"sessions_list_json_not_found:{sessions_list_path}"}))
        return 2

    try:
        summary_obj = _load_json(summary_path)
        if not isinstance(summary_obj, dict):
            raise ValueError("summary JSON payload must be an object")
        rows = _extract_reconcile_rows(summary_obj)

        sessions_list_obj = _load_json(sessions_list_path)
        if not isinstance(sessions_list_obj, dict):
            raise ValueError("sessions list JSON payload must be an object")
        store_path_map_raw = _extract_session_store_paths_from_sessions_list(sessions_list_obj)

        store_path_by_agent: Dict[str, Path] = {}
        for agent_id, raw_path in store_path_map_raw.items():
            path = Path(raw_path)
            store_path_by_agent[agent_id] = path

        store_paths: List[Path] = list(store_path_by_agent.values()) + extra_store_paths

        result = reconcile(
            rows=rows,
            store_paths=store_paths,
            store_path_by_agent=store_path_by_agent,
            apply_changes=bool(args.apply),
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("ok") is True else 1
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
