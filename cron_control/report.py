"""Diff report for Hermes Cron shadow verdicts.

This module compares two shadow/evidence payloads and produces a deterministic
new-vs-old report for rollout review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .evaluator import evaluate_snapshot
from .normalizer import utc_now_iso
from .shadow import collect_shadow_snapshot

_VERDICT_FIELDS = (
    "verdict_id",
    "incident_id",
    "job_id",
    "state",
    "evidence_state",
    "rule_id",
    "recommended_action",
    "automatic_action_allowed",
    "blocked_by",
    "evidence_refs",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _snapshot_from_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list) and isinstance(payload.get("evidence"), list):
        return dict(payload)
    if isinstance(payload, dict) and isinstance(payload.get("snapshot"), dict):
        snapshot = payload.get("snapshot")
        if isinstance(snapshot, dict):
            return dict(snapshot)
    return None


def _verdicts_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        verdicts = payload.get("verdicts")
        if isinstance(verdicts, list):
            return [dict(item) for item in verdicts if isinstance(item, dict)]
        snapshot = _snapshot_from_payload(payload)
        if snapshot is not None:
            return evaluate_snapshot(snapshot)
    return []


def _normalized_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalized_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _normalized_value(item)) for key, item in value.items()))
    return value


def _verdict_key(verdict: dict[str, Any]) -> tuple[str, str]:
    return (
        str(verdict.get("incident_id") or ""),
        str(verdict.get("job_id") or ""),
    )


def _verdict_summary(verdict: dict[str, Any]) -> dict[str, Any]:
    return {field: verdict.get(field) for field in _VERDICT_FIELDS}


def _count_by_field(verdicts: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(verdict.get(field) or "unknown") for verdict in verdicts)
    return dict(sorted(counts.items()))


def _diff_fields(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    for field in _VERDICT_FIELDS:
        if field in {"verdict_id", "incident_id", "job_id"}:
            continue
        before_value = before.get(field)
        after_value = after.get(field)
        if _normalized_value(before_value) != _normalized_value(after_value):
            diff[field] = {"baseline": before_value, "current": after_value}
    return diff


def build_shadow_diff_report(
    current_payload: Any,
    baseline_payload: Any | None = None,
) -> dict[str, Any]:
    """Build a deterministic new-vs-old diff report for shadow verdicts."""

    current_snapshot = _snapshot_from_payload(current_payload)
    baseline_snapshot = _snapshot_from_payload(baseline_payload) if baseline_payload is not None else None
    current_verdicts = _verdicts_from_payload(current_payload)
    baseline_verdicts = _verdicts_from_payload(baseline_payload) if baseline_payload is not None else []

    current_map = {_verdict_key(verdict): verdict for verdict in current_verdicts}
    baseline_map = {_verdict_key(verdict): verdict for verdict in baseline_verdicts}

    all_keys = sorted(set(current_map) | set(baseline_map))
    added: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    unchanged = 0
    state_transitions: Counter[str] = Counter()
    action_transitions: Counter[str] = Counter()

    for key in all_keys:
        before = baseline_map.get(key)
        after = current_map.get(key)
        if before is None and after is not None:
            added.append(_verdict_summary(after))
            continue
        if after is None and before is not None:
            removed.append(_verdict_summary(before))
            continue
        if before is None or after is None:
            continue
        field_diff = _diff_fields(before, after)
        if field_diff:
            changed.append(
                {
                    "key": {"incident_id": key[0], "job_id": key[1]},
                    "baseline": _verdict_summary(before),
                    "current": _verdict_summary(after),
                    "changed_fields": field_diff,
                }
            )
            state_transitions[f"{before.get('state')} -> {after.get('state')}"] += 1
            action_transitions[f"{before.get('recommended_action')} -> {after.get('recommended_action')}"] += 1
        else:
            unchanged += 1

    report = {
        "generated_at": utc_now_iso(),
        "baseline": {
            "snapshot_at": baseline_snapshot.get("collected_at") if baseline_snapshot else None,
            "verdict_count": len(baseline_verdicts),
            "state_counts": _count_by_field(baseline_verdicts, "state"),
            "action_counts": _count_by_field(baseline_verdicts, "recommended_action"),
        },
        "current": {
            "snapshot_at": current_snapshot.get("collected_at") if current_snapshot else None,
            "verdict_count": len(current_verdicts),
            "state_counts": _count_by_field(current_verdicts, "state"),
            "action_counts": _count_by_field(current_verdicts, "recommended_action"),
        },
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
            "unchanged": unchanged,
            "state_transitions": dict(sorted(state_transitions.items())),
            "action_transitions": dict(sorted(action_transitions.items())),
        },
        "changes": {
            "added": added,
            "removed": removed,
            "changed": changed,
        },
    }
    return report
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a diff report for Hermes Cron shadow verdicts")
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--current-json", type=Path, default=None)
    parser.add_argument("--jobs", type=Path, default=None)
    parser.add_argument("--executions", type=Path, default=None)
    parser.add_argument("--control-plane", type=Path, default=None)
    args = parser.parse_args(argv)

    baseline_payload = _load_json(args.baseline_json)
    if args.current_json is not None:
        current_payload = _load_json(args.current_json)
    else:
        current_payload = collect_shadow_snapshot(
            jobs_path=args.jobs,
            executions_path=args.executions,
            control_plane_path=args.control_plane,
        )

    report = build_shadow_diff_report(current_payload, baseline_payload)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
