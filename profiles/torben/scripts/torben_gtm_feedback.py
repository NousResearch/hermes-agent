#!/usr/bin/env python3
"""Torben GTM feedback, liveness, and engagement dedupe TTL helpers."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCHEMA = "torben.gtm-feedback.v1"
DEFAULT_LIVENESS_THRESHOLD_DAYS = 4
DEFAULT_DEDUPE_TTL_DAYS = 14


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8") or "")
    except json.JSONDecodeError:
        return deepcopy(default)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_feedback(path: Path) -> dict[str, Any]:
    payload = _load_json(path, {"schema": SCHEMA, "events": []})
    if not isinstance(payload, dict):
        payload = {"schema": SCHEMA, "events": []}
    payload.setdefault("schema", SCHEMA)
    payload.setdefault("events", [])
    return payload


def record_feedback_event(
    *,
    path: Path,
    candidate_id: str,
    event: str,
    edit_summary: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    payload = load_feedback(path)
    entry = {
        "created_at": _iso(now),
        "candidate_id": candidate_id,
        "event": event,
        "edit_summary": edit_summary,
    }
    payload["events"].append(entry)
    payload["updated_at"] = entry["created_at"]
    if event in {"selected", "edited"}:
        payload["current_focus"] = candidate_id
    dismissed = set(payload.get("dismissed_from_latest_batch") or [])
    if event in {"dismissed", "dropped"}:
        dismissed.add(candidate_id)
    payload["dismissed_from_latest_batch"] = sorted(dismissed)
    _write_json(path, payload)
    return payload


def _event_weight(event: str) -> int:
    if event == "selected":
        return 8
    if event == "edited":
        return 5
    if event in {"dismissed", "dropped"}:
        return -6
    return 0


def feedback_weights(feedback: dict[str, Any]) -> dict[str, int]:
    weights: dict[str, int] = {}
    for event in feedback.get("events") or []:
        if not isinstance(event, dict):
            continue
        candidate_id = str(event.get("candidate_id") or "")
        if not candidate_id:
            continue
        weights[candidate_id] = weights.get(candidate_id, 0) + _event_weight(str(event.get("event") or ""))
    return weights


def apply_feedback_to_radar(radar: Any, feedback: dict[str, Any]) -> Any:
    weights = feedback_weights(feedback)
    if not weights:
        return radar
    adjusted = deepcopy(radar)

    def adjust_item(item: dict[str, Any]) -> None:
        identifiers = {str(item.get("id") or ""), str(item.get("fingerprint") or "")}
        bonus = sum(weights.get(identifier, 0) for identifier in identifiers if identifier)
        if bonus == 0:
            return
        item["feedback_score_bonus"] = int(item.get("feedback_score_bonus") or 0) + bonus
        for score_key in ("content_score", "llm_score", "score"):
            if score_key in item and isinstance(item.get(score_key), (int, float)):
                item[score_key] = max(0, min(100, int(item[score_key]) + bonus))
        item["feedback_applied"] = True

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "id" in value or "fingerprint" in value:
                adjust_item(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(adjusted)
    return adjusted


def evaluate_liveness(
    *,
    conversions_by_day: dict[str, int],
    state_path: Path,
    today: date,
    threshold_days: int = DEFAULT_LIVENESS_THRESHOLD_DAYS,
) -> dict[str, Any]:
    zero_days = []
    for offset in range(threshold_days):
        day = today - timedelta(days=offset)
        if int(conversions_by_day.get(day.isoformat(), 0)) == 0:
            zero_days.append(day.isoformat())
    zero_days = list(reversed(zero_days))
    state = _load_json(state_path, {"schema": "torben.gtm-liveness-state.v1", "alerts": []})
    if len(zero_days) < threshold_days:
        return {"schema": SCHEMA, "status": "ok", "wakeAgent": False, "zero_days": zero_days}
    alert_key = f"zero-conversions:{zero_days[0]}:{zero_days[-1]}"
    alerts = set(state.get("alerts") or [])
    if alert_key in alerts:
        return {"schema": SCHEMA, "status": "deduped", "wakeAgent": False, "alert_key": alert_key, "zero_days": zero_days}
    alerts.add(alert_key)
    state["alerts"] = sorted(alerts)
    state["updated_at"] = _iso()
    _write_json(state_path, state)
    return {
        "schema": SCHEMA,
        "status": "escalate",
        "wakeAgent": True,
        "alert_key": alert_key,
        "zero_days": zero_days,
        "threshold_days": threshold_days,
        "public_actions_taken": 0,
        "external_mutations": 0,
        "text": (
            f"Torben GTM liveness: zero human conversions for {threshold_days} consecutive days "
            f"({zero_days[0]} through {zero_days[-1]}). Confirm Q6 threshold/behavior or adjust the GTM loop."
        ),
    }


def ensure_engagement_dedupe_ttl(
    *,
    state_path: Path,
    now: datetime | None = None,
    ttl_days: int = DEFAULT_DEDUPE_TTL_DAYS,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    state = _load_json(state_path, {"schema_version": 1, "delivered_opportunities": {}})
    delivered = state.get("delivered_opportunities")
    if not isinstance(delivered, dict):
        return {"schema": SCHEMA, "status": "no_dedupe_dict", "kept": 0, "expired": 0}
    kept: dict[str, Any] = {}
    expired = 0
    for key, entry in delivered.items():
        if not isinstance(entry, dict):
            continue
        base = _parse_datetime(entry.get("last_seen_at")) or _parse_datetime(entry.get("first_delivered_at")) or current
        expires_at = base + timedelta(days=ttl_days)
        if expires_at <= current:
            expired += 1
            continue
        entry["expires_at"] = _iso(expires_at)
        entry["ttl_days"] = ttl_days
        kept[key] = entry
    state["delivered_opportunities"] = kept
    state["updated_at"] = _iso(current)
    _write_json(state_path, state)
    return {"schema": SCHEMA, "status": "ok", "kept": len(kept), "expired": expired, "ttl_days": ttl_days}


def build_gtm_post_packet(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "type": "gtm_post_packet",
        "category": "gtm_post",
        "status": "packet_only",
        "candidate_id": candidate.get("id") or candidate.get("fingerprint"),
        "summary": candidate.get("summary") or candidate.get("title") or candidate.get("angle") or "GTM draft",
        "recommended_action": candidate.get("recommended_action") or "draft",
        "approval_request": "Approve GTM post/reply packet, request edits, defer, or drop.",
        "blocked_actions": ["no autonomous public post", "no autonomous reply", "no scheduling without approval"],
        "public_actions_taken": 0,
        "external_actions_taken": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record")
    record.add_argument("--path", required=True)
    record.add_argument("--candidate-id", required=True)
    record.add_argument("--event", required=True)
    record.add_argument("--edit-summary", default="")
    ttl = subparsers.add_parser("ttl")
    ttl.add_argument("--state", required=True)
    ttl.add_argument("--ttl-days", type=int, default=DEFAULT_DEDUPE_TTL_DAYS)
    args = parser.parse_args(argv)

    if args.command == "record":
        payload = record_feedback_event(
            path=Path(args.path),
            candidate_id=args.candidate_id,
            event=args.event,
            edit_summary=args.edit_summary,
        )
    elif args.command == "ttl":
        payload = ensure_engagement_dedupe_ttl(state_path=Path(args.state), ttl_days=args.ttl_days)
    else:
        raise ValueError(f"Unhandled command: {args.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
