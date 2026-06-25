"""Rolling pre-call meeting prep selection for Torben."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .action_ledger import ActionLedger, ActionRecord, parse_time


DEFAULT_WINDOW_MINUTES = 30
DEFAULT_BUCKET_MINUTES = 5


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def event_start(event: dict[str, Any]) -> datetime | None:
    for key in ("start_at", "start", "starts_at"):
        parsed = parse_time(str(event.get(key) or ""))
        if parsed:
            return parsed
    return None


def event_end(event: dict[str, Any]) -> datetime | None:
    for key in ("end_at", "end", "ends_at"):
        parsed = parse_time(str(event.get(key) or ""))
        if parsed:
            return parsed
    return None


def meeting_event_uid(event: dict[str, Any]) -> str:
    evidence = "|".join(str(item) for item in (event.get("evidence_ids") or []))
    raw = "|".join(
        [
            str(event.get("account_alias") or ""),
            str(event.get("calendar_id") or ""),
            str(event.get("title") or event.get("summary") or ""),
            str(event.get("start_at") or ""),
            str(event.get("end_at") or ""),
            evidence,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def is_synthetic_busy_block(event: dict[str, Any]) -> bool:
    title = str(event.get("title") or event.get("summary") or "").strip().lower()
    description = str(event.get("description") or "").lower()
    private_props = ((event.get("extended_properties") or {}).get("private") or {})
    evidence = " ".join(str(item) for item in (event.get("evidence_ids") or [])).lower()
    return bool(
        private_props.get("torben_alignment") == "true"
        or "torben auto calendar alignment block" in description
        or (title == "busy" and ":torben" in evidence)
    )


def is_meeting_like(event: dict[str, Any]) -> bool:
    if event.get("all_day"):
        return False
    if is_synthetic_busy_block(event):
        return False
    title = str(event.get("title") or event.get("summary") or "").strip().lower()
    if not title:
        return False
    has_people_or_link = int(event.get("attendees_count") or 0) > 0 or bool(event.get("hangout_link_present"))
    if has_people_or_link:
        return True
    if title in {"busy", "block", "blocked", "hold", "focus time", "focus", "ooo"}:
        return False
    if title.startswith("block ") or title.startswith("busy "):
        return False
    return True


def alert_bucket(minutes_until: int, *, bucket_minutes: int = DEFAULT_BUCKET_MINUTES) -> int:
    if minutes_until <= 0:
        return 0
    return int(math.ceil(minutes_until / bucket_minutes) * bucket_minutes)


def alert_key(event: dict[str, Any], *, minutes_until: int, bucket_minutes: int = DEFAULT_BUCKET_MINUTES) -> str:
    return f"{meeting_event_uid(event)}:t-{alert_bucket(minutes_until, bucket_minutes=bucket_minutes)}"


def load_meeting_prep_state(path: str | Path, *, now: datetime | None = None) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {"version": 1, "alerted": {}}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {"version": 1, "alerted": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "alerted": {}}
    alerted = payload.get("alerted")
    if not isinstance(alerted, dict):
        payload["alerted"] = {}
    cutoff = (now or utc_now()).astimezone(timezone.utc) - timedelta(days=2)
    pruned = {}
    for key, value in (payload.get("alerted") or {}).items():
        sent_at = parse_time(str((value or {}).get("sent_at") or ""))
        if sent_at and sent_at >= cutoff:
            pruned[str(key)] = value
    return {**payload, "version": 1, "alerted": pruned}


def save_meeting_prep_state(path: str | Path, state: dict[str, Any]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, state_path)


def select_pre_call_alerts(
    events: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    now: datetime | None = None,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    bucket_minutes: int = DEFAULT_BUCKET_MINUTES,
    max_alerts: int = 1,
) -> list[dict[str, Any]]:
    now = (now or utc_now()).astimezone(timezone.utc)
    candidates: list[tuple[int, datetime, dict[str, Any]]] = []
    alerted = state.get("alerted") or {}
    for event in events:
        start = event_start(event)
        end = event_end(event)
        if not start or not end:
            continue
        if end <= now:
            continue
        minutes_until = round((start - now).total_seconds() / 60)
        if minutes_until < 0 or minutes_until > window_minutes:
            continue
        if not is_meeting_like(event):
            continue
        key = alert_key(event, minutes_until=minutes_until, bucket_minutes=bucket_minutes)
        if key in alerted:
            continue
        candidates.append((minutes_until, start, {**event, "minutes_until": minutes_until, "alert_key": key}))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in candidates[:max_alerts]]


def _existing_meeting_prep_actions(ledger: ActionLedger) -> dict[str, ActionRecord]:
    existing: dict[str, ActionRecord] = {}
    for record in ledger.load():
        state = record.executor_state or {}
        event_uid = state.get("meeting_event_uid")
        if record.scope == "ea" and isinstance(event_uid, str) and record.status in {"staged", "approval_required", "approved"}:
            existing[event_uid] = record
    return existing


def stage_meeting_prep_action(
    *,
    ledger: ActionLedger,
    event: dict[str, Any],
    now: datetime | None = None,
) -> ActionRecord:
    event_uid = meeting_event_uid(event)
    existing = _existing_meeting_prep_actions(ledger).get(event_uid)
    if existing is not None:
        return existing
    title = str(event.get("title") or event.get("summary") or "meeting")
    start = event_start(event)
    return ledger.add_action(
        scope="EA",
        summary=f"Pre-call prep: {title}",
        evidence_ids=list(event.get("evidence_ids") or []),
        allowed_next_actions=["revise", "approve_note", "discard"],
        status="staged",
        risk_class="low",
        ttl_hours=12,
        now=now,
        executor_state={
            "mutation_type": "none",
            "provider": "local",
            "mutation_status": "draft_only",
            "meeting_event_uid": event_uid,
            "title": title,
            "start_at": iso(start) if start else event.get("start_at"),
            "goal": event.get("goal"),
            "last_conversation": event.get("last_conversation"),
            "recommended_line": event.get("recommended_line"),
        },
    )
