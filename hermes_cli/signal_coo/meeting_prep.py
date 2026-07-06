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
DEFAULT_T5_WINDOW_MINUTES = 5
FOLLOW_WATCH_HOURS = 72


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


def meeting_context_hash(event: dict[str, Any]) -> str:
    material = {
        "goal": event.get("goal"),
        "last_conversation": event.get("last_conversation"),
        "recommended_line": event.get("recommended_line"),
        "evidence_ids": event.get("evidence_ids") or [],
        "updated_at": event.get("updated_at") or event.get("updated"),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def thin_context_line(event: dict[str, Any]) -> str:
    if event.get("last_conversation") or event.get("goal") or event.get("recommended_line"):
        return "Context is current from available calendar/email evidence."
    return "Context is thin: no recent thread or goal found."


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
    return f"{meeting_event_uid(event)}:prep"


def _meeting_state(state: dict[str, Any], event_uid: str) -> dict[str, Any]:
    meetings = state.setdefault("meetings", {})
    current = meetings.get(event_uid)
    if not isinstance(current, dict):
        current = {}
        meetings[event_uid] = current
    return current


def _migrate_bucket_alerts(state: dict[str, Any]) -> None:
    alerted = state.get("alerted") or {}
    if not isinstance(alerted, dict):
        state["alerted"] = {}
        return
    for key, value in alerted.items():
        if ":t-" not in str(key):
            continue
        event_uid = str(key).split(":t-", 1)[0]
        meeting = _meeting_state(state, event_uid)
        if not meeting.get("prep_alerted_at"):
            meeting["prep_alerted_at"] = (value or {}).get("sent_at") if isinstance(value, dict) else None
        meeting.setdefault("legacy_bucket_alert_keys", []).append(str(key))


def attendee_snapshot(event: dict[str, Any]) -> dict[str, str]:
    attendees = event.get("attendees") or event.get("attendee_states") or []
    if not isinstance(attendees, list):
        return {}
    snapshot: dict[str, str] = {}
    for attendee in attendees:
        if not isinstance(attendee, dict):
            continue
        key = str(
            attendee.get("email")
            or attendee.get("address")
            or attendee.get("displayName")
            or attendee.get("name")
            or attendee.get("id")
            or ""
        ).strip().lower()
        if not key:
            continue
        status = str(
            attendee.get("responseStatus")
            or attendee.get("response_status")
            or attendee.get("rsvp")
            or attendee.get("status")
            or "unknown"
        ).strip().lower()
        snapshot[key] = status
    return snapshot


def declined_attendees(event: dict[str, Any]) -> list[str]:
    snapshot = attendee_snapshot(event)
    return sorted(name for name, status in snapshot.items() if status in {"declined", "cancelled", "canceled"})


def is_organizer_cancelled(event: dict[str, Any]) -> bool:
    status = str(event.get("status") or event.get("event_status") or "").strip().lower()
    return bool(event.get("organizer_cancelled") or status in {"cancelled", "canceled"})


def _has_unresolved_meeting_decision(state: dict[str, Any], event_uid: str) -> bool:
    meeting = (state.get("meetings") or {}).get(event_uid) or {}
    decision = meeting.get("decision") if isinstance(meeting, dict) else None
    return isinstance(decision, dict) and decision.get("status") == "unresolved"


def load_meeting_prep_state(path: str | Path, *, now: datetime | None = None) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {"version": 2, "alerted": {}, "meetings": {}, "follow_watches": {}}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {"version": 2, "alerted": {}, "meetings": {}, "follow_watches": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "alerted": {}}
    alerted = payload.get("alerted")
    if not isinstance(alerted, dict):
        payload["alerted"] = {}
    if not isinstance(payload.get("meetings"), dict):
        payload["meetings"] = {}
    if not isinstance(payload.get("follow_watches"), dict):
        payload["follow_watches"] = {}
    _migrate_bucket_alerts(payload)
    cutoff = (now or utc_now()).astimezone(timezone.utc) - timedelta(days=2)
    pruned = {}
    for key, value in (payload.get("alerted") or {}).items():
        sent_at = parse_time(str((value or {}).get("sent_at") or ""))
        if sent_at and sent_at >= cutoff:
            pruned[str(key)] = value
    watch_cutoff = (now or utc_now()).astimezone(timezone.utc)
    watches = {}
    for key, value in (payload.get("follow_watches") or {}).items():
        expires_at = parse_time(str((value or {}).get("expires_at") or ""))
        if expires_at is None or expires_at >= watch_cutoff:
            watches[str(key)] = value
    return {**payload, "version": 2, "alerted": pruned, "follow_watches": watches}


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
        event_uid = meeting_event_uid(event)
        if _has_unresolved_meeting_decision(state, event_uid):
            continue
        key = alert_key(event, minutes_until=minutes_until, bucket_minutes=bucket_minutes)
        meeting = _meeting_state(state, event_uid)
        context_hash = meeting_context_hash(event)
        already_alerted = bool(meeting.get("prep_alerted_at") or key in alerted)
        context_changed = already_alerted and meeting.get("prep_context_hash") != context_hash
        if already_alerted and not (minutes_until <= DEFAULT_T5_WINDOW_MINUTES and context_changed and not meeting.get("prep_t5_alerted_at")):
            continue
        candidates.append(
            (
                minutes_until,
                start,
                {
                    **event,
                    "minutes_until": minutes_until,
                    "alert_key": key,
                    "meeting_event_uid": event_uid,
                    "prep_followup": bool(already_alerted),
                    "thin_context_line": thin_context_line(event),
                    "context_hash": context_hash,
                },
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in candidates[:max_alerts]]


def record_meeting_prep_alert(
    *,
    state: dict[str, Any],
    event: dict[str, Any],
    handle: str,
    now: datetime | None = None,
) -> None:
    now_text = iso((now or utc_now()).astimezone(timezone.utc))
    event_uid = str(event.get("meeting_event_uid") or meeting_event_uid(event))
    meeting = _meeting_state(state, event_uid)
    if event.get("prep_followup"):
        meeting["prep_t5_alerted_at"] = now_text
    else:
        meeting["prep_alerted_at"] = now_text
    meeting["prep_context_hash"] = str(event.get("context_hash") or meeting_context_hash(event))
    meeting["prep_handle"] = handle
    meeting["last_seen_at"] = now_text
    state.setdefault("alerted", {})[str(event.get("alert_key") or f"{event_uid}:prep")] = {
        "sent_at": now_text,
        "handle": handle,
    }


def detect_meeting_lifecycle_transitions(
    events: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now_text = iso((now or utc_now()).astimezone(timezone.utc))
    transitions: list[dict[str, Any]] = []
    for event in events:
        if not is_meeting_like(event):
            continue
        event_uid = meeting_event_uid(event)
        meeting = _meeting_state(state, event_uid)
        current_attendees = attendee_snapshot(event)
        previous_attendees = dict(meeting.get("attendees") or {})
        declined = declined_attendees(event)
        organizer_cancelled = is_organizer_cancelled(event)
        decision = meeting.get("decision") if isinstance(meeting.get("decision"), dict) else None
        unresolved = decision is not None and decision.get("status") == "unresolved"
        new_declines = [
            attendee
            for attendee in declined
            if previous_attendees.get(attendee) not in {"declined", "cancelled", "canceled"}
        ]
        if (organizer_cancelled or new_declines or (declined and not previous_attendees)) and not unresolved:
            reason = "organizer_cancelled" if organizer_cancelled else "attendee_declined"
            transition = {
                "type": "meeting_decline_decision",
                "severity": "yellow",
                "meeting_event_uid": event_uid,
                "alert_key": f"{event_uid}:decline-decision",
                "title": str(event.get("title") or event.get("summary") or "meeting"),
                "reason": reason,
                "declined_attendees": declined,
                "previous_attendees": previous_attendees,
                "current_attendees": current_attendees,
                "original_prep_handle": meeting.get("prep_handle"),
                "created_at": now_text,
                "event": event,
            }
            meeting["decision"] = {
                "status": "unresolved",
                "reason": reason,
                "alerted_at": now_text,
                "alert_key": transition["alert_key"],
            }
            transitions.append(transition)
        meeting["attendees"] = current_attendees
        meeting["last_seen_at"] = now_text
    return transitions


def _existing_meeting_prep_actions(ledger: ActionLedger) -> dict[str, ActionRecord]:
    existing: dict[str, ActionRecord] = {}
    for record in ledger.load():
        state = record.executor_state or {}
        event_uid = state.get("meeting_event_uid")
        if record.scope == "ea" and isinstance(event_uid, str) and record.status in {"staged", "approval_required", "approved"}:
            existing[event_uid] = record
    return existing


def _existing_meeting_decision_actions(ledger: ActionLedger) -> dict[str, ActionRecord]:
    existing: dict[str, ActionRecord] = {}
    for record in ledger.load():
        state = record.executor_state or {}
        event_uid = state.get("meeting_event_uid")
        if (
            record.scope == "ea"
            and isinstance(event_uid, str)
            and record.status in {"staged", "approval_required", "approved"}
            and state.get("mutation_type") == "calendar_edit"
        ):
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


def stage_meeting_decline_decision(
    *,
    ledger: ActionLedger,
    transition: dict[str, Any],
    now: datetime | None = None,
) -> ActionRecord:
    event = transition.get("event") if isinstance(transition.get("event"), dict) else {}
    event_uid = str(transition.get("meeting_event_uid") or meeting_event_uid(event))
    existing = _existing_meeting_decision_actions(ledger).get(event_uid)
    if existing is not None:
        return existing
    title = str(transition.get("title") or event.get("title") or event.get("summary") or "meeting")
    return ledger.add_action(
        scope="EA",
        summary=f"Meeting changed: {title}",
        evidence_ids=list(event.get("evidence_ids") or []),
        allowed_next_actions=["remove_from_calendar", "propose_new_time", "leave_as_is"],
        status="approval_required",
        risk_class="medium",
        ttl_hours=72,
        now=now,
        executor_state={
            "mutation_type": "calendar_edit",
            "mutation_status": "approval_required",
            "approval_gate": "calendar_edit",
            "meeting_event_uid": event_uid,
            "transition_type": transition.get("type"),
            "reason": transition.get("reason"),
            "declined_attendees": list(transition.get("declined_attendees") or []),
            "original_prep_handle": transition.get("original_prep_handle"),
            "title": title,
            "start_at": event.get("start_at"),
        },
    )


def mark_meeting_decision_staged(
    *,
    state: dict[str, Any],
    transition: dict[str, Any],
    record: ActionRecord,
) -> None:
    event_uid = str(transition.get("meeting_event_uid"))
    decision = _meeting_state(state, event_uid).setdefault("decision", {})
    decision["status"] = "unresolved"
    decision["handle"] = record.handle
    decision["allowed_next_actions"] = list(record.allowed_next_actions)


def register_reschedule_follow_watch(
    *,
    state: dict[str, Any],
    record: ActionRecord,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    executor = record.executor_state or {}
    if executor.get("mutation_type") != "calendar_reschedule_accept":
        return None
    now_utc = (now or utc_now()).astimezone(timezone.utc)
    event_uid = str(executor.get("meeting_event_uid") or executor.get("event_uid") or "")
    if not event_uid:
        return None
    watch = {
        "handle": record.handle,
        "meeting_event_uid": event_uid,
        "calendar_id": executor.get("calendar_id"),
        "event_id": executor.get("event_id"),
        "thread_id": executor.get("thread_id"),
        "created_at": iso(now_utc),
        "expires_at": iso(now_utc + timedelta(hours=FOLLOW_WATCH_HOURS)),
        "status": "watching",
    }
    state.setdefault("follow_watches", {})[record.handle] = watch
    return watch


def detect_reschedule_follow_watch_transitions(
    events: list[dict[str, Any]],
    *,
    state: dict[str, Any],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now_utc = (now or utc_now()).astimezone(timezone.utc)
    transitions: list[dict[str, Any]] = []
    watches = state.get("follow_watches") or {}
    for watch in watches.values():
        if not isinstance(watch, dict) or watch.get("status") != "watching":
            continue
        expires_at = parse_time(str(watch.get("expires_at") or ""))
        if expires_at and expires_at < now_utc:
            watch["status"] = "expired"
            continue
        target_uid = str(watch.get("meeting_event_uid") or "")
        for event in events:
            event_uid = meeting_event_uid(event)
            if event_uid != target_uid and event.get("event_id") != watch.get("event_id"):
                continue
            if declined_attendees(event) or event.get("counterproposal") or str(event.get("status") or "").lower() in {"declined", "countered"}:
                transitions.append(
                    {
                        "type": "reschedule_follow_watch_changed",
                        "severity": "yellow",
                        "meeting_event_uid": event_uid,
                        "original_handle": watch.get("handle"),
                        "title": str(event.get("title") or event.get("summary") or "meeting"),
                        "reason": "decline_or_counter_after_reschedule_accept",
                        "event": event,
                    }
                )
                watch["status"] = "triggered"
                break
    return transitions
