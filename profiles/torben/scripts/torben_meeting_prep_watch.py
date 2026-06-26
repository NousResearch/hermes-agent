from __future__ import annotations

import json
import os
import time
from json import JSONDecodeError
from datetime import datetime, timezone
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.google_evidence import collect_google_ea_evidence, write_json_artifact
from hermes_cli.signal_coo.meeting_prep import (
    alert_bucket,
    event_start,
    load_meeting_prep_state,
    save_meeting_prep_state,
    select_pre_call_alerts,
    stage_meeting_prep_action,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    start = event_start(event)
    attendees = list(event.get("attendees") or [])
    return {
        "handle": event.get("handle"),
        "title": event.get("title") or event.get("summary"),
        "calendar_account": event.get("account_alias"),
        "calendar_email": event.get("account_email"),
        "start_at": event.get("start_at"),
        "end_at": event.get("end_at"),
        "minutes_until": event.get("minutes_until"),
        "alert_bucket_minutes": alert_bucket(int(event.get("minutes_until") or 0)),
        "local_start_hint": start.astimezone().strftime("%-I:%M %p") if start else None,
        "goal": event.get("goal"),
        "last_conversation": event.get("last_conversation"),
        "recommended_line": event.get("recommended_line"),
        "description_excerpt": str(event.get("description") or "")[:600],
        "attendees_count": event.get("attendees_count"),
        "attendees": [
            {
                "display_name": attendee.get("display_name"),
                "email_domain": str(attendee.get("email") or "").split("@")[-1] if attendee.get("email") else "",
                "response_status": attendee.get("response_status"),
            }
            for attendee in attendees[:8]
        ],
        "hangout_link_present": bool(event.get("hangout_link_present")),
        "evidence_ids": list(event.get("evidence_ids") or []),
    }


def main() -> int:
    home = get_hermes_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    preview = _truthy(os.getenv("TORBEN_MEETING_PREP_WATCH_PREVIEW"))
    force = _truthy(os.getenv("TORBEN_MEETING_PREP_WATCH_FORCE"))
    window_minutes = int(os.getenv("TORBEN_MEETING_PREP_WINDOW_MINUTES", "30"))
    bucket_minutes = int(os.getenv("TORBEN_MEETING_PREP_BUCKET_MINUTES", "5"))
    now = _utc_now()

    last_token_parse_error: JSONDecodeError | None = None
    for attempt in range(3):
        try:
            payload = collect_google_ea_evidence(
                config_path=home / "config" / "google_accounts.yaml",
                days=1,
                max_calendar_events=int(os.getenv("TORBEN_MEETING_PREP_MAX_EVENTS", "120")),
                max_email_messages=0,
                max_calendar_block_candidates=0,
                include_secondary_calendars=False,
                now=now,
            )
            break
        except JSONDecodeError as exc:
            # Google auth refresh can briefly rewrite a token file while the watcher reads it.
            # Treat that as a transient race and retry instead of waking Eric on a one-off parse miss.
            last_token_parse_error = exc
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))
                continue
            raise RuntimeError("Google token JSON stayed unreadable after retry; check auth token file") from last_token_parse_error
    write_json_artifact(payload, state_dir / "torben-meeting-prep-watch-latest.json")
    state_path = state_dir / "torben-meeting-prep-watch-state.json"
    state = {"version": 1, "alerted": {}} if force else load_meeting_prep_state(state_path, now=now)
    events = list(((payload.get("ea") or {}).get("calendar_events") or []))
    alerts = select_pre_call_alerts(
        events,
        state=state,
        now=now,
        window_minutes=window_minutes,
        bucket_minutes=bucket_minutes,
        max_alerts=int(os.getenv("TORBEN_MEETING_PREP_MAX_ALERTS", "1")),
    )

    if not alerts:
        print(json.dumps({"wakeAgent": False, "reason": "no upcoming meeting prep alert"}))
        return 0

    ledger = ActionLedger(state_dir / "torben-action-ledger.json")
    for event in alerts:
        if preview:
            event["handle"] = "PREVIEW-MEETING-001"
        else:
            action = stage_meeting_prep_action(ledger=ledger, event=event, now=now)
            event["handle"] = action.handle
            state.setdefault("alerted", {})[str(event.get("alert_key"))] = {
                "sent_at": _iso(now),
                "title": event.get("title") or event.get("summary"),
                "start_at": event.get("start_at"),
                "handle": action.handle,
            }
    if not preview:
        state["updated_at"] = _iso(now)
        save_meeting_prep_state(state_path, state)

    output = {
        "task": "torben_meeting_prep_watch",
        "wakeAgent": True,
        "preview": preview,
        "generated_at": _iso(now),
        "mutation_boundary": "read/stage only; do not mutate calendars, send email, open attachments, or join calls",
        "response_contract": [
            "Send one concise conversational pre-call alert, not a report.",
            "Start with: 'You have <meeting> in <minutes> minutes.'",
            "Include why the call appears to exist, the likely goal/outcome, and one recommended line or question.",
            "If calendar context is weak, say what is missing instead of inventing prior context.",
            "Mention the handle so Eric can ask for changes or add context.",
            "Do not include raw attendee emails; use names/domains only when useful.",
            "If the alert is not actionable, respond exactly [SILENT].",
        ],
        "alerts": [_compact_event(event) for event in alerts],
        "day_context": {
            "remaining_event_count": len([event for event in events if event_start(event) and event_start(event) >= now]),
            "next_events_after_alert": [
                _compact_event(event)
                for event in events
                if event not in alerts and event_start(event) and event_start(event) >= now
            ][:5],
        },
        "diagnostics": {
            "calendar_events_scanned": len(events),
            "google_reads": ((payload.get("source_diagnostics") or {}).get("google") or {}).get("audit", {}).get(
                "google_read_api_calls", 0
            ),
            "google_writes": ((payload.get("source_diagnostics") or {}).get("google") or {}).get("audit", {}).get(
                "google_write_api_calls", 0
            ),
            "external_mutations": ((payload.get("source_diagnostics") or {}).get("google") or {}).get("audit", {}).get(
                "external_mutations", 0
            ),
            "state_path": str(state_path),
            "window_minutes": window_minutes,
            "bucket_minutes": bucket_minutes,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
