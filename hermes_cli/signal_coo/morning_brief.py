"""Morning brief assembly for Torben's EA scope."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .action_ledger import parse_time

LOCAL_TZ = ZoneInfo("America/New_York")
BRIEF_RULES = [
    "Name actual times and actual meetings; never give generic advice.",
    "If a source is unavailable, flag it and keep going.",
    "Research instead of summarizing when the claim depends on outside context.",
    "If the brief does not know something, say so instead of guessing.",
]


def _local(value: datetime) -> datetime:
    return value.astimezone(LOCAL_TZ)


def _format_window(start_at: str | None, end_at: str | None) -> str:
    start = parse_time(start_at)
    end = parse_time(end_at)
    if not start or not end:
        return "time unknown"
    start_local = _local(start)
    end_local = _local(end)
    if start_local.date() == end_local.date():
        return f"{start_local:%a %-m/%-d %-I:%M %p}-{end_local:%-I:%M %p}"
    return f"{start_local:%a %-m/%-d %-I:%M %p} to {end_local:%a %-m/%-d %-I:%M %p}"


def _events_for_today(events: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    today = _local(now).date()
    todays_events: list[tuple[datetime, dict[str, Any]]] = []
    for event in events:
        start = parse_time(event.get("start_at"))
        if start and _local(start).date() == today:
            todays_events.append((start, event))
    todays_events.sort(key=lambda item: item[0])
    return [event for _, event in todays_events]


def _open_blocks(events: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    local_now = _local(now)
    work_start = datetime.combine(local_now.date(), time(8, 0), tzinfo=LOCAL_TZ)
    work_end = datetime.combine(local_now.date(), time(18, 0), tzinfo=LOCAL_TZ)
    cursor = max(local_now, work_start)
    blocks: list[dict[str, Any]] = []
    for event in events:
        start = parse_time(event.get("start_at"))
        end = parse_time(event.get("end_at"))
        if not start or not end:
            continue
        start_local = _local(start)
        end_local = _local(end)
        if end_local <= cursor or start_local >= work_end:
            continue
        if start_local > cursor and (start_local - cursor) >= timedelta(minutes=30):
            blocks.append({"start_at": cursor.isoformat(), "end_at": start_local.isoformat()})
        cursor = max(cursor, end_local)
    if cursor < work_end and (work_end - cursor) >= timedelta(minutes=30):
        blocks.append({"start_at": cursor.isoformat(), "end_at": work_end.isoformat()})
    return blocks[:4]


def build_morning_brief_scope(
    ea_evidence: dict[str, Any],
    *,
    now: datetime | None = None,
    north_star: str = "advance the highest-leverage founder work before the inbox takes over",
) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    events = list(ea_evidence.get("calendar_events") or [])
    todays_events = _events_for_today(events, now)
    block_candidates = list(ea_evidence.get("calendar_block_candidates") or [])
    reply_candidates = list(ea_evidence.get("email_reply_candidates") or [])
    open_blocks = _open_blocks(todays_events, now)

    decisions: list[dict[str, Any]] = []
    for candidate in block_candidates[:5]:
        decisions.append(
            {
                "kind": "calendar_alignment",
                "summary": (
                    f"{candidate.get('summary') or 'Busy time'} is on "
                    f"{candidate.get('source_account') or 'one calendar'} but missing blocks on "
                    f"{', '.join(candidate.get('target_accounts') or []) or 'other calendars'}."
                ),
                "cost_of_delay": "calendar conflict risk increases if the account receives a competing invite",
                "time": _format_window(candidate.get("start_at"), candidate.get("end_at")),
                "evidence_ids": list(candidate.get("evidence_ids") or []),
            }
        )
    for candidate in reply_candidates[:3]:
        decisions.append(
            {
                "kind": "email_reply",
                "summary": candidate.get("context_line") or f"Reply candidate from {candidate.get('sender') or 'sender'}.",
                "cost_of_delay": "thread stays open and may require a colder restart later",
                "draft_detail": candidate.get("staged_response_detail"),
                "evidence_ids": list(candidate.get("evidence_ids") or []),
            }
        )

    meetings: list[dict[str, Any]] = []
    for event in todays_events[:12]:
        meetings.append(
            {
                "summary": event.get("summary") or "Busy",
                "time": _format_window(event.get("start_at"), event.get("end_at")),
                "why_it_exists": event.get("goal") or "calendar context only; agenda not captured",
                "desired_outcome": event.get("goal") or "make the meeting produce a decision or next step",
                "one_question": event.get("recommended_line") or "What decision or blocker needs to be resolved before we end?",
                "unknowns": ["invite agenda"] if str(event.get("last_conversation") or "").startswith("calendar context only") else [],
                "evidence_ids": list(event.get("evidence_ids") or []),
            }
        )

    if block_candidates:
        top = block_candidates[0]
        move = {
            "summary": (
                f"Align calendars for {top.get('summary') or 'busy time'} "
                f"before anything else lands on {', '.join(top.get('target_accounts') or [])}."
            ),
            "draft_message": "Approve the calendar block action and I will create busy holds only for the missing target calendars.",
            "evidence_ids": list(top.get("evidence_ids") or []),
        }
    elif open_blocks:
        first_block = open_blocks[0]
        move = {
            "summary": f"Protect the open block {_format_window(first_block.get('start_at'), first_block.get('end_at'))}.",
            "draft_message": "I would hold this block for founder work before opening the inbox.",
            "evidence_ids": [],
        }
    else:
        move = {
            "summary": "No clean 30-minute work block is visible today.",
            "draft_message": "I would shorten or move the lowest-value meeting before checking email.",
            "evidence_ids": [],
        }

    return {
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "north_star": north_star,
        "rules": BRIEF_RULES,
        "day": {
            "summary": f"{len(todays_events)} event(s) today; {len(open_blocks)} open work block(s) of at least 30 minutes.",
            "events": [
                {
                    "summary": event.get("summary") or "Busy",
                    "time": _format_window(event.get("start_at"), event.get("end_at")),
                    "account": event.get("account_alias"),
                    "calendar": event.get("calendar_summary"),
                    "evidence_ids": list(event.get("evidence_ids") or []),
                }
                for event in todays_events[:12]
            ],
            "open_blocks": open_blocks,
        },
        "decisions": decisions,
        "people": {
            "summary": "People radar is not fully connected yet; 1:1 notes, chat, and commitment history are required before calling a person.",
            "status": "needs_connectors",
            "specific_90_second_move": "Use today's external meeting prep first; do not invent a people-risk signal.",
        },
        "meetings": meetings,
        "world": {
            "summary": "World scan is delegated to GTM/Magnus research; no live research source is attached to this EA-only evidence run.",
            "status": "placeholder",
        },
        "move": move,
    }


def render_morning_brief_text(morning: dict[str, Any]) -> str:
    day = morning.get("day") or {}
    people = morning.get("people") or {}
    world = morning.get("world") or {}
    move = morning.get("move") or {}
    decisions = list(morning.get("decisions") or [])
    meetings = list(morning.get("meetings") or [])

    lines = [
        "Morning Brief",
        f"Day: {day.get('summary') or 'No day summary available.'}",
        f"Decisions: {len(decisions)} item(s) blocked on you or worth staging early.",
        f"People: {people.get('summary') or 'No people signal available.'}",
        f"Meetings: {len(meetings)} meeting(s) prepared with purpose/outcome/question lines.",
        f"World: {world.get('summary') or 'No world scan available.'}",
        f"Move: {move.get('summary') or 'No move selected.'}",
    ]
    if move.get("draft_message"):
        lines.append(f"Draft: {move['draft_message']}")
    if decisions:
        lines.append("Top decisions:")
        for decision in decisions[:3]:
            lines.append(f"- {decision.get('summary')} ({decision.get('time') or decision.get('cost_of_delay')})")
    if meetings:
        lines.append("Meeting prep:")
        for meeting in meetings[:4]:
            lines.append(f"- {meeting.get('time')}: {meeting.get('summary')} - ask: {meeting.get('one_question')}")
    return "\n".join(lines).strip() + "\n"
