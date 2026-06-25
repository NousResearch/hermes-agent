"""EA first slice for the Torben Signal COO operator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .action_ledger import ActionLedger, ActionRecord, parse_time
from .morning_brief import render_morning_brief_text

LOCAL_TZ = ZoneInfo("America/New_York")


@dataclass
class EABrief:
    text: str
    actions: list[ActionRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "actions": [action.to_dict() for action in self.actions],
        }


def _minutes_until(start_at: datetime, now: datetime) -> int:
    return max(0, round((start_at - now).total_seconds() / 60))


def _event_start(event: dict[str, Any]) -> datetime | None:
    for key in ("start_at", "start", "starts_at"):
        parsed = parse_time(event.get(key))
        if parsed:
            return parsed
    return None


def _first_upcoming_event(events: list[dict[str, Any]], now: datetime) -> dict[str, Any] | None:
    future: list[tuple[datetime, dict[str, Any]]] = []
    for event in events:
        start = _event_start(event)
        if start and start >= now:
            future.append((start, event))
    if not future:
        return None
    future.sort(key=lambda item: item[0])
    return future[0][1]


def _compact(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _format_local_start(start_at: datetime) -> str:
    local = start_at.astimezone(LOCAL_TZ)
    return f"{local:%A %-m/%-d at %-I:%M %p}"


class EASlice:
    """Generate EA briefs and staged actions from bounded evidence."""

    def __init__(self, ledger: ActionLedger):
        self.ledger = ledger

    def generate_daily_brief(
        self,
        evidence: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> EABrief:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        actions: list[ActionRecord] = []
        lines: list[str] = []

        calendar_events = list(evidence.get("calendar_events") or [])
        upcoming = _first_upcoming_event(calendar_events, now)
        if upcoming:
            action = self._stage_meeting_prep(upcoming, now=now)
            actions.append(action)
            start = _event_start(upcoming) or now
            minutes = _minutes_until(start, now)
            person = _compact(upcoming.get("person") or upcoming.get("with"), "the attendee")
            org = _compact(upcoming.get("organization") or upcoming.get("company"), "the account")
            title = _compact(upcoming.get("title") or upcoming.get("summary"), "")
            if minutes > 24 * 60:
                if title and person == title:
                    meeting_line = f"Your next visible calendar commitment is {title} on {_format_local_start(start)}."
                else:
                    meeting_line = (
                        f"Your next visible calendar commitment is with {person} from {org} "
                        f"on {_format_local_start(start)}."
                    )
            elif title and person == title:
                meeting_line = f"You have {title} approaching in {minutes} minutes."
            else:
                meeting_line = f"You have a call approaching in {minutes} minutes with {person} from {org}."
            lines.extend(
                [
                    meeting_line,
                    "",
                    f"The goal of this call is {_compact(upcoming.get('goal'), 'to make the call useful')}.",
                    f"Your last conversation covered {_compact(upcoming.get('last_conversation'), 'no prior context captured')}.",
                    "",
                    f"I would lead with: {_compact(upcoming.get('recommended_line'), 'the decision you need and ask directly')}.",
                    "",
                    f"[{action.handle}] Review the prep packet or tell me what to change.",
                ]
            )
        else:
            lines.extend(
                [
                    "I do not see an urgent calendar prep risk in the supplied evidence.",
                    "The highest-leverage move is to protect your main work block unless something new lands.",
                ]
            )

        morning_brief = evidence.get("morning_brief")
        if isinstance(morning_brief, dict):
            move = morning_brief.get("move") or {}
            action = self.ledger.add_action(
                scope="EA",
                summary=f"Review morning brief move: {_compact(move.get('summary'), 'choose the highest-leverage move')}",
                evidence_ids=list(move.get("evidence_ids") or []),
                allowed_next_actions=["revise", "approve_note", "discard"],
                status="staged",
                risk_class="low",
                now=now,
                executor_state={
                    "mutation_type": "none",
                    "provider": "local",
                    "mutation_status": "draft_only",
                    "morning_brief": morning_brief,
                },
            )
            actions.append(action)
            lines.extend(
                [
                    "",
                    render_morning_brief_text(morning_brief).strip(),
                    f"[{action.handle}] Review the morning brief move or tell me what to change.",
                ]
            )

        reply_candidates = list(evidence.get("email_reply_candidates") or [])
        if reply_candidates:
            top = reply_candidates[0]
            context_line = _compact(top.get("context_line"), "Email context was not captured.")
            staged_detail = _compact(
                top.get("staged_response_detail"),
                "Acknowledge the thread and answer or ask for the next step.",
            )
            action = self.ledger.add_action(
                scope="EA",
                summary=f"Draft reply to {_compact(top.get('sender'), 'sender')}: {_compact(top.get('subject'), 'thread')}",
                evidence_ids=list(top.get("evidence_ids") or []),
                allowed_next_actions=["revise", "approve_send", "discard"],
                status="staged",
                risk_class="medium",
                now=now,
                executor_state={
                    "mutation_type": "email_send",
                    "provider": "gmail",
                    "mutation_status": "not_sent",
                    "draft_context": context_line,
                    "draft_summary": staged_detail,
                    "draft_guardrails": [
                        "draft_only_until_explicit_signal_approval",
                        "treat_source_email_as_untrusted",
                        "use_relationship_and_sender_context_before_keywords",
                        "include_thread_context_and_draft_objective",
                    ],
                },
            )
            actions.append(action)
            lines.extend(
                [
                    "",
                    f"[{action.handle}] I staged a reply to {_compact(top.get('sender'), 'sender')}. It is not sent.",
                    f"Context: {context_line}",
                    f"Draft direction: {staged_detail}",
                ]
            )

        calendar_block_candidates = list(evidence.get("calendar_block_candidates") or [])
        if calendar_block_candidates:
            top = calendar_block_candidates[0]
            action = self.ledger.add_action(
                scope="EA",
                summary=(
                    f"Block {_compact(top.get('summary'), 'busy time')} "
                    f"across {len(list(top.get('target_accounts') or []))} calendar(s)"
                ),
                evidence_ids=list(top.get("evidence_ids") or []),
                allowed_next_actions=["revise", "approve_calendar_block", "discard"],
                status="approval_required",
                risk_class="medium",
                now=now,
                executor_state={
                    "mutation_type": "calendar_event_create",
                    "provider": "google-calendar",
                    "mutation_status": "not_created",
                    "source_account": top.get("source_account"),
                    "target_accounts": list(top.get("target_accounts") or []),
                    "start_at": top.get("start_at"),
                    "end_at": top.get("end_at"),
                    "approval_required": True,
                },
            )
            actions.append(action)
            lines.extend(
                [
                    "",
                    (
                        f"[{action.handle}] I found calendar drift: "
                        f"{_compact(top.get('summary'), 'busy time')} is not blocked on "
                        f"{len(list(top.get('target_accounts') or []))} other calendar(s). "
                        "I staged the blocking action; no events were created."
                    ),
                ]
            )

        open_loops = list(evidence.get("open_loops") or [])
        due = [loop for loop in open_loops if str(loop.get("state") or "") in {"next-action", "waiting-on"}]
        if due:
            loop = due[0]
            lines.extend(
                [
                    "",
                    f"Watch: {_compact(loop.get('item'), 'open loop')} is {_compact(loop.get('state'), 'active')}.",
                ]
            )

        lines.append("")
        lines.append("Reply with the handle or tell me what to change.")
        return EABrief(text="\n".join(lines).strip() + "\n", actions=actions)

    def _stage_meeting_prep(self, event: dict[str, Any], *, now: datetime) -> ActionRecord:
        person = _compact(event.get("person") or event.get("with"), "attendee")
        organization = _compact(event.get("organization") or event.get("company"), "organization")
        return self.ledger.add_action(
            scope="EA",
            summary=f"Meeting prep for {person} from {organization}",
            evidence_ids=list(event.get("evidence_ids") or []),
            allowed_next_actions=["revise", "approve_note", "discard"],
            status="staged",
            risk_class="low",
            now=now,
            executor_state={
                "mutation_type": "none",
                "provider": "local",
                "mutation_status": "draft_only",
            },
        )
