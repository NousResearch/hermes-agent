from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.meeting_prep import (
    detect_meeting_lifecycle_transitions,
    detect_reschedule_follow_watch_transitions,
    mark_meeting_decision_staged,
    meeting_event_uid,
    record_meeting_prep_alert,
    register_reschedule_follow_watch,
    select_pre_call_alerts,
    stage_meeting_decline_decision,
)


NOW = datetime(2026, 7, 6, 13, 30, tzinfo=timezone.utc)


def _event(*, start: datetime, attendee_status: str = "accepted", **overrides: object) -> dict:
    payload = {
        "account_alias": "personal",
        "calendar_id": "primary",
        "event_id": "eno-1",
        "title": "Eno / Eric",
        "start_at": start.isoformat().replace("+00:00", "Z"),
        "end_at": (start + timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
        "attendees_count": 2,
        "attendees": [
            {"email": "eric@example.com", "responseStatus": "accepted"},
            {"email": "eno@example.com", "responseStatus": attendee_status},
        ],
        "evidence_ids": ["calendar:eno-1"],
        "recommended_line": "Ask whether the new time still works.",
    }
    payload.update(overrides)
    return payload


def test_decline_transition_stages_one_calendar_edit_decision_and_suppresses_prep(tmp_path: Path) -> None:
    state: dict = {}
    ledger = ActionLedger(tmp_path / "ledger.json")
    start = NOW + timedelta(minutes=30)
    accepted = _event(start=start, attendee_status="accepted")
    declined = _event(start=start, attendee_status="declined")

    assert detect_meeting_lifecycle_transitions([accepted], state=state, now=NOW) == []
    transitions = detect_meeting_lifecycle_transitions([declined], state=state, now=NOW + timedelta(minutes=5))
    repeated = detect_meeting_lifecycle_transitions([declined], state=state, now=NOW + timedelta(minutes=10))

    assert len(transitions) == 1
    assert repeated == []
    record = stage_meeting_decline_decision(ledger=ledger, transition=transitions[0], now=NOW)
    mark_meeting_decision_staged(state=state, transition=transitions[0], record=record)
    assert record.status == "approval_required"
    assert record.executor_state["mutation_type"] == "calendar_edit"
    assert record.allowed_next_actions == ["remove_from_calendar", "propose_new_time", "leave_as_is"]
    assert select_pre_call_alerts([declined], state=state, now=NOW + timedelta(minutes=10)) == []


def test_single_prep_alert_allows_only_new_context_t5_followup() -> None:
    state: dict = {}
    start = NOW + timedelta(minutes=30)
    event = _event(start=start)

    first = select_pre_call_alerts([event], state=state, now=NOW)
    assert len(first) == 1
    assert first[0]["alert_key"] == f"{meeting_event_uid(event)}:prep"
    assert first[0]["prep_followup"] is False
    record_meeting_prep_alert(state=state, event=first[0], handle="EA-20260706-001", now=NOW)

    assert select_pre_call_alerts([event], state=state, now=NOW + timedelta(minutes=5)) == []
    assert select_pre_call_alerts([event], state=state, now=start - timedelta(minutes=5)) == []

    changed = _event(start=start, last_conversation="New reply arrived after first prep.")
    followup = select_pre_call_alerts([changed], state=state, now=start - timedelta(minutes=5))
    assert len(followup) == 1
    assert followup[0]["prep_followup"] is True


def test_reschedule_accept_registers_72h_follow_watch_and_decline_references_original_handle(tmp_path: Path) -> None:
    state: dict = {}
    ledger = ActionLedger(tmp_path / "ledger.json")
    event = _event(start=NOW + timedelta(hours=2))
    record = ledger.add_action(
        scope="EA",
        summary="Accept reschedule",
        allowed_next_actions=["calendar_reschedule_accept"],
        status="approved",
        executor_state={
            "mutation_type": "calendar_reschedule_accept",
            "meeting_event_uid": meeting_event_uid(event),
            "event_id": event["event_id"],
            "calendar_id": event["calendar_id"],
            "thread_id": "thread-1",
        },
        now=NOW,
    )

    watch = register_reschedule_follow_watch(state=state, record=record, now=NOW)
    changed = _event(start=NOW + timedelta(hours=2), attendee_status="declined")
    transitions = detect_reschedule_follow_watch_transitions([changed], state=state, now=NOW + timedelta(hours=1))

    assert watch is not None
    assert watch["expires_at"] == (NOW + timedelta(hours=72)).isoformat().replace("+00:00", "Z")
    assert len(transitions) == 1
    assert transitions[0]["original_handle"] == record.handle
    assert transitions[0]["reason"] == "decline_or_counter_after_reschedule_accept"


def test_eno_replay_produces_one_prep_and_one_decline_decision() -> None:
    state: dict = {}
    start = NOW + timedelta(minutes=30)
    messages: list[str] = []

    for offset, status in [
        (0, "accepted"),
        (5, "accepted"),
        (10, "accepted"),
        (20, "declined"),
        (25, "declined"),
    ]:
        now = NOW + timedelta(minutes=offset)
        event = _event(start=start, attendee_status=status)
        for transition in detect_meeting_lifecycle_transitions([event], state=state, now=now):
            messages.append(transition["type"])
        for alert in select_pre_call_alerts([event], state=state, now=now):
            messages.append("prep")
            record_meeting_prep_alert(state=state, event=alert, handle="EA-20260706-001", now=now)

    assert messages == ["prep", "meeting_decline_decision"]
