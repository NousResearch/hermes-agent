from __future__ import annotations

from datetime import date
from pathlib import Path

from torben_open_loops import load_loops
from torben_subscription_hunt import cancellation_packet, run_subscription_hunt


TODAY = date(2026, 7, 6)


def test_recurring_charge_creates_one_loop_and_dedupes_within_ttl(tmp_path: Path) -> None:
    tracker = tmp_path / "loops.csv"
    state = tmp_path / "subscription-state.json"
    charge = {"merchant": "News App", "amount": "$9.99", "cadence": "monthly", "recurring": True}

    first = run_subscription_hunt(
        monarch_status="ok",
        charges=[charge],
        refunds=[],
        subscriptions=[],
        tracker_path=tracker,
        state_path=state,
        today=TODAY,
    )
    second = run_subscription_hunt(
        monarch_status="ok",
        charges=[charge],
        refunds=[],
        subscriptions=[],
        tracker_path=tracker,
        state_path=state,
        today=TODAY,
    )

    loops = load_loops(tracker)
    assert first["events"][0]["status"] == "created"
    assert second["events"][0]["status"] == "deduped"
    assert len(loops) == 1
    assert loops[0].item == "Review recurring charge: News App $9.99 (monthly)"


def test_refund_followup_surfaces_once_after_due_date(tmp_path: Path) -> None:
    tracker = tmp_path / "loops.csv"
    state = tmp_path / "subscription-state.json"
    refund = {
        "merchant": "Airline",
        "amount": "$120",
        "expected_by": "2026-07-01",
        "source_id": "gmail:thread:refund-1",
        "arrived": False,
    }

    first = run_subscription_hunt(
        monarch_status="ok",
        charges=[],
        refunds=[refund],
        subscriptions=[],
        tracker_path=tracker,
        state_path=state,
        today=TODAY,
    )
    second = run_subscription_hunt(
        monarch_status="ok",
        charges=[],
        refunds=[refund],
        subscriptions=[],
        tracker_path=tracker,
        state_path=state,
        today=TODAY,
    )

    loops = load_loops(tracker)
    assert first["events"][0]["status"] == "created"
    assert second["events"][0]["status"] == "deduped"
    assert len(loops) == 1
    assert loops[0].item == "Follow up on missing refund: Airline $120"
    assert loops[0].due == "2026-07-06"


def test_cancellation_opportunity_is_packet_only_payment_adjacent() -> None:
    packet = cancellation_packet(
        {
            "merchant": "Unused SaaS",
            "amount": "$19",
            "cadence": "monthly",
            "cancel_pointer": "https://unused.example/cancel",
            "reason": "no login in 90d",
        }
    )

    assert packet["type"] == "cancellation_packet"
    assert packet["category"] == "payment_adjacent"
    assert packet["status"] == "packet_only"
    assert packet["external_actions_taken"] == []
    assert "no autonomous cancellation" in packet["blocked_actions"]


def test_empty_monarch_source_reports_floor_not_false_clear(tmp_path: Path) -> None:
    payload = run_subscription_hunt(
        monarch_status="empty",
        charges=[],
        refunds=[],
        subscriptions=[],
        tracker_path=tmp_path / "loops.csv",
        state_path=tmp_path / "subscription-state.json",
        today=TODAY,
    )

    assert payload["status"] == "source_failure"
    assert payload["reason"] == "p0_7_monarch_source_unavailable_or_empty"
    assert payload["events"] == []
    assert payload["packets"] == []
    assert not (tmp_path / "loops.csv").exists()
