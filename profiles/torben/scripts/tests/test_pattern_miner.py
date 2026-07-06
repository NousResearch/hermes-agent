from __future__ import annotations

from datetime import date

from torben_open_loops import LoopRow
from torben_pattern_miner import mine_patterns, proposals_for_weekly_reset
from torben_weekly_reset import build_weekly_packet


TODAY = date(2026, 7, 6)


def test_four_manual_actions_produce_review_gated_automation_candidate() -> None:
    events = [
        {"source": "email", "text": "Draft weekly investor update", "confidence": 0.9},
        {"source": "calendar", "text": "Draft weekly investor update", "confidence": 0.85},
        {"source": "signal", "text": "Draft weekly investor update", "confidence": 0.95},
        {"source": "harness", "text": "Draft weekly investor update", "confidence": 0.9},
    ]

    payload = mine_patterns(events=events)

    assert len(payload["proposals"]) == 1
    proposal = payload["proposals"][0]
    assert proposal["status"] == "review_gated"
    assert proposal["support_count"] == 4
    assert proposal["validation_score"] >= 0.7
    assert proposal["review_gated"] is True


def test_low_support_pattern_is_withheld() -> None:
    payload = mine_patterns(events=[{"source": "email", "text": "Renew one-off permit", "confidence": 0.95}])

    assert payload["proposals"] == []
    assert payload["withheld"][0]["status"] == "withheld"
    assert payload["withheld"][0]["passes"] is False


def test_harness_secret_values_are_redacted_from_stored_proposal() -> None:
    payload = mine_patterns(
        events=[
            {"source": "harness", "text": "Run deploy with access_token=abc123SECRET", "confidence": 1.0},
            {"source": "harness", "text": "Run deploy with access_token=abc123SECRET", "confidence": 1.0},
            {"source": "harness", "text": "Run deploy with access_token=abc123SECRET", "confidence": 1.0},
            {"source": "harness", "text": "Run deploy with access_token=abc123SECRET", "confidence": 1.0},
        ]
    )
    serialized = str(payload)

    assert "abc123SECRET" not in serialized
    assert "[REDACTED]" in serialized


def test_review_gated_proposals_appear_in_weekly_reset_pending_decisions() -> None:
    mined = mine_patterns(
        events=[
            {"source": "email", "text": "Draft weekly investor update", "confidence": 0.9},
            {"source": "calendar", "text": "Draft weekly investor update", "confidence": 0.9},
            {"source": "signal", "text": "Draft weekly investor update", "confidence": 0.9},
            {"source": "harness", "text": "Draft weekly investor update", "confidence": 0.9},
        ]
    )
    proposals = proposals_for_weekly_reset(mined)

    packet = build_weekly_packet(
        loops=[LoopRow(1, "Pay bill", "next-action", "eric", "", "admin", "", "2026-07-01", "2026-07-01")],
        pending_decisions=[],
        pattern_proposals=proposals,
        today=TODAY,
    )

    pending = packet["sections"]["PENDING DECISIONS"]
    assert pending[0]["category"] == "pattern_miner"
    assert pending[0]["status"] == "review_gated"
    assert pending[0]["support_count"] == 4
