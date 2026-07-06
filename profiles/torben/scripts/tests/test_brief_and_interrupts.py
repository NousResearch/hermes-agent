from __future__ import annotations

import json
from pathlib import Path

from torben_attention_contract import (
    BRIEF_SECTIONS,
    build_brief_attention_section,
    dedupe_meeting_alerts,
    load_pattern_proposals,
    load_pending_decisions,
    should_send_interrupt,
    silence_on_success,
    summarize_open_loops,
)
from torben_open_loops import add_loop


def test_brief_attention_contains_six_reads_plus_pending_decisions(tmp_path: Path) -> None:
    loops = tmp_path / "loops.csv"
    pending = tmp_path / "pending.json"
    add_loop(path=loops, item="Pay insurance bill", state="next-action", domain="money")
    pending.write_text(json.dumps([{"handle": "EA-1", "summary": "Approve packet"}]), encoding="utf-8")

    section = build_brief_attention_section(
        pending_decisions=load_pending_decisions(pending),
        open_loops=summarize_open_loops(loops),
    )

    assert section["sections"] == BRIEF_SECTIONS
    assert section["sections"][-1] == "Pending Decisions"
    assert section["pending_decisions"][0]["handle"] == "EA-1"
    assert section["open_loops"]["active"][0]["item"] == "Pay insurance bill"


def test_brief_attention_includes_review_gated_pattern_proposals(tmp_path: Path) -> None:
    loops = tmp_path / "loops.csv"
    pending = tmp_path / "pending.json"
    proposals = tmp_path / "pattern.json"
    add_loop(path=loops, item="Pay insurance bill", state="next-action", domain="money")
    pending.write_text("[]", encoding="utf-8")
    proposals.write_text(
        json.dumps(
            {
                "schema": "torben.pattern-miner.v1",
                "proposals": [
                    {
                        "id": "pattern-1",
                        "summary": "Automation candidate: draft investor update",
                        "status": "review_gated",
                        "support_count": 4,
                        "validation_score": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    section = build_brief_attention_section(
        pending_decisions=load_pending_decisions(pending),
        open_loops=summarize_open_loops(loops),
        pattern_proposals=load_pattern_proposals(proposals),
    )

    assert section["pending_decisions"][0]["handle"] == "pattern-1"
    assert section["pending_decisions"][0]["status"] == "review_gated"


def test_meeting_alerts_dedupe_to_one_per_meeting() -> None:
    alerts = [
        {"alert_key": "meeting-1", "title": "Kim", "start_at": "2026-07-06T12:00:00Z"},
        {"alert_key": "meeting-1", "title": "Kim", "start_at": "2026-07-06T12:00:00Z"},
        {"alert_key": "meeting-2", "title": "Alex", "start_at": "2026-07-06T13:00:00Z"},
    ]

    deduped = dedupe_meeting_alerts(alerts)

    assert [alert["alert_key"] for alert in deduped] == ["meeting-1", "meeting-2"]


def test_interrupt_gate_requires_actionable_time_sensitive_and_not_duplicate() -> None:
    assert should_send_interrupt({"actionable": True, "time_sensitive": True, "duplicate": False}) is True
    assert should_send_interrupt({"actionable": False, "time_sensitive": True, "duplicate": False}) is False
    assert should_send_interrupt({"actionable": True, "time_sensitive": False, "duplicate": False}) is False
    assert should_send_interrupt({"actionable": True, "time_sensitive": True, "duplicate": True}) is False


def test_silence_on_success_contract() -> None:
    assert silence_on_success("canary") == {
        "task": "canary",
        "wakeAgent": False,
        "reason": "success_no_actionable_payload",
    }
