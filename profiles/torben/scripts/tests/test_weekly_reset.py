from __future__ import annotations

from datetime import date
from pathlib import Path

from torben_open_loops import LoopRow
from torben_weekly_reset import age_loops, build_weekly_packet, render_packet
from torben_weekly_reset_job import build_weekly_reset_from_state, write_weekly_reset_artifacts


TODAY = date(2026, 7, 6)


def test_weekly_packet_contains_defined_sections_from_loops_and_pending_decisions() -> None:
    loops = [
        LoopRow(1, "Pay insurance bill", "next-action", "eric", "2026-07-06", "money", "", "2026-07-01", "2026-07-01"),
        LoopRow(2, "Draft Kim follow-up", "next-action", "eric", "", "gtm", "", "2026-07-01", "2026-07-01"),
        LoopRow(3, "Waiting on dentist forms", "waiting-on", "dentist", "", "health", "", "2026-07-01", "2026-07-01"),
        LoopRow(4, "Fix garage keypad", "deferred-until", "eric", "2026-07-10", "home", "", "2026-07-01", "2026-07-01"),
    ]
    pending = [{"handle": "EA-1", "summary": "Approve form packet", "risk_class": "high"}]

    packet = build_weekly_packet(loops=loops, pending_decisions=pending, today=TODAY)

    assert list(packet["sections"]) == [
        "MUST NOT DROP",
        "SCHEDULE FLAGS",
        "PAPERWORK & ADMIN",
        "MESSAGES TO DRAFT",
        "WAITING ON",
        "PENDING DECISIONS",
        "ONE THING FOR THE WEEKEND",
        "STATE FLAGS",
    ]
    assert packet["sections"]["SCHEDULE FLAGS"][0]["item"] == "Pay insurance bill"
    assert packet["sections"]["PENDING DECISIONS"] == pending
    assert "PENDING DECISIONS" in render_packet(packet)


def test_aging_is_idempotent() -> None:
    loops = [
        LoopRow(1, "Pay bill", "next-action", "eric", "2026-07-06", "money", "", "2026-07-01", "2026-07-01"),
        LoopRow(2, "Done item", "done", "eric", "", "admin", "", "2026-07-01", "2026-07-02"),
    ]

    first = age_loops(loops)
    second = age_loops(loops)

    assert first == second
    assert first["active_count"] == 1
    assert first["terminal_count"] == 1


def test_stateless_loop_is_flagged_not_silently_left_unreported() -> None:
    loops = [LoopRow(1, "Missing state", "", "eric", "", "admin", "", "2026-07-01", "2026-07-01")]

    packet = build_weekly_packet(loops=loops, pending_decisions=[], today=TODAY)

    assert packet["aging"]["stateless_flags"][0]["id"] == 1
    assert packet["sections"]["STATE FLAGS"][0]["item"] == "Missing state"


def test_weekly_reset_job_reads_state_writes_one_packet(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "torben-open-loops.csv").write_text(
        "\n".join(
            [
                "id,item,state,owner,due,domain,note,created,updated",
                "1,Wire weekly reset,next-action,eric,,admin,,2026-07-06,2026-07-06",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (state / "torben-pending-decisions.json").write_text("[]", encoding="utf-8")
    (state / "torben-pattern-proposals.json").write_text(
        '{"schema":"torben.pattern-miner.v1","proposals":[{"id":"pattern-1","summary":"Automation candidate: test","status":"review_gated","support_count":4,"validation_score":0.8}]}',
        encoding="utf-8",
    )

    packet = build_weekly_reset_from_state(tmp_path)
    artifacts = write_weekly_reset_artifacts(packet, tmp_path)

    assert packet["sections"]["MUST NOT DROP"][0]["item"] == "Wire weekly reset"
    assert packet["sections"]["PENDING DECISIONS"][0]["handle"] == "pattern-1"
    assert artifacts["text"].startswith("Torben weekly reset - ")
    assert artifacts["text"].count("Torben weekly reset - ") == 1
    assert (state / "torben-weekly-reset-latest.json").exists()
