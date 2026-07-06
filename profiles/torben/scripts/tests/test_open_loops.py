from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from torben_open_loops import (
    HEADER,
    add_loop,
    load_loops,
    overdue_loops,
    stale_waiting_loops,
    validate_loops,
    write_loops,
    LoopRow,
)


TODAY = date(2026, 7, 6)


def test_add_loop_assigns_incrementing_id_valid_state_and_dates(tmp_path: Path) -> None:
    path = tmp_path / "torben-open-loops.csv"

    first = add_loop(path=path, item="Call dentist", domain="health", today=TODAY)
    second = add_loop(path=path, item="Renew passport", state="waiting-on", due="2026-07-10", today=TODAY)

    assert first.id == 1
    assert second.id == 2
    assert second.state == "waiting-on"
    assert second.created == "2026-07-06"
    assert second.updated == "2026-07-06"
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(HEADER)


def test_validate_flags_loop_with_missing_state(tmp_path: Path) -> None:
    path = tmp_path / "torben-open-loops.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerow(
            {
                "id": "1",
                "item": "Call dentist",
                "state": "",
                "owner": "eric",
                "due": "",
                "domain": "health",
                "note": "",
                "created": "2026-07-06",
                "updated": "2026-07-06",
            }
        )

    invalid = validate_loops(load_loops(path))

    assert invalid == [{"id": 1, "field": "state", "reason": "invalid_or_missing_state"}]


def test_overdue_lists_active_due_loops_only(tmp_path: Path) -> None:
    rows = [
        LoopRow(1, "Pay bill", "next-action", "eric", "2026-07-06", "money", "", "2026-07-01", "2026-07-01"),
        LoopRow(2, "Done bill", "done", "eric", "2026-07-06", "money", "", "2026-07-01", "2026-07-01"),
        LoopRow(3, "Dropped bill", "dropped", "eric", "2026-07-06", "money", "", "2026-07-01", "2026-07-01"),
        LoopRow(4, "Future bill", "next-action", "eric", "2026-07-07", "money", "", "2026-07-01", "2026-07-01"),
    ]

    overdue = overdue_loops(rows, today=TODAY)

    assert [row.id for row in overdue] == [1]


def test_stale_waiting_loop_gets_drafted_nudge() -> None:
    rows = [
        LoopRow(1, "Alex reply", "waiting-on", "alex", "", "gtm", "", "2026-06-01", "2026-06-20"),
        LoopRow(2, "Recent reply", "waiting-on", "alex", "", "gtm", "", "2026-07-01", "2026-07-03"),
        LoopRow(3, "Not waiting", "next-action", "eric", "", "admin", "", "2026-06-01", "2026-06-01"),
    ]

    stale = stale_waiting_loops(rows, today=TODAY, stale_days=7)

    assert [row["id"] for row in stale] == ["1"]
    assert stale[0]["drafted_nudge"] == "Nudge alex: still waiting on Alex reply"


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "torben-open-loops.csv"
    rows = [LoopRow(1, "Call dentist", "next-action", "eric", "", "health", "bring insurance card", "2026-07-06", "2026-07-06")]

    write_loops(path, rows)

    loaded = load_loops(path)
    assert loaded == rows
