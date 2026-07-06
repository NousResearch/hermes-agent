from __future__ import annotations

import json
from pathlib import Path

from torben_capture import capture_auto_candidate, capture_signal
from torben_open_loops import load_loops


ERIC = "+15163843337"


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "torben-open-loops.csv",
        tmp_path / "torben-capture-dedupe.json",
        tmp_path / "torben-capture-confirmations.jsonl",
    )


def test_explicit_signal_capture_creates_loop_and_confirmation(tmp_path: Path) -> None:
    tracker, dedupe, confirmations = _paths(tmp_path)

    payload = capture_signal(
        text="task: call dentist about Christie forms",
        sender=ERIC,
        tracker_path=tracker,
        dedupe_path=dedupe,
        confirmations_path=confirmations,
        domain="health",
    )

    loops = load_loops(tracker)
    confirmation = json.loads(confirmations.read_text(encoding="utf-8").splitlines()[0])
    assert payload["status"] == "captured"
    assert loops[0].item == "call dentist about Christie forms"
    assert loops[0].state == "next-action"
    assert confirmation["loop_id"] == 1
    assert confirmation["text"] == "Captured loop #1 as next-action: call dentist about Christie forms"


def test_ambiguous_signal_capture_asks_one_clarifying_question(tmp_path: Path) -> None:
    tracker, dedupe, confirmations = _paths(tmp_path)

    payload = capture_signal(
        text="dentist",
        sender=ERIC,
        tracker_path=tracker,
        dedupe_path=dedupe,
        confirmations_path=confirmations,
    )

    assert payload == {
        "status": "clarify",
        "question": "What should I track, and what outcome do you want?",
        "wakeAgent": True,
    }
    assert not tracker.exists()
    assert not confirmations.exists()


def test_auto_capture_candidate_surfaces_only_above_gate(tmp_path: Path) -> None:
    tracker, dedupe, _ = _paths(tmp_path)

    withheld = capture_auto_candidate(
        text="Renew passport before September travel window",
        source="email",
        source_id="gmail:thread:1",
        score=0.4,
        tracker_path=tracker,
        dedupe_path=dedupe,
    )
    surfaced = capture_auto_candidate(
        text="Renew passport before September travel window",
        source="email",
        source_id="gmail:thread:1",
        score=0.8,
        tracker_path=tracker,
        dedupe_path=dedupe,
        domain="admin",
    )

    loops = load_loops(tracker)
    assert withheld["status"] == "withheld"
    assert surfaced["status"] == "candidate"
    assert loops[0].item == "Renew passport before September travel window"
    assert "score=0.8" in loops[0].note


def test_duplicate_capture_dedupes_to_existing_loop(tmp_path: Path) -> None:
    tracker, dedupe, confirmations = _paths(tmp_path)

    first = capture_signal(
        text="task: call dentist about Christie forms",
        sender=ERIC,
        tracker_path=tracker,
        dedupe_path=dedupe,
        confirmations_path=confirmations,
    )
    second = capture_signal(
        text="task: call dentist about Christie forms",
        sender=ERIC,
        tracker_path=tracker,
        dedupe_path=dedupe,
        confirmations_path=confirmations,
    )

    assert first["status"] == "captured"
    assert second["status"] == "duplicate"
    assert second["loop_id"] == 1
    assert len(load_loops(tracker)) == 1


def test_non_eric_sender_is_rejected(tmp_path: Path) -> None:
    tracker, dedupe, confirmations = _paths(tmp_path)

    payload = capture_signal(
        text="task: call dentist",
        sender="+15551234567",
        tracker_path=tracker,
        dedupe_path=dedupe,
        confirmations_path=confirmations,
    )

    assert payload["status"] == "rejected"
    assert payload["reason"] == "sender_not_eric"
    assert not tracker.exists()
