from __future__ import annotations

import json
from pathlib import Path

from torben_capture_gate import gate_auto_capture_candidate
from torben_open_loops import load_loops


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "torben-open-loops.csv",
        tmp_path / "torben-capture-dedupe.json",
        tmp_path / "torben-capture-gate-retained.jsonl",
    )


def test_above_threshold_candidate_surfaces_to_eric_as_loop(tmp_path: Path) -> None:
    tracker, dedupe, retained = _paths(tmp_path)

    payload = gate_auto_capture_candidate(
        text="Renew passport before travel",
        source="email",
        source_id="gmail:thread:1",
        support_count=4,
        confidence=0.9,
        tracker_path=tracker,
        dedupe_path=dedupe,
        retained_path=retained,
        domain="admin",
    )

    loops = load_loops(tracker)
    assert payload["status"] == "candidate"
    assert payload["validation_score"] >= 0.7
    assert loops[0].item == "Renew passport before travel"
    assert not retained.exists()


def test_below_threshold_candidate_is_withheld_and_retained(tmp_path: Path) -> None:
    tracker, dedupe, retained = _paths(tmp_path)

    payload = gate_auto_capture_candidate(
        text="Check one-off coupon",
        source="email",
        source_id="gmail:thread:2",
        support_count=1,
        confidence=0.6,
        tracker_path=tracker,
        dedupe_path=dedupe,
        retained_path=retained,
    )

    retained_payload = json.loads(retained.read_text(encoding="utf-8").splitlines()[0])
    assert payload["status"] == "withheld"
    assert payload["reason"] == "below_validation_threshold"
    assert retained_payload["text"] == "Check one-off coupon"
    assert not tracker.exists()


def test_duplicate_above_threshold_candidate_dedupes_against_existing_loop(tmp_path: Path) -> None:
    tracker, dedupe, retained = _paths(tmp_path)
    kwargs = {
        "text": "Renew passport before travel",
        "source": "email",
        "source_id": "gmail:thread:1",
        "support_count": 4,
        "confidence": 0.9,
        "tracker_path": tracker,
        "dedupe_path": dedupe,
        "retained_path": retained,
        "domain": "admin",
    }

    first = gate_auto_capture_candidate(**kwargs)
    second = gate_auto_capture_candidate(**kwargs)

    assert first["status"] == "candidate"
    assert second["status"] == "duplicate"
    assert second["capture"]["loop_id"] == 1
    assert len(load_loops(tracker)) == 1
