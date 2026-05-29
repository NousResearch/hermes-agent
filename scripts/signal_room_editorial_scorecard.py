#!/usr/bin/env python3
"""Create and gate structured Signal Room editorial scorecards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CRITICAL_CRITERIA = (
    ("first_frame", "First frame makes the human problem clear."),
    ("fee_split", "Fee split reads without narration."),
    ("machine_reveal", "Machine reveal explains cause and effect."),
    ("acting_read", "Character acting is readable at phone size."),
    ("memory_anchor", "Final memory frame can be recalled after the video stops."),
    ("audio_support", "Sound design supports the mechanism without masking narration."),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_scorecard_template(package_dir: Path) -> dict[str, Any]:
    frame_plan = read_json(package_dir / "fee_machine_v2_scaffold" / "retention_frame_plan.json")
    choreography = read_json(package_dir / "fee_machine_v2_scaffold" / "scene_choreography.json")
    first_frame_manifest = read_json(package_dir / "first_frame_candidates" / "first_frame_manifest.json")
    return {
        "status": "review-required",
        "public_release": False,
        "overall_decision": "pending",
        "allowed_decisions": {
            "overall": ["pending", "pass", "revise", "reject"],
            "critical": ["pending", "pass", "revise"],
            "first_frame": ["pending", "keep", "revise", "reject"],
            "retention_frame": ["pending", "pass", "revise", "replace"],
        },
        "critical_criteria": [
            {"id": criterion_id, "label": label, "decision": "pending", "notes": ""}
            for criterion_id, label in CRITICAL_CRITERIA
        ],
        "first_frame_candidates": [
            {
                "id": str(candidate["id"]),
                "decision": "pending",
                "notes": "",
            }
            for candidate in first_frame_manifest.get("candidates", [])
        ],
        "retention_frames": [
            {
                "id": str(frame["id"]),
                "question": frame.get("review_question"),
                "decision": "pending",
                "notes": "",
            }
            for frame in frame_plan.get("frames", [])
        ],
        "choreography_beats": [
            {
                "id": str(beat["id"]),
                "acting_objective": beat.get("acting_objective"),
                "primary_motion": beat.get("primary_motion"),
                "review_frame": beat.get("review_frame"),
                "decision": "pending",
                "notes": "",
            }
            for beat in choreography.get("beats", [])
        ],
        "failure_conditions": [
            {"condition": str(condition), "present": None, "notes": ""}
            for condition in frame_plan.get("failure_conditions", [])
        ],
        "reviewer": "",
        "review_date": "",
    }


def evaluate_scorecard(scorecard: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if scorecard.get("overall_decision") != "pass":
        errors.append("overall_decision must be pass")
    for item in scorecard.get("critical_criteria", []):
        decision = item.get("decision")
        if decision != "pass":
            errors.append(f"critical criterion {item.get('id')} is {decision}")
    for item in scorecard.get("first_frame_candidates", []):
        decision = item.get("decision")
        if decision not in ("keep", "revise"):
            errors.append(f"first-frame candidate {item.get('id')} is {decision}")
    for item in scorecard.get("retention_frames", []):
        decision = item.get("decision")
        if decision != "pass":
            errors.append(f"retention frame {item.get('id')} is {decision}")
    for item in scorecard.get("choreography_beats", []):
        decision = item.get("decision")
        if decision != "pass":
            errors.append(f"choreography beat {item.get('id')} is {decision}")
    for item in scorecard.get("failure_conditions", []):
        if item.get("present") is not False:
            errors.append(f"failure condition present: {item.get('condition')}")
    return {"passed": not errors, "errors": errors}


def write_scorecard_template(package_dir: Path, out: Path) -> dict[str, Any]:
    scorecard = build_scorecard_template(package_dir)
    out.write_text(json.dumps(scorecard, indent=2) + "\n")
    return {"passed": True, "output": str(out)}


def write_gate_result(scorecard_path: Path, out: Path) -> dict[str, Any]:
    result = evaluate_scorecard(read_json(scorecard_path))
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser("template")
    template_parser.add_argument("package_dir", type=Path)
    template_parser.add_argument("--out", required=True, type=Path)

    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument("scorecard", type=Path)
    gate_parser.add_argument("--out", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "template":
        result = write_scorecard_template(args.package_dir, args.out)
    else:
        result = write_gate_result(args.scorecard, args.out)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
