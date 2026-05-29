#!/usr/bin/env python3
"""Build a Signal Room review package index from gate scorecards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCORECARDS = {
    "video_env": "video_env_scorecard.json",
    "scaffold": "scaffold_scorecard.json",
    "render_asset": "render_asset_scorecard.json",
    "motion_smoothness": "motion_smoothness_scorecard.json",
    "audio_mux": "audio_mux_scorecard.json",
    "audio_assets": "audio_asset_scorecard.json",
    "proof_frames": "retention_frame_scorecard.json",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _placeholder_count(items: list[dict[str, Any]]) -> int:
    return sum(1 for item in items if item.get("placeholder") is True)


def _objective_evidence(scorecards: dict[str, Any]) -> dict[str, int]:
    beat_motion = scorecards.get("motion_smoothness", {}).get("beat_motion", [])
    primitive_motion = scorecards.get("motion_smoothness", {}).get("primitive_motion", [])
    frame_coverage = scorecards.get("proof_frames", {}).get("choreography_frame_coverage", [])
    return {
        "choreography_beats_with_motion": sum(1 for beat in beat_motion if beat.get("passed") is True),
        "choreography_beats_without_motion": sum(1 for beat in beat_motion if beat.get("passed") is not True),
        "choreography_beats_with_valid_review_frames": sum(
            1 for beat in frame_coverage if beat.get("frame_valid") is True
        ),
        "choreography_beats_without_valid_review_frames": sum(
            1 for beat in frame_coverage if beat.get("frame_valid") is not True
        ),
        "motion_primitives_with_evidence": sum(
            1 for primitive in primitive_motion if primitive.get("passed") is True
        ),
        "motion_primitives_without_evidence": sum(
            1 for primitive in primitive_motion if primitive.get("passed") is not True
        ),
    }


def build_review_package_index(package_dir: Path) -> dict[str, Any]:
    blockers: list[str] = []
    scorecards: dict[str, Any] = {}

    for name, filename in SCORECARDS.items():
        data = _read_json(package_dir / filename)
        if data is None:
            blockers.append(f"missing scorecard: {filename}")
            scorecards[name] = {"passed": False, "missing": True}
            continue
        scorecards[name] = data
        if data.get("passed") is not True:
            for blocker in data.get("blockers", []):
                blockers.append(str(blocker))
            for error in data.get("errors", []):
                blockers.append(f"{name}: {error}")

    audio_assets = scorecards.get("audio_assets", {}).get("manifest", {}).get("assets", [])
    proof_frames = scorecards.get("proof_frames", {}).get("manifest", {}).get("frames", [])
    placeholder_counts = {
        "audio_assets": _placeholder_count(audio_assets),
        "proof_frames": _placeholder_count(proof_frames),
    }
    if placeholder_counts["audio_assets"]:
        blockers.append("placeholder audio assets must be replaced before editorial review")
    if placeholder_counts["proof_frames"]:
        blockers.append("placeholder proof frames must be replaced before visual review")
    objective_evidence = _objective_evidence(scorecards)
    missing_motion = objective_evidence["choreography_beats_without_motion"]
    missing_review_frames = objective_evidence["choreography_beats_without_valid_review_frames"]
    missing_primitive_motion = objective_evidence["motion_primitives_without_evidence"]
    if missing_motion:
        blockers.append(f"choreography motion evidence missing for {missing_motion} beat(s)")
    if missing_review_frames:
        blockers.append(f"choreography review-frame evidence missing for {missing_review_frames} beat(s)")
    if missing_primitive_motion:
        blockers.append(f"motion primitive evidence missing for {missing_primitive_motion} primitive(s)")

    return {
        "passed": not blockers,
        "package_dir": str(package_dir),
        "blockers": blockers,
        "placeholder_counts": placeholder_counts,
        "objective_evidence": objective_evidence,
        "scorecards": scorecards,
    }


def render_markdown_report(result: dict[str, Any]) -> str:
    status = "passed" if result.get("passed") else "blocked"
    lines = [
        "# Signal Room Review Package Status",
        "",
        f"**Status:** {status}",
        f"**Package:** `{result.get('package_dir', '')}`",
        "",
        "## Scorecards",
    ]
    for name, scorecard in result.get("scorecards", {}).items():
        card_status = "passed" if scorecard.get("passed") is True else "blocked"
        lines.append(f"- {name}: {card_status}")

    lines.extend(["", "## Placeholders"])
    for name, count in result.get("placeholder_counts", {}).items():
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Objective Choreography Evidence"])
    evidence = result.get("objective_evidence", {})
    lines.append(f"- choreography beats with motion: {evidence.get('choreography_beats_with_motion', 0)}")
    lines.append(f"- choreography beats without motion: {evidence.get('choreography_beats_without_motion', 0)}")
    lines.append(
        "- choreography beats with valid review frames: "
        f"{evidence.get('choreography_beats_with_valid_review_frames', 0)}"
    )
    lines.append(
        "- choreography beats without valid review frames: "
        f"{evidence.get('choreography_beats_without_valid_review_frames', 0)}"
    )
    lines.append(f"- motion primitives with evidence: {evidence.get('motion_primitives_with_evidence', 0)}")
    lines.append(f"- motion primitives without evidence: {evidence.get('motion_primitives_without_evidence', 0)}")

    lines.extend(["", "## Blockers"])
    blockers = result.get("blockers", [])
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- none")

    lines.extend(["", "## Next Actions"])
    next_actions = []
    if result.get("placeholder_counts", {}).get("audio_assets", 0):
        next_actions.append("Replace placeholder audio assets before editorial audio review.")
    if result.get("placeholder_counts", {}).get("proof_frames", 0):
        next_actions.append("Replace placeholder proof frames with sampled HyperFrames exports before visual review.")
    if result.get("scorecards", {}).get("video_env", {}).get("passed") is not True:
        next_actions.append("Resolve local character pose rendering or import approved Blender/Moho pose exports.")
    if not next_actions:
        next_actions.append("Run editorial visual review against the draft render and contact sheet.")
    lines.extend(f"- {action}" for action in next_actions)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--out", type=Path, help="write review package index JSON")
    parser.add_argument("--markdown-out", type=Path, help="write human-readable Markdown report")
    args = parser.parse_args()

    result = build_review_package_index(args.package_dir)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    if args.markdown_out:
        args.markdown_out.write_text(render_markdown_report(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
