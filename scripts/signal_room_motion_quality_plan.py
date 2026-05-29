#!/usr/bin/env python3
"""Create the Signal Room fee-machine motion quality upgrade plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _review_frames(package_dir: Path) -> list[dict[str, Any]]:
    frame_plan = _read_json(package_dir / "fee_machine_v2_scaffold" / "retention_frame_plan.json")
    frames = frame_plan.get("frames", [])
    if not isinstance(frames, list):
        return []
    return [
        {
            "id": str(frame.get("id", "")),
            "sample_time": frame.get("sample_time"),
            "upgrade_check": "must read as staged action, not a static card",
        }
        for frame in frames
        if isinstance(frame, dict) and frame.get("id")
    ]


def build_motion_quality_plan(package_dir: Path) -> dict[str, Any]:
    return {
        "target": "Signal Room Motion Quality Pass v1",
        "status": "review-only",
        "public_release": False,
        "package_dir": str(package_dir),
        "baseline_assessment": {
            "current_quality": "prototype-grade",
            "what_worked": [
                "deterministic HyperFrames render path works",
                "audio mux, proof frames, review hub, handoff bundle, and verifier work",
                "fee-machine concept is understandable enough for pipeline testing",
            ],
            "major_gaps": [
                "animation still reads as basic tweened layout",
                "character acting depends on pose swaps rather than staged performance",
                "machine motion needs continuous causal movement and smoother settles",
                "motion quality is not yet independently gated before editorial review",
            ],
        },
        "success_criteria": {
            "smoothness": {
                "minimum_fps": 30,
                "target_fps": 60,
                "no_static_hold_longer_than_seconds": 1.25,
                "requires_overlapping_transitions": True,
                "requires_secondary_motion": True,
            },
            "readability": {
                "phone_safe_zone": True,
                "fee_labels_read_without_narration": True,
                "machine_cause_effect_visible": True,
                "first_frame_understandable_within_seconds": 1.0,
            },
            "character_acting": {
                "approved_png_pose_exports_required": True,
                "minimum_distinct_poses": 5,
                "requires_anticipation_reaction_and_settle": True,
                "transparent_background_required": True,
            },
            "objective_choreography_evidence": {
                "all_choreography_beats_require_sampled_motion": True,
                "all_choreography_beats_require_valid_review_frame": True,
                "package_index_blocks_missing_motion_or_review_frame_evidence": True,
            },
            "editorial": {
                "scorecard_required": "EDITORIAL_SCORECARD.json",
                "overall_decision_required": "pass",
                "failure_conditions_must_be_false": True,
            },
        },
        "motion_primitives": [
            {
                "id": "fee_tag_reveal",
                "purpose": "Make fee split feel mechanically pushed into view.",
                "requirements": [
                    "staggered entrance with ease-out",
                    "small overshoot and settle",
                    "audio tick alignment within 2 frames",
                ],
            },
            {
                "id": "machine_reveal_panel",
                "purpose": "Reveal hidden mechanism as spatial cause, not a simple layer fade.",
                "requirements": [
                    "wall panel slide with parallax shadow",
                    "machine pre-roll begins before full reveal",
                    "camera push or depth shift during reveal",
                ],
            },
            {
                "id": "gear_drive_loop",
                "purpose": "Provide continuous causal motion during the mechanism section.",
                "requirements": [
                    "gears, belt, lever, and fee stack move in linked timing",
                    "no purely decorative motion",
                    "loop must not stutter at sampled frames",
                ],
            },
            {
                "id": "character_reaction_swap",
                "purpose": "Replace abrupt pose cuts with readable acting beats.",
                "requirements": [
                    "anticipation pose before reveal",
                    "reaction pose on fee split",
                    "pointing pose tied to machine action",
                    "settle pose for memory frame",
                ],
            },
            {
                "id": "memory_anchor_hold",
                "purpose": "End on a memorable visual that is still alive, not frozen.",
                "requirements": [
                    "slow camera hold",
                    "subtle machine decay",
                    "final fee stack pulse resolves before cut",
                ],
            },
        ],
        "verification_gates": [
            {
                "id": "motion_smoothness_gate",
                "evidence": [
                    "rendered MP4 exists",
                    "ffprobe confirms frame rate and duration",
                    "frame-difference report shows motion through mechanism section",
                    "beat_motion covers every choreography beat",
                    "no flagged static interval exceeds threshold",
                ],
            },
            {
                "id": "phone_readability_gate",
                "evidence": [
                    "proof frames are 1080x1920",
                    "choreography_frame_coverage maps every beat to a valid sampled proof frame",
                    "fee labels readable in sampled frames",
                    "first frame and memory anchor pass human review",
                ],
            },
            {
                "id": "rig_acting_gate",
                "evidence": [
                    "approved transparent PNG pose export candidate exists",
                    "rig_pass_manifest.json is present",
                    "required acting poses pass deterministic validation",
                ],
            },
            {
                "id": "editorial_readiness_gate",
                "evidence": [
                    "EDITORIAL_SCORECARD.json overall_decision is pass",
                    "all critical criteria pass",
                    "all failure conditions are false",
                ],
            },
        ],
        "implementation_sequence": [
            "Install approved Blender/Moho transparent PNG pose export candidate.",
            "Replace abrupt pose swaps with staged character-reaction primitive.",
            "Upgrade fee tag and machine reveal primitives with overlapping eased motion.",
            "Render a pose-refresh draft and sample proof frames.",
            "Run smoothness, readability, rig acting, and editorial gates.",
            "Regenerate review hub, handoff bundle, checksums, and transfer verifier output.",
        ],
        "review_frames": _review_frames(package_dir),
    }


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Signal Room Motion Quality Pass v1",
        "",
        f"**Status:** {plan.get('status', '')}",
        f"**Package:** `{plan.get('package_dir', '')}`",
        "",
        "## Baseline",
        f"- Current quality: {plan.get('baseline_assessment', {}).get('current_quality', '')}",
        "",
        "## Major Gaps",
    ]
    lines.extend(f"- {gap}" for gap in plan.get("baseline_assessment", {}).get("major_gaps", []))
    lines.extend(["", "## Motion Primitives"])
    for primitive in plan.get("motion_primitives", []):
        lines.append(f"- {primitive['id']}: {primitive['purpose']}")
    lines.extend(["", "## Verification Gates"])
    for gate in plan.get("verification_gates", []):
        lines.append(f"- {gate['id']}")
    lines.extend(["", "## Objective Choreography Evidence"])
    for key, value in plan.get("success_criteria", {}).get("objective_choreography_evidence", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Implementation Sequence"])
    lines.extend(f"{idx}. {step}" for idx, step in enumerate(plan.get("implementation_sequence", []), start=1))
    return "\n".join(lines) + "\n"


def write_motion_quality_plan(package_dir: Path, *, json_out: Path, markdown_out: Path) -> dict[str, Any]:
    plan = build_motion_quality_plan(package_dir)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(plan, indent=2) + "\n")
    markdown_out.write_text(render_markdown(plan))
    return {"passed": True, "json_out": str(json_out), "markdown_out": str(markdown_out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    result = write_motion_quality_plan(
        args.package_dir,
        json_out=args.json_out or args.package_dir / "motion_quality_plan.json",
        markdown_out=args.markdown_out or args.package_dir / "MOTION_QUALITY_PASS.md",
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
