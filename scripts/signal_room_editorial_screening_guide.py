#!/usr/bin/env python3
"""Create a timestamped editorial screening guide."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CRITICAL_TO_FRAME = {
    "first_frame": "first-frame candidates",
    "fee_split": "retention_frames.number_split",
    "machine_reveal": "retention_frames.machine_reveal",
    "acting_read": "retention_frames.acting_read",
    "memory_anchor": "retention_frames.memory_anchor",
    "audio_support": "audio cue review",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fmt_time(value: Any) -> str:
    try:
        return f"{float(value):g}s"
    except (TypeError, ValueError):
        return "unknown"


def render_screening_guide(package_dir: Path) -> str:
    frame_plan = read_json(package_dir / "fee_machine_v2_scaffold" / "retention_frame_plan.json")
    choreography = read_json(package_dir / "fee_machine_v2_scaffold" / "scene_choreography.json")
    first_manifest = read_json(package_dir / "first_frame_candidates" / "first_frame_manifest.json")
    scorecard = read_json(package_dir / "EDITORIAL_SCORECARD.json")

    lines = [
        "# Signal Room Editorial Screening Guide",
        "",
        "Status: review-only",
        "",
        "Watch file: `fee_machine_v2_draft_with_audio.mp4`",
        "Scorecard to fill: `EDITORIAL_SCORECARD.json`",
        "",
        "## Review Order",
        "",
        "1. Watch once without pausing for retained story clarity.",
        "2. Review the first-frame candidates before judging the opening.",
        "3. Scrub to each retention sample time and fill the matching scorecard entry.",
        "4. Mark every failure condition as `false` before approving.",
        "",
        "## First-Frame Candidates",
    ]

    for candidate in first_manifest.get("candidates", []):
        lines.extend(
            [
                f"### `{candidate.get('id')}` at {_fmt_time(candidate.get('sample_time'))}",
                f"- Image: `first_frame_candidates/{candidate.get('filename')}`",
                f"- Question: {candidate.get('question')}",
                f"- Fill: `first_frame_candidates.{candidate.get('id')}`",
                "- Accept if: the human problem and starting object read before narration.",
                "",
            ]
        )

    lines.extend(["## Critical Criteria Map", ""])
    for item in scorecard.get("critical_criteria", []):
        criterion_id = str(item.get("id"))
        lines.append(
            f"- `critical_criteria.{criterion_id}` - {item.get('label')} "
            f"Evidence: `{CRITICAL_TO_FRAME.get(criterion_id, 'review packet')}`."
        )

    lines.extend(["", "## Retention Frame Checks", ""])
    for frame in frame_plan.get("frames", []):
        must_show = ", ".join(str(item) for item in frame.get("must_show", []))
        frame_id = str(frame.get("id"))
        lines.extend(
            [
                f"### `{frame_id}` at {_fmt_time(frame.get('sample_time'))}",
                f"- Beat: {frame.get('beat')}",
                f"- Question: {frame.get('review_question')}",
                f"- Must show: {must_show}",
                f"- Fill: `retention_frames.{frame_id}`",
                "",
            ]
        )

    lines.extend(["## Choreography Beat Checks", ""])
    beats = scorecard.get("choreography_beats") or choreography.get("beats", [])
    for beat in beats:
        beat_id = str(beat.get("id"))
        lines.extend(
            [
                f"### `{beat_id}`",
                f"- Time: {_fmt_time(beat.get('start'))}-{_fmt_time(beat.get('end'))}",
                f"- Acting objective: {beat.get('acting_objective')}",
                f"- Primary motion: {beat.get('primary_motion')}",
                f"- Evidence frame: `retention_frames.{beat.get('review_frame')}`",
                f"- Fill: `choreography_beats.{beat_id}`",
                "- Accept if: the acting objective is readable and the primary motion clarifies the mechanism.",
                "",
            ]
        )

    lines.extend(["## Failure Conditions", ""])
    for condition in frame_plan.get("failure_conditions", []):
        lines.append(f"- Confirm absent: {condition}")

    lines.extend(
        [
            "",
            "## Approval Rule",
            "",
            "Approve only when `overall_decision` is `pass`, all critical criteria are `pass`, one or more first-frame candidates are `keep` or `revise`, all retention frames are `pass`, all choreography beats are `pass`, and every failure condition is `false`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_screening_guide(package_dir: Path, out: Path) -> dict[str, Any]:
    out.write_text(render_screening_guide(package_dir))
    return {"passed": True, "output": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = write_screening_guide(args.package_dir, args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
