#!/usr/bin/env python3
"""Create a Signal Room editorial retention review packet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _checkline(label: str) -> str:
    return f"- [ ] Pass / [ ] Revise - {label}"


def render_editorial_review_packet(package_dir: Path) -> str:
    scaffold = package_dir / "fee_machine_v2_scaffold"
    frame_plan = read_json(scaffold / "retention_frame_plan.json")
    cue_sheet = read_json(scaffold / "audio_cue_sheet.json")
    package_index = read_json(package_dir / "review_package_index.json")

    lines = [
        "# Signal Room Editorial Review Packet",
        "",
        "Status: review-only",
        "",
        "## Primary Review Files",
        "- Watchable draft: `fee_machine_v2_draft_with_audio.mp4`",
        "- Proof contact sheet: `proof_retention_contact_sheet.svg`",
        "- Package status: `REVIEW_STATUS.md`",
        "- Pose-export intake: `pose_export_intake/README.md`",
        "",
        "## Current Blockers",
    ]
    blockers = package_index.get("blockers", [])
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- none")

    lines.extend(["", "## Immediate Pass/Revise Decision"])
    lines.append(_checkline("First frame makes the human problem clear."))
    lines.append(_checkline("Fee split reads without narration."))
    lines.append(_checkline("Machine reveal explains cause and effect."))
    lines.append(_checkline("Character acting is readable at phone size."))
    lines.append(_checkline("Final memory frame can be recalled after the video stops."))
    lines.append(_checkline("Sound design supports the mechanism without masking narration."))

    lines.extend(["", "## Failure Conditions"])
    for condition in frame_plan.get("failure_conditions", []):
        lines.append(f"- [ ] Not present - {condition}")

    lines.extend(["", "## Retention Frame Review"])
    for frame in frame_plan.get("frames", []):
        must_show = ", ".join(str(item) for item in frame.get("must_show", []))
        lines.extend(
            [
                f"### {frame.get('id')} - {frame.get('beat')}",
                f"- Sample: {frame.get('sample_time')}s",
                f"- Question: {frame.get('review_question')}",
                f"- Must show: {must_show}",
                "- Decision: [ ] Pass / [ ] Revise / [ ] Replace",
                "- Notes:",
                "",
            ]
        )

    lines.extend(["## Audio Cue Review"])
    for cue in cue_sheet.get("cues", []):
        lines.append(
            f"- `{cue.get('id')}` at {cue.get('start')}s for {cue.get('duration')}s: {cue.get('sound')}"
        )

    lines.extend(
        [
            "",
            "## Editorial Notes",
            "- Strongest retained visual:",
            "- Weakest unclear beat:",
            "- Required pose/export changes:",
            "- Required audio changes:",
            "- Decision owner/date:",
        ]
    )
    return "\n".join(lines) + "\n"


def write_editorial_review_packet(package_dir: Path, out: Path) -> dict[str, Any]:
    out.write_text(render_editorial_review_packet(package_dir))
    return {"passed": True, "output": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    result = write_editorial_review_packet(args.package_dir, args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
