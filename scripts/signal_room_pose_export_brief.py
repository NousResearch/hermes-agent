#!/usr/bin/env python3
"""Create an animator brief for Signal Room pose exports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POSE_NOTES = {
    "neutral_read.png": "calm documentary investigator, readable face, hands relaxed and visible",
    "bill_shock.png": "first reaction to the bill splitting, eyebrows up, torso held mature and grounded",
    "skeptical_point.png": "skeptical pointing pose; finger/hand line aims toward the fee machine area",
    "look_to_machine.png": "eyes and shoulders redirect toward the machine reveal without becoming cartoonish",
    "lean_weight_shift.png": "subtle weight shift for motion variety, feet and elbows uncropped",
    "mouth_closed.png": "closed-mouth hold for narration gaps, same head scale as the other mouth frames",
    "mouth_a.png": "open A mouth shape for speech, distinct from closed and O at phone size",
    "mouth_o.png": "rounded O mouth shape for speech, distinct from A at phone size",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _frame_lines(frames: list[str]) -> str:
    lines = []
    for frame in frames:
        note = POSE_NOTES.get(frame, "match the established adult documentary investigator tone")
        lines.append(f"- `{frame}` - {note}.")
    return "\n".join(lines)


def render_pose_export_brief(intake_dir: Path) -> str:
    required = read_json(intake_dir / "required_frames.json")
    template = read_json(intake_dir / "rig_pass_manifest.template.json")
    frames = [str(frame) for frame in required.get("frames", [])]
    review_items = "\n".join(f"- {item}" for item in required.get("review", []))
    candidate_name = str(template.get("candidate_name", "Suit_Male"))
    output_dir = f"pose_export_intake/character_frames/{candidate_name}"
    release_line = (
        "This package is not approved for public release."
        if template.get("public_release") is False
        else "Confirm release status before publishing."
    )

    return f"""# Signal Room Pose Export Brief

Candidate: `{candidate_name}`
Tool: `{template.get("render_tool", "Blender or Moho")}`
Status: review-only

{release_line}

## Delivery Folder

Place all exported files in:

```text
{output_dir}/
```

## Technical Requirements

- Format: {required.get("format", "transparent RGBA PNG")}
- Minimum character height: {required.get("minimum_height", 1400)}px
- Transparent background with clean alpha edges
- No crop on hair, hands, elbows, or feet
- Consistent camera scale across all exported frames

## Required Frames

{_frame_lines(frames)}

## Review Priorities

{review_items}

## Required Manifest

Copy `rig_pass_manifest.template.json` into the delivery folder as `rig_pass_manifest.json`, then fill in source rig, license status, render tool, and export notes.

## Validation Command

```bash
python scripts/signal_room_rig_acting_gate.py {output_dir} --out /tmp/signal-room-review/rig_acting_scorecard.json
```
"""


def write_pose_export_brief(intake_dir: Path, out: Path) -> dict[str, Any]:
    out.write_text(render_pose_export_brief(intake_dir))
    return {"passed": True, "output": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intake_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = write_pose_export_brief(args.intake_dir, args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
