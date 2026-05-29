#!/usr/bin/env python3
"""Create a Signal Room Blender/Moho pose export intake pack."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


REQUIRED_FRAMES = (
    "neutral_read.png",
    "bill_shock.png",
    "skeptical_point.png",
    "look_to_machine.png",
    "lean_weight_shift.png",
    "mouth_closed.png",
    "mouth_a.png",
    "mouth_o.png",
)
MIN_HEIGHT = 1400


def _readme(out_dir: Path, candidate_name: str) -> str:
    candidate_dir = f"character_frames/{candidate_name}"
    candidate_path = out_dir / candidate_dir
    return f"""# Signal Room Pose Export Intake

Status: review-only

Export the adult investigator rig poses from Blender or Moho into:

`{candidate_dir}/`

## Required Files

Every frame must be a transparent RGBA PNG, at least {MIN_HEIGHT}px tall, with no crop on hair, hands, elbows, or feet.

{chr(10).join(f"- `{frame}`" for frame in REQUIRED_FRAMES)}

Copy `rig_pass_manifest.template.json` into `{candidate_dir}/rig_pass_manifest.json` and fill in the render tool, source rig, license status, and export notes.

## Validation

From the repo root, run:

```bash
python scripts/signal_room_rig_acting_gate.py \\
  {candidate_path} \\
  --out {out_dir / "rig_acting_scorecard.json"}
```

If the gate passes, build the acting contact sheet:

```bash
python scripts/signal_room_contact_sheet.py \\
  {candidate_path} \\
  --out {out_dir / f"{candidate_name}_contact_sheet.svg"}
```

Reject the export if the pose changes do not read at phone size or if the skeptical point does not aim toward the fee machine area.
"""


def _validate_script(out_dir: Path, candidate_name: str) -> str:
    candidate_dir = out_dir / "character_frames" / candidate_name
    return f"""#!/usr/bin/env bash
set -euo pipefail

python scripts/signal_room_rig_acting_gate.py \\
  {candidate_dir} \\
  --out {out_dir / "rig_acting_scorecard.json"}

python scripts/signal_room_contact_sheet.py \\
  {candidate_dir} \\
  --out {out_dir / f"{candidate_name}_contact_sheet.svg"}
"""


def create_pose_export_intake_pack(
    out_dir: Path,
    *,
    candidate_name: str = "Suit_Male",
    force: bool = False,
) -> dict[str, Any]:
    if out_dir.exists():
        if not force:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)

    candidate_dir = out_dir / "character_frames" / candidate_name
    candidate_dir.mkdir(parents=True)
    (candidate_dir / ".gitkeep").write_text("")

    manifest_template = {
        "title": "Signal Room Adult Investigator Pose Export",
        "status": "review-only",
        "public_release": False,
        "candidate_name": candidate_name,
        "render_tool": "Blender or Moho",
        "source_rig": "REQUIRED: rig/source file name",
        "license_status": "REQUIRED: source/license notes",
        "export_notes": "REQUIRED: scale, camera, alpha, and crop notes",
        "required_frames": list(REQUIRED_FRAMES),
        "minimum_height": MIN_HEIGHT,
    }
    required_frames = {
        "frames": list(REQUIRED_FRAMES),
        "format": "transparent RGBA PNG",
        "minimum_height": MIN_HEIGHT,
        "review": [
            "adult documentary tone",
            "phone-readable face and hands",
            "skeptical point aims at the fee machine area",
            "mouth shapes are distinct",
            "no transparent edge halos",
        ],
    }
    manifest = {
        "status": "review-only",
        "public_release": False,
        "candidate_name": candidate_name,
        "required_frame_count": len(REQUIRED_FRAMES),
        "candidate_dir": str(candidate_dir),
        "validation": {
            "rig_gate": "python scripts/signal_room_rig_acting_gate.py",
            "contact_sheet": "python scripts/signal_room_contact_sheet.py",
        },
    }

    (out_dir / "README.md").write_text(_readme(out_dir, candidate_name))
    (out_dir / "rig_pass_manifest.template.json").write_text(json.dumps(manifest_template, indent=2) + "\n")
    (out_dir / "required_frames.json").write_text(json.dumps(required_frames, indent=2) + "\n")
    validate_script = out_dir / "validate_pose_export.sh"
    validate_script.write_text(_validate_script(out_dir, candidate_name))
    validate_script.chmod(0o755)
    (out_dir / "pose_export_intake_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidate-name", default="Suit_Male")
    parser.add_argument("--force", action="store_true", help="replace existing output directory")
    args = parser.parse_args()

    manifest = create_pose_export_intake_pack(args.out, candidate_name=args.candidate_name, force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
