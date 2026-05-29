#!/usr/bin/env python3
"""Create a review-only temporary Signal Room vector rig."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


POSES = {
    "neutral_read": {
        "title": "neutral_read",
        "expression": "neutral brow",
        "left_arm": "M438 508 C360 570 330 690 336 792",
        "right_arm": "M542 508 C616 572 640 686 634 782",
        "prop": "bill",
    },
    "bill_shock": {
        "title": "bill_shock",
        "expression": "raised brow",
        "left_arm": "M432 500 C340 532 292 630 300 738",
        "right_arm": "M548 500 C658 522 708 626 700 738",
        "prop": "bill",
    },
    "look_to_machine": {
        "title": "look_to_machine",
        "expression": "side glance",
        "left_arm": "M438 508 C366 572 340 680 352 778",
        "right_arm": "M542 508 C624 550 690 626 728 720",
        "prop": "machine glance",
    },
    "skeptical_point": {
        "title": "skeptical_point",
        "expression": "skeptical brow",
        "left_arm": "M438 508 C368 566 344 676 352 782",
        "right_arm": "M542 506 C690 492 804 476 932 430",
        "prop": "point-arm",
    },
    "slight_lean": {
        "title": "slight_lean",
        "expression": "settled concern",
        "left_arm": "M420 512 C350 590 330 704 348 820",
        "right_arm": "M526 512 C608 590 654 692 676 798",
        "prop": "lean",
    },
}


def pose_svg(pose: str, spec: dict[str, str]) -> str:
    eye_offset = 16 if spec["expression"] == "side glance" else 0
    brow_angle = -7 if "skeptical" in spec["expression"] else -2
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1500" role="img" aria-labelledby="title desc">
  <title>{spec["title"]}</title>
  <desc>review-only temporary vector rig for Signal Room adult animated documentary blocking</desc>
  <defs>
    <linearGradient id="suit" x1="0" x2="1">
      <stop offset="0" stop-color="#1a2731"/>
      <stop offset="1" stop-color="#263844"/>
    </linearGradient>
  </defs>
  <g id="review-only temporary vector rig" fill="none" stroke-linecap="round" stroke-linejoin="round">
    <ellipse cx="494" cy="1390" rx="260" ry="42" fill="rgba(0,0,0,.28)" stroke="none"/>
    <path d="M392 548 C326 702 302 930 282 1268 L708 1268 C686 922 662 706 592 548 Z" fill="url(#suit)" stroke="#d89a3a" stroke-width="10"/>
    <path d="M456 560 L492 694 L536 560" stroke="#f3eadb" stroke-width="16"/>
    <path d="M424 546 C442 640 466 718 492 792 C524 712 552 634 566 546" fill="#f3eadb" stroke="#8fa0aa" stroke-width="8"/>
    <path id="left-arm" d="{spec["left_arm"]}" stroke="#1d2d38" stroke-width="72"/>
    <path id="{spec["prop"]}" d="{spec["right_arm"]}" stroke="#1d2d38" stroke-width="72"/>
    <circle cx="492" cy="398" r="130" fill="#c99468" stroke="#5d4030" stroke-width="8"/>
    <path d="M374 350 C404 220 592 214 632 344 C576 288 450 288 374 350 Z" fill="#20252a" stroke="#20252a" stroke-width="8"/>
    <path d="M430 372 L472 366" stroke="#20252a" stroke-width="10" transform="rotate({brow_angle} 450 370)"/>
    <path d="M528 366 L570 372" stroke="#20252a" stroke-width="10"/>
    <circle cx="{450 + eye_offset}" cy="410" r="12" fill="#101820" stroke="none"/>
    <circle cx="{548 + eye_offset}" cy="410" r="12" fill="#101820" stroke="none"/>
    <path d="M450 474 C486 500 526 498 558 472" stroke="#5d4030" stroke-width="10"/>
    <rect x="278" y="730" width="196" height="128" rx="12" fill="#f3eadb" stroke="#8fa0aa" stroke-width="8"/>
    <path d="M302 770 H444 M302 812 H406" stroke="#c85f3f" stroke-width="8"/>
    <path d="M354 1268 L338 1450 M618 1268 L648 1450" stroke="#1a2731" stroke-width="76"/>
  </g>
</svg>
"""


def create_temp_rig(out_dir: Path, *, force: bool = False) -> dict[str, Any]:
    if out_dir.exists():
        if not force:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)

    poses_dir = out_dir / "poses"
    poses_dir.mkdir(parents=True)
    for pose, spec in POSES.items():
        (poses_dir / f"{pose}.svg").write_text(pose_svg(pose, spec))

    manifest = {
        "title": "Signal Room Adult Investigator Temporary Vector Rig",
        "status": "review-only",
        "public_release": False,
        "pose_count": len(POSES),
        "poses": list(POSES),
        "replacement_required": "Blender/Moho adult character candidate",
        "use": "temporary choreography and retention-review scaffold only",
    }
    (out_dir / "rig_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="replace existing output directory")
    args = parser.parse_args()
    manifest = create_temp_rig(args.out, force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
