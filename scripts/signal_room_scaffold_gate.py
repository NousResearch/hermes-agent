#!/usr/bin/env python3
"""Offline gate for a Signal Room fee-machine scaffold package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COMPOSITION_ID = "fee-machine-v2-review"
REQUIRED_FILES = (
    "index.html",
    "manifest.json",
    "DESIGN.md",
    ".hyperframes/expanded-prompt.md",
    "audio_cue_sheet.json",
    "motion_primitives.json",
    "motion_primitives.js",
    "scene_choreography.json",
    "retention_frame_plan.json",
    "source_rig_manifest.json",
)
REQUIRED_POSES = (
    "neutral_read",
    "bill_shock",
    "look_to_machine",
    "skeptical_point",
    "slight_lean",
)
REQUIRED_REVIEW_ITEMS = (
    "run scripts/signal_room_video_env_gate.py before local render attempts",
    "run npx hyperframes lint",
    "run npx hyperframes inspect",
    "source or design SFX from audio_cue_sheet.json",
    "run scripts/signal_room_retention_frame_gate.py on sampled proof frames",
    "replace v0 vector rig if Blender/Moho candidate passes",
)
REQUIRED_MOTION_PRIMITIVES = {
    "continuous_drift": "SignalRoomMotion.continuousDrift(",
    "ambient_loop": "SignalRoomMotion.ambientLoop(",
    "character_micro_acting": "SignalRoomMotion.microActing(",
    "pose_transition": "SignalRoomMotion.poseTransition(",
    "fee_stack_simmer": "SignalRoomMotion.feeStackSimmer(",
    "machine_idle": "SignalRoomMotion.machineIdle(",
    "machine_causality": "SignalRoomMotion.machineCausality(",
}
MINIMUM_BEAT_OVERLAP_SECONDS = 0.10


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        errors.append(f"missing required file: {path.name}")
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {path.name}: {exc.msg}")
    return {}


def evaluate_scaffold(scaffold_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    for rel_path in REQUIRED_FILES:
        if not (scaffold_dir / rel_path).exists():
            errors.append(f"missing required file: {rel_path}")

    manifest = _read_json(scaffold_dir / "manifest.json", errors)
    if manifest:
        if manifest.get("status") != "review-only":
            errors.append("manifest status must be review-only")
        if manifest.get("public_release") is not False:
            errors.append("manifest public_release must be false")
        review_items = set(manifest.get("required_review", []))
        for item in REQUIRED_REVIEW_ITEMS:
            if item not in review_items:
                errors.append(f"manifest missing required review item: {item}")

    index_path = scaffold_dir / "index.html"
    html = index_path.read_text() if index_path.exists() else ""
    if html:
        if f'data-composition-id="{COMPOSITION_ID}"' not in html:
            errors.append(f"index.html missing composition id: {COMPOSITION_ID}")
        if f'window.__timelines["{COMPOSITION_ID}"]' not in html:
            errors.append(f"index.html missing registered timeline: {COMPOSITION_ID}")
        if 'src="motion_primitives.js"' not in html:
            errors.append("index.html missing reusable motion library script: motion_primitives.js")
        for pose in REQUIRED_POSES:
            pose_ref = f"assets/poses/{pose}.svg"
            if pose_ref not in html:
                errors.append(f"index.html missing pose reference: {pose_ref}")

    for pose in REQUIRED_POSES:
        rel_path = f"assets/poses/{pose}.svg"
        if not (scaffold_dir / rel_path).exists():
            errors.append(f"missing required pose asset: {rel_path}")

    audio = _read_json(scaffold_dir / "audio_cue_sheet.json", errors)
    if audio and not audio.get("cues"):
        errors.append("audio_cue_sheet.json must include cues")

    motion_primitives = _read_json(scaffold_dir / "motion_primitives.json", errors)
    motion_library = (scaffold_dir / "motion_primitives.js").read_text() if (scaffold_dir / "motion_primitives.js").exists() else ""
    if motion_library and "window.SignalRoomMotion" not in motion_library:
        errors.append("motion_primitives.js missing window.SignalRoomMotion export")
    if motion_primitives:
        if motion_primitives.get("status") != "review-only":
            errors.append("motion_primitives.json status must be review-only")
        if motion_primitives.get("public_release") is not False:
            errors.append("motion_primitives.json public_release must be false")
        primitive_ids = {
            str(primitive.get("id"))
            for primitive in motion_primitives.get("primitives", [])
            if isinstance(primitive, dict)
        }
        for primitive_id, marker in REQUIRED_MOTION_PRIMITIVES.items():
            if primitive_id not in primitive_ids:
                errors.append(f"motion_primitives.json missing primitive: {primitive_id}")
            if html and marker not in html:
                errors.append(f"index.html missing motion primitive marker: {marker}")
            if motion_library and marker.replace("SignalRoomMotion.", "") not in motion_library:
                errors.append(f"motion_primitives.js missing primitive function: {marker}")

    choreography = _read_json(scaffold_dir / "scene_choreography.json", errors)
    if choreography:
        if choreography.get("status") != "review-only":
            errors.append("scene_choreography.json status must be review-only")
        if choreography.get("public_release") is not False:
            errors.append("scene_choreography.json public_release must be false")
        if choreography.get("composition_id") != COMPOSITION_ID:
            errors.append(f"scene_choreography.json composition_id must be {COMPOSITION_ID}")
        beats = choreography.get("beats", [])
        if not isinstance(beats, list) or len(beats) < 5:
            errors.append("scene_choreography.json must include at least five staged beats")
        if isinstance(beats, list):
            for idx, beat in enumerate(beats):
                if not isinstance(beat, dict):
                    errors.append("scene_choreography.json contains non-object beat")
                    continue
                beat_id = str(beat.get("id") or f"beat_{idx + 1}")
                for field in ("acting_objective", "primary_motion"):
                    if not beat.get(field):
                        errors.append(f"scene choreography beat {beat_id} missing {field}")
                if idx > 0 and not beat.get("acting_phases"):
                    errors.append(f"scene choreography beat {beat_id} missing acting_phases")
                if idx > 0:
                    try:
                        overlap = float(beat.get("overlap_in_seconds", 0))
                    except (TypeError, ValueError):
                        overlap = 0.0
                    if overlap < MINIMUM_BEAT_OVERLAP_SECONDS:
                        errors.append(
                            f"scene choreography beat {beat_id} must overlap adjacent motion "
                            f"by at least {MINIMUM_BEAT_OVERLAP_SECONDS:.2f}s"
                        )

    frame_plan = _read_json(scaffold_dir / "retention_frame_plan.json", errors)
    if frame_plan and len(frame_plan.get("frames", [])) < 5:
        errors.append("retention_frame_plan.json must include at least five frames")

    return {
        "passed": not errors,
        "errors": errors,
        "scaffold_dir": str(scaffold_dir),
        "composition_id": COMPOSITION_ID,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scaffold_dir", type=Path)
    parser.add_argument("--out", type=Path, help="write gate result JSON to this path")
    args = parser.parse_args()

    result = evaluate_scaffold(args.scaffold_dir)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
