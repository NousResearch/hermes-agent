#!/usr/bin/env python3
"""Create a Signal Room review handoff tarball with a manifest."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "REVIEW_STATUS.md",
    "REVIEW_HUB.html",
    "EDITORIAL_REVIEW_PACKET.md",
    "EDITORIAL_SCREENING_GUIDE.md",
    "EDITORIAL_SCORECARD.json",
    "editorial_scorecard_gate.json",
    "package_integrity_scorecard.json",
    "review_package_index.json",
    "MOTION_QUALITY_PASS.md",
    "motion_quality_plan.json",
    "motion_frame_diff_report.json",
    "motion_smoothness_scorecard.json",
    "fee_machine_v2_draft_with_audio.mp4",
    "proof_retention_contact_sheet.svg",
    "audio_mux_scorecard.json",
    "audio_asset_scorecard.json",
    "retention_frame_scorecard.json",
    "render_asset_scorecard.json",
    "scaffold_scorecard.json",
    "video_env_scorecard.json",
    "pose_export_intake/README.md",
    "pose_export_intake/POSE_EXPORT_BRIEF.md",
    "pose_export_intake/required_frames.json",
    "pose_export_intake/rig_pass_manifest.template.json",
    "pose_export_intake/validate_pose_export.sh",
    "post_pose_pipeline_plan.json",
    "first_frame_candidates/first_frame_manifest.json",
    "first_frame_candidates/first_frame_review.md",
    "proof_frames/ordinary_bill.png",
    "proof_frames/number_split.png",
    "proof_frames/machine_reveal.png",
    "proof_frames/acting_read.png",
    "proof_frames/memory_anchor.png",
    "audio_assets/room_tone_bed.wav",
    "audio_assets/paper_bill_snap.wav",
    "fee_machine_v2_scaffold/index.html",
    "fee_machine_v2_scaffold/motion_primitives.js",
    "fee_machine_v2_scaffold/audio_cue_sheet.json",
    "fee_machine_v2_scaffold/scene_choreography.json",
    "fee_machine_v2_scaffold/retention_frame_plan.json",
)


OPTIONAL_PATTERNS = (
    "audio_assets/*.wav",
    "proof_frames/*.png",
    "fee_machine_v2_scaffold/*.json",
    "fee_machine_v2_scaffold/*.js",
    "fee_machine_v2_scaffold/*.md",
    "fee_machine_v2_scaffold/index.html",
    "pose_export_intake/**/*",
    "post_pose_pipeline_plan.json",
    "first_frame_candidates/*",
    "MOTION_QUALITY_PASS.md",
    "motion_quality_plan.json",
    "motion_frame_diff_report.json",
    "motion_smoothness_scorecard.json",
)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _objective_evidence(package_dir: Path) -> dict[str, Any]:
    index_path = package_dir / "review_package_index.json"
    if not index_path.exists():
        return {}
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return {}
    evidence = index.get("objective_evidence", {})
    return evidence if isinstance(evidence, dict) else {}


def collect_artifacts(package_dir: Path) -> list[str]:
    artifacts = set(REQUIRED_FILES)
    for pattern in OPTIONAL_PATTERNS:
        for path in package_dir.glob(pattern):
            if path.is_file():
                artifacts.add(_rel(path, package_dir))
    return sorted(artifacts)


def create_handoff_bundle(package_dir: Path, bundle_path: Path) -> dict[str, Any]:
    artifacts = collect_artifacts(package_dir)
    missing = [rel_path for rel_path in REQUIRED_FILES if not (package_dir / rel_path).exists()]
    manifest = {
        "status": "review-only",
        "public_release": False,
        "package_dir": str(package_dir),
        "bundle_path": str(bundle_path),
        "passed": not missing,
        "missing": missing,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "objective_evidence": _objective_evidence(package_dir),
        "primary_review_files": {
            "watchable_draft": "fee_machine_v2_draft_with_audio.mp4",
            "review_hub": "REVIEW_HUB.html",
            "contact_sheet": "proof_retention_contact_sheet.svg",
            "status_report": "REVIEW_STATUS.md",
            "motion_quality_plan": "MOTION_QUALITY_PASS.md",
            "motion_quality_plan_json": "motion_quality_plan.json",
            "scene_choreography": "fee_machine_v2_scaffold/scene_choreography.json",
            "editorial_review": "EDITORIAL_REVIEW_PACKET.md",
            "editorial_screening_guide": "EDITORIAL_SCREENING_GUIDE.md",
            "editorial_scorecard": "EDITORIAL_SCORECARD.json",
            "first_frame_review": "first_frame_candidates/first_frame_review.md",
            "pose_export_brief": "pose_export_intake/POSE_EXPORT_BRIEF.md",
            "pose_export_intake": "pose_export_intake/README.md",
        },
    }
    if missing:
        return manifest

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = json.dumps(manifest, indent=2).encode() + b"\n"
    with tarfile.open(bundle_path, "w:gz") as archive:
        for rel_path in artifacts:
            archive.add(package_dir / rel_path, arcname=f"signal-room-review/{rel_path}")
        info = tarfile.TarInfo("signal-room-review/handoff_manifest.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, fileobj=__import__("io").BytesIO(manifest_bytes))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest-out", type=Path)
    args = parser.parse_args()

    result = create_handoff_bundle(args.package_dir, args.out)
    text = json.dumps(result, indent=2)
    if args.manifest_out:
        args.manifest_out.write_text(text + "\n")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
