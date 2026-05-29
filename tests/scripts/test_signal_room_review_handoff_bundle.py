from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_review_handoff_bundle.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_review_handoff_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_package(root: Path) -> None:
    files = [
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
    ]
    for rel_path in files:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel_path == "review_package_index.json":
            path.write_text(
                json.dumps(
                    {
                        "objective_evidence": {
                            "choreography_beats_with_motion": 2,
                            "choreography_beats_without_motion": 0,
                            "choreography_beats_with_valid_review_frames": 2,
                            "choreography_beats_without_valid_review_frames": 0,
                        }
                    }
                )
            )
        else:
            path.write_bytes(b"bundle-test")


def test_review_handoff_bundle_builds_manifest_and_tarball(tmp_path: Path) -> None:
    module = load_module()
    package_dir = tmp_path / "review"
    bundle_path = tmp_path / "handoff.tar.gz"
    write_package(package_dir)

    result = module.create_handoff_bundle(package_dir, bundle_path)

    assert result["passed"] is True
    assert result["missing"] == []
    assert result["bundle_path"] == str(bundle_path)
    assert result["artifact_count"] >= 20
    assert bundle_path.exists()
    with tarfile.open(bundle_path, "r:gz") as archive:
        names = archive.getnames()
    assert "signal-room-review/REVIEW_STATUS.md" in names
    assert "signal-room-review/REVIEW_HUB.html" in names
    assert "signal-room-review/EDITORIAL_REVIEW_PACKET.md" in names
    assert "signal-room-review/EDITORIAL_SCREENING_GUIDE.md" in names
    assert "signal-room-review/EDITORIAL_SCORECARD.json" in names
    assert "signal-room-review/package_integrity_scorecard.json" in names
    assert "signal-room-review/MOTION_QUALITY_PASS.md" in names
    assert "signal-room-review/motion_quality_plan.json" in names
    assert "signal-room-review/motion_frame_diff_report.json" in names
    assert "signal-room-review/motion_smoothness_scorecard.json" in names
    assert "signal-room-review/fee_machine_v2_scaffold/scene_choreography.json" in names
    assert "signal-room-review/fee_machine_v2_scaffold/motion_primitives.js" in names
    assert "signal-room-review/pose_export_intake/README.md" in names
    assert "signal-room-review/pose_export_intake/POSE_EXPORT_BRIEF.md" in names
    assert "signal-room-review/post_pose_pipeline_plan.json" in names
    assert "signal-room-review/first_frame_candidates/first_frame_review.md" in names
    assert "signal-room-review/handoff_manifest.json" in names
    assert result["primary_review_files"]["scene_choreography"] == "fee_machine_v2_scaffold/scene_choreography.json"
    assert result["objective_evidence"] == {
        "choreography_beats_with_motion": 2,
        "choreography_beats_without_motion": 0,
        "choreography_beats_with_valid_review_frames": 2,
        "choreography_beats_without_valid_review_frames": 0,
    }
    with tarfile.open(bundle_path, "r:gz") as archive:
        embedded_manifest = json.load(archive.extractfile("signal-room-review/handoff_manifest.json"))
    assert embedded_manifest["objective_evidence"] == result["objective_evidence"]


def test_review_handoff_bundle_reports_missing_required_files(tmp_path: Path) -> None:
    module = load_module()
    package_dir = tmp_path / "review"
    package_dir.mkdir()

    result = module.create_handoff_bundle(package_dir, tmp_path / "handoff.tar.gz")

    assert result["passed"] is False
    assert "REVIEW_STATUS.md" in result["missing"]
    assert "fee_machine_v2_draft_with_audio.mp4" in result["missing"]
