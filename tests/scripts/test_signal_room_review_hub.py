from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_review_hub.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_review_hub", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_package(root: Path) -> None:
    (root / "handoff_manifest.json").write_text(
        json.dumps(
            {
                "primary_review_files": {
                    "watchable_draft": "fee_machine_v2_draft_with_audio.mp4",
                    "contact_sheet": "proof_retention_contact_sheet.svg",
                    "motion_quality_plan": "MOTION_QUALITY_PASS.md",
                    "motion_quality_plan_json": "motion_quality_plan.json",
                    "scene_choreography": "fee_machine_v2_scaffold/scene_choreography.json",
                    "editorial_review": "EDITORIAL_REVIEW_PACKET.md",
                    "editorial_screening_guide": "EDITORIAL_SCREENING_GUIDE.md",
                    "editorial_scorecard": "EDITORIAL_SCORECARD.json",
                    "first_frame_review": "first_frame_candidates/first_frame_review.md",
                    "pose_export_brief": "pose_export_intake/POSE_EXPORT_BRIEF.md",
                    "pose_export_intake": "pose_export_intake/README.md",
                }
            }
        )
    )
    (root / "fee_machine_v2_scaffold").mkdir()
    (root / "fee_machine_v2_scaffold" / "scene_choreography.json").write_text(
        json.dumps(
            {
                "beats": [
                    {
                        "id": "machine_reveal_handoff",
                        "start": 5.0,
                        "end": 8.65,
                        "acting_objective": "look from bill to the revealed machine",
                        "primary_motion": "wall slide reveals active gears and lever",
                        "review_frame": "machine_reveal",
                    },
                    {
                        "id": "skeptical_point_drive",
                        "start": 8.5,
                        "end": 12.1,
                        "acting_objective": "point with skepticism at fee source",
                        "primary_motion": "pointing pose locks machine cause to fee stack effect",
                        "review_frame": "acting_read",
                    },
                ]
            }
        )
    )
    (root / "review_package_index.json").write_text(
        json.dumps(
            {
                "blockers": ["Blender or Moho is required"],
                "placeholder_counts": {"audio_assets": 0},
                "objective_evidence": {
                    "choreography_beats_with_motion": 1,
                    "choreography_beats_without_motion": 1,
                    "choreography_beats_with_valid_review_frames": 2,
                    "choreography_beats_without_valid_review_frames": 0,
                },
            }
        )
    )
    (root / "first_frame_candidates").mkdir()
    (root / "first_frame_candidates" / "first_frame_manifest.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {"id": "cold_open_bill", "filename": "cold_open_bill.png", "sample_time": 0.35},
                    {"id": "human_read", "filename": "human_read.png", "sample_time": 1.25},
                ]
            }
        )
    )


def test_review_hub_renders_primary_links_and_first_frames(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)

    html = module.render_review_hub(tmp_path)

    assert "<title>Signal Room Review Hub</title>" in html
    assert "fee_machine_v2_draft_with_audio.mp4" in html
    assert "proof_retention_contact_sheet.svg" in html
    assert "MOTION_QUALITY_PASS.md" in html
    assert "motion_quality_plan.json" in html
    assert "fee_machine_v2_scaffold/scene_choreography.json" in html
    assert "machine_reveal_handoff" in html
    assert "look from bill to the revealed machine" in html
    assert "skeptical_point_drive" in html
    assert "Objective Choreography Evidence" in html
    assert "Choreography beats with motion" in html
    assert "<dd>1</dd>" in html
    assert "Choreography beats without motion" in html
    assert "Choreography beats with valid review frames" in html
    assert "Choreography beats without valid review frames" in html
    assert "EDITORIAL_SCREENING_GUIDE.md" in html
    assert "EDITORIAL_SCORECARD.json" in html
    assert "POSE_EXPORT_BRIEF.md" in html
    assert "first_frame_candidates/cold_open_bill.png" in html
    assert "Blender or Moho is required" in html
    assert "review-only" in html


def test_review_hub_writes_html(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)
    out = tmp_path / "review_hub.html"

    result = module.write_review_hub(tmp_path, out)

    assert result["passed"] is True
    assert result["output"] == str(out)
    assert "Signal Room Review Hub" in out.read_text()


def test_review_hub_renders_before_handoff_manifest_exists(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)
    (tmp_path / "handoff_manifest.json").unlink()

    html = module.render_review_hub(tmp_path)

    assert "fee_machine_v2_draft_with_audio.mp4" in html
    assert "proof_retention_contact_sheet.svg" in html
    assert "EDITORIAL_SCORECARD.json" in html
