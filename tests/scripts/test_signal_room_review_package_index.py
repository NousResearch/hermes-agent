from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_review_package_index.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_review_package_index", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def test_review_package_index_summarizes_scorecards_and_placeholders(tmp_path: Path) -> None:
    module = load_module()
    write_json(tmp_path / "video_env_scorecard.json", {"passed": False, "blockers": ["Blender missing"]})
    write_json(tmp_path / "scaffold_scorecard.json", {"passed": True, "errors": []})
    write_json(tmp_path / "render_asset_scorecard.json", {"passed": True, "render_path": "draft.mp4"})
    write_json(
        tmp_path / "motion_smoothness_scorecard.json",
        {
            "passed": True,
            "beat_motion": [
                {"beat_id": "machine_reveal_handoff", "passed": True},
                {"beat_id": "skeptical_point_drive", "passed": True},
            ],
            "primitive_motion": [
                {"primitive_id": "machine_causality", "passed": True},
            ],
        },
    )
    write_json(tmp_path / "audio_mux_scorecard.json", {"passed": True, "output": "draft_with_audio.mp4"})
    write_json(
        tmp_path / "audio_asset_scorecard.json",
        {"passed": True, "manifest": {"assets": [{"placeholder": True}, {"placeholder": True}]}},
    )
    write_json(
        tmp_path / "retention_frame_scorecard.json",
        {
            "passed": True,
            "manifest": {"frames": [{"placeholder": True}]},
            "choreography_frame_coverage": [
                {"beat_id": "machine_reveal_handoff", "frame_valid": True},
                {"beat_id": "skeptical_point_drive", "frame_valid": True},
            ],
        },
    )

    result = module.build_review_package_index(tmp_path)

    assert result["passed"] is False
    assert result["scorecards"]["scaffold"]["passed"] is True
    assert result["scorecards"]["render_asset"]["passed"] is True
    assert result["scorecards"]["motion_smoothness"]["passed"] is True
    assert result["scorecards"]["audio_mux"]["passed"] is True
    assert result["scorecards"]["audio_assets"]["passed"] is True
    assert result["placeholder_counts"] == {"audio_assets": 2, "proof_frames": 1}
    assert result["objective_evidence"] == {
        "choreography_beats_with_motion": 2,
        "choreography_beats_without_motion": 0,
        "choreography_beats_with_valid_review_frames": 2,
        "choreography_beats_without_valid_review_frames": 0,
        "motion_primitives_with_evidence": 1,
        "motion_primitives_without_evidence": 0,
    }
    assert "Blender missing" in result["blockers"]
    assert "placeholder audio assets must be replaced before editorial review" in result["blockers"]
    assert "placeholder proof frames must be replaced before visual review" in result["blockers"]


def test_review_package_index_reports_missing_required_scorecards(tmp_path: Path) -> None:
    module = load_module()
    write_json(tmp_path / "scaffold_scorecard.json", {"passed": True, "errors": []})

    result = module.build_review_package_index(tmp_path)

    assert result["passed"] is False
    assert "missing scorecard: video_env_scorecard.json" in result["blockers"]
    assert "missing scorecard: audio_asset_scorecard.json" in result["blockers"]
    assert "missing scorecard: render_asset_scorecard.json" in result["blockers"]
    assert "missing scorecard: motion_smoothness_scorecard.json" in result["blockers"]
    assert "missing scorecard: audio_mux_scorecard.json" in result["blockers"]
    assert "missing scorecard: retention_frame_scorecard.json" in result["blockers"]


def test_review_package_index_blocks_incomplete_choreography_evidence(tmp_path: Path) -> None:
    module = load_module()
    write_json(tmp_path / "video_env_scorecard.json", {"passed": True})
    write_json(tmp_path / "scaffold_scorecard.json", {"passed": True})
    write_json(tmp_path / "render_asset_scorecard.json", {"passed": True})
    write_json(
        tmp_path / "motion_smoothness_scorecard.json",
        {
            "passed": True,
            "beat_motion": [
                {"beat_id": "machine_reveal_handoff", "passed": True},
                {"beat_id": "skeptical_point_drive", "passed": False},
            ],
            "primitive_motion": [
                {"primitive_id": "machine_causality", "passed": False},
            ],
        },
    )
    write_json(tmp_path / "audio_mux_scorecard.json", {"passed": True})
    write_json(tmp_path / "audio_asset_scorecard.json", {"passed": True, "manifest": {"assets": []}})
    write_json(
        tmp_path / "retention_frame_scorecard.json",
        {
            "passed": True,
            "manifest": {"frames": []},
            "choreography_frame_coverage": [
                {"beat_id": "machine_reveal_handoff", "frame_valid": True},
                {"beat_id": "skeptical_point_drive", "frame_valid": False},
            ],
        },
    )

    result = module.build_review_package_index(tmp_path)

    assert result["passed"] is False
    assert "choreography motion evidence missing for 1 beat(s)" in result["blockers"]
    assert "choreography review-frame evidence missing for 1 beat(s)" in result["blockers"]
    assert "motion primitive evidence missing for 1 primitive(s)" in result["blockers"]


def test_review_package_report_renders_handoff_markdown(tmp_path: Path) -> None:
    module = load_module()
    result = {
        "passed": False,
        "package_dir": str(tmp_path),
        "blockers": [
            "Blender or Moho is required for local character pose rendering",
            "placeholder audio assets must be replaced before editorial review",
        ],
        "placeholder_counts": {"audio_assets": 11, "proof_frames": 5},
        "scorecards": {
            "video_env": {"passed": False},
            "scaffold": {"passed": True},
            "render_asset": {"passed": True},
            "motion_smoothness": {
                "passed": True,
                "beat_motion": [
                    {"beat_id": "machine_reveal_handoff", "passed": True},
                    {"beat_id": "skeptical_point_drive", "passed": False},
                ],
                "primitive_motion": [
                    {"primitive_id": "machine_causality", "passed": False},
                ],
            },
            "audio_mux": {"passed": True},
            "audio_assets": {"passed": True},
            "proof_frames": {
                "passed": True,
                "choreography_frame_coverage": [
                    {"beat_id": "machine_reveal_handoff", "frame_valid": True},
                    {"beat_id": "skeptical_point_drive", "frame_valid": False},
                ],
            },
        },
        "objective_evidence": {
            "choreography_beats_with_motion": 1,
            "choreography_beats_without_motion": 1,
            "choreography_beats_with_valid_review_frames": 1,
            "choreography_beats_without_valid_review_frames": 1,
            "motion_primitives_with_evidence": 0,
            "motion_primitives_without_evidence": 1,
        },
    }

    markdown = module.render_markdown_report(result)

    assert "# Signal Room Review Package Status" in markdown
    assert "**Status:** blocked" in markdown
    assert "- video_env: blocked" in markdown
    assert "- scaffold: passed" in markdown
    assert "- render_asset: passed" in markdown
    assert "- motion_smoothness: passed" in markdown
    assert "- audio_mux: passed" in markdown
    assert "- choreography beats with motion: 1" in markdown
    assert "- choreography beats without motion: 1" in markdown
    assert "- choreography beats with valid review frames: 1" in markdown
    assert "- choreography beats without valid review frames: 1" in markdown
    assert "- motion primitives with evidence: 0" in markdown
    assert "- motion primitives without evidence: 1" in markdown
    assert "- audio_assets: 11" in markdown
    assert "- proof_frames: 5" in markdown
    assert "Blender or Moho is required" in markdown
    assert "Replace placeholder proof frames" in markdown


def test_review_package_report_omits_resolved_placeholder_actions(tmp_path: Path) -> None:
    module = load_module()
    result = {
        "passed": False,
        "package_dir": str(tmp_path),
        "blockers": ["Blender or Moho is required for local character pose rendering"],
        "placeholder_counts": {"audio_assets": 0, "proof_frames": 0},
        "scorecards": {
            "video_env": {"passed": False},
            "scaffold": {"passed": True},
            "render_asset": {"passed": True},
            "motion_smoothness": {"passed": True},
            "audio_mux": {"passed": True},
            "audio_assets": {"passed": True},
            "proof_frames": {"passed": True},
        },
    }

    markdown = module.render_markdown_report(result)

    assert "Replace placeholder audio assets" not in markdown
    assert "Replace placeholder proof frames" not in markdown
    assert "Resolve local character pose rendering" in markdown
