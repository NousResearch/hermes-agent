from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_post_pose_pipeline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_post_pose_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_post_pose_pipeline_builds_expected_command_plan(tmp_path: Path) -> None:
    module = load_module()
    package_dir = tmp_path / "review"
    candidate_dir = package_dir / "pose_export_intake" / "character_frames" / "Suit_Male"

    plan = module.build_pipeline_plan(package_dir, candidate_dir)

    names = [step["name"] for step in plan]
    assert names == [
        "rig_acting_gate",
        "acting_contact_sheet",
        "install_pose_export",
        "hyperframes_lint",
        "hyperframes_inspect",
        "hyperframes_render",
        "render_asset_gate",
        "motion_smoothness_gate",
        "sample_proof_frames",
        "retention_frame_gate",
        "proof_contact_sheet",
        "audio_mux",
        "motion_quality_plan",
        "review_package_index",
        "handoff_bundle",
    ]
    assert plan[0]["command"][:2] == ["python", "scripts/signal_room_rig_acting_gate.py"]
    assert str(candidate_dir) in plan[0]["command"]
    assert plan[5]["command"][:3] == ["npx", "--yes", "hyperframes@0.6.38"]
    assert plan[5]["cwd"] == str(package_dir / "fee_machine_v2_scaffold")
    assert plan[7]["command"][:2] == ["python", "scripts/signal_room_motion_smoothness_gate.py"]
    assert "--frame-diff-report" in plan[7]["command"]
    assert str(package_dir / "motion_frame_diff_report.json") in plan[7]["command"]
    assert "--choreography" in plan[7]["command"]
    assert str(package_dir / "fee_machine_v2_scaffold" / "scene_choreography.json") in plan[7]["command"]
    assert "--motion-primitives" in plan[7]["command"]
    assert str(package_dir / "fee_machine_v2_scaffold" / "motion_primitives.json") in plan[7]["command"]
    assert plan[9]["command"][:2] == ["python", "scripts/signal_room_retention_frame_gate.py"]
    assert "--choreography" in plan[9]["command"]
    assert str(package_dir / "fee_machine_v2_scaffold" / "scene_choreography.json") in plan[9]["command"]
    assert plan[12]["command"][:2] == ["python", "scripts/signal_room_motion_quality_plan.py"]
    assert plan[-1]["command"][:2] == ["python", "scripts/signal_room_review_handoff_bundle.py"]


def test_post_pose_pipeline_dry_run_does_not_execute(tmp_path: Path) -> None:
    module = load_module()
    package_dir = tmp_path / "review"
    candidate_dir = package_dir / "candidate"
    called = []

    result = module.run_pipeline(package_dir, candidate_dir, dry_run=True, runner=lambda step: called.append(step))

    assert result["passed"] is True
    assert result["dry_run"] is True
    assert result["step_count"] == 15
    assert called == []


def test_post_pose_pipeline_stops_on_failed_step(tmp_path: Path) -> None:
    module = load_module()
    package_dir = tmp_path / "review"
    candidate_dir = package_dir / "candidate"

    def failing_runner(step):
        return 7

    result = module.run_pipeline(package_dir, candidate_dir, runner=failing_runner)

    assert result["passed"] is False
    assert result["failed_step"] == "rig_acting_gate"
    assert result["completed_steps"] == []
