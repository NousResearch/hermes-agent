from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_motion_quality_plan.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_motion_quality_plan", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_motion_quality_plan_defines_upgrade_requirements(tmp_path: Path) -> None:
    module = load_module()
    package_dir = tmp_path / "review"
    scaffold = package_dir / "fee_machine_v2_scaffold"
    scaffold.mkdir(parents=True)
    (scaffold / "retention_frame_plan.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"id": "ordinary_bill", "sample_time": 1.25},
                    {"id": "machine_reveal", "sample_time": 6.5},
                    {"id": "memory_anchor", "sample_time": 13.2},
                ]
            }
        )
    )

    plan = module.build_motion_quality_plan(package_dir)

    assert plan["status"] == "review-only"
    assert plan["target"] == "Signal Room Motion Quality Pass v1"
    assert plan["baseline_assessment"]["current_quality"] == "prototype-grade"
    assert plan["success_criteria"]["smoothness"]["minimum_fps"] == 30
    assert plan["success_criteria"]["smoothness"]["no_static_hold_longer_than_seconds"] == 1.25
    assert plan["success_criteria"]["objective_choreography_evidence"] == {
        "all_choreography_beats_require_sampled_motion": True,
        "all_choreography_beats_require_valid_review_frame": True,
        "package_index_blocks_missing_motion_or_review_frame_evidence": True,
    }
    assert "fee_tag_reveal" in [primitive["id"] for primitive in plan["motion_primitives"]]
    assert "character_reaction_swap" in [primitive["id"] for primitive in plan["motion_primitives"]]
    assert "motion_smoothness_gate" in [gate["id"] for gate in plan["verification_gates"]]
    assert "editorial_readiness_gate" in [gate["id"] for gate in plan["verification_gates"]]
    assert [frame["id"] for frame in plan["review_frames"]] == [
        "ordinary_bill",
        "machine_reveal",
        "memory_anchor",
    ]


def test_motion_quality_plan_writes_json_and_markdown(tmp_path: Path) -> None:
    module = load_module()
    package_dir = tmp_path / "review"
    (package_dir / "fee_machine_v2_scaffold").mkdir(parents=True)
    (package_dir / "fee_machine_v2_scaffold" / "retention_frame_plan.json").write_text("{}")
    json_out = package_dir / "motion_quality_plan.json"
    md_out = package_dir / "MOTION_QUALITY_PASS.md"

    result = module.write_motion_quality_plan(package_dir, json_out=json_out, markdown_out=md_out)

    assert result["passed"] is True
    assert json_out.exists()
    assert md_out.exists()
    assert "Signal Room Motion Quality Pass v1" in md_out.read_text()
    assert "motion_smoothness_gate" in md_out.read_text()
    assert "Objective Choreography Evidence" in md_out.read_text()
    assert "package_index_blocks_missing_motion_or_review_frame_evidence" in md_out.read_text()
