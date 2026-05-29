from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_editorial_scorecard.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_editorial_scorecard", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_package(root: Path) -> None:
    scaffold = root / "fee_machine_v2_scaffold"
    scaffold.mkdir(parents=True)
    (scaffold / "retention_frame_plan.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"id": "ordinary_bill", "review_question": "Who is affected?"},
                    {"id": "machine_reveal", "review_question": "Is the machine visible?"},
                ],
                "failure_conditions": ["five similar cards"],
            }
        )
    )
    (scaffold / "scene_choreography.json").write_text(
        json.dumps(
            {
                "beats": [
                    {
                        "id": "machine_reveal_handoff",
                        "acting_objective": "look from bill to the revealed machine",
                        "primary_motion": "wall slide reveals active gears and lever",
                        "review_frame": "machine_reveal",
                    },
                    {
                        "id": "skeptical_point_drive",
                        "acting_objective": "point with skepticism at fee source",
                        "primary_motion": "pointing pose locks machine cause to fee stack effect",
                        "review_frame": "acting_read",
                    },
                ]
            }
        )
    )
    (root / "first_frame_candidates").mkdir()
    (root / "first_frame_candidates" / "first_frame_manifest.json").write_text(
        json.dumps({"candidates": [{"id": "cold_open_bill"}, {"id": "human_read"}]})
    )


def test_editorial_scorecard_template_uses_package_plans(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)

    scorecard = module.build_scorecard_template(tmp_path)

    assert scorecard["status"] == "review-required"
    assert [item["id"] for item in scorecard["first_frame_candidates"]] == ["cold_open_bill", "human_read"]
    assert [item["id"] for item in scorecard["retention_frames"]] == ["ordinary_bill", "machine_reveal"]
    assert [item["id"] for item in scorecard["choreography_beats"]] == [
        "machine_reveal_handoff",
        "skeptical_point_drive",
    ]
    assert scorecard["choreography_beats"][0]["acting_objective"] == "look from bill to the revealed machine"
    assert scorecard["choreography_beats"][1]["primary_motion"] == "pointing pose locks machine cause to fee stack effect"
    assert scorecard["failure_conditions"][0]["condition"] == "five similar cards"
    assert scorecard["overall_decision"] == "pending"


def test_editorial_scorecard_gate_passes_completed_review(tmp_path: Path) -> None:
    module = load_module()
    scorecard = {
        "overall_decision": "pass",
        "critical_criteria": [{"id": "first_frame", "decision": "pass"}],
        "first_frame_candidates": [{"id": "cold_open_bill", "decision": "keep"}],
        "retention_frames": [{"id": "ordinary_bill", "decision": "pass"}],
        "choreography_beats": [{"id": "machine_reveal_handoff", "decision": "pass"}],
        "failure_conditions": [{"condition": "five similar cards", "present": False}],
    }

    result = module.evaluate_scorecard(scorecard)

    assert result["passed"] is True
    assert result["errors"] == []


def test_editorial_scorecard_gate_fails_revise_reject_or_present_failure() -> None:
    module = load_module()
    scorecard = {
        "overall_decision": "pass",
        "critical_criteria": [{"id": "first_frame", "decision": "revise"}],
        "first_frame_candidates": [{"id": "cold_open_bill", "decision": "reject"}],
        "retention_frames": [{"id": "ordinary_bill", "decision": "replace"}],
        "choreography_beats": [{"id": "machine_reveal_handoff", "decision": "revise"}],
        "failure_conditions": [{"condition": "five similar cards", "present": True}],
    }

    result = module.evaluate_scorecard(scorecard)

    assert result["passed"] is False
    assert "critical criterion first_frame is revise" in result["errors"]
    assert "first-frame candidate cold_open_bill is reject" in result["errors"]
    assert "retention frame ordinary_bill is replace" in result["errors"]
    assert "choreography beat machine_reveal_handoff is revise" in result["errors"]
    assert "failure condition present: five similar cards" in result["errors"]


def test_editorial_scorecard_writes_template_and_gate_result(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)
    template_path = tmp_path / "EDITORIAL_SCORECARD.json"
    gate_path = tmp_path / "editorial_scorecard_gate.json"

    template = module.write_scorecard_template(tmp_path, template_path)
    gate = module.write_gate_result(template_path, gate_path)

    assert template["output"] == str(template_path)
    assert gate["passed"] is False
    assert "overall_decision must be pass" in gate["errors"]
    assert gate_path.exists()
