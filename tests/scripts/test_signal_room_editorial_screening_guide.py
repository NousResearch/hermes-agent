from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_editorial_screening_guide.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_editorial_screening_guide", SCRIPT)
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
                "failure_conditions": ["machine reveal does not show cause and effect"],
                "frames": [
                    {
                        "id": "ordinary_bill",
                        "sample_time": 1.25,
                        "beat": "human problem",
                        "review_question": "Who is affected?",
                        "must_show": ["foreground character", "ordinary bill"],
                    },
                    {
                        "id": "machine_reveal",
                        "sample_time": 6.5,
                        "beat": "mechanism reveal",
                        "review_question": "Is the fee machine revealed?",
                        "must_show": ["open wall", "money flow direction"],
                    },
                ],
            }
        )
    )
    (scaffold / "scene_choreography.json").write_text(
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
                    }
                ]
            }
        )
    )
    first_frames = root / "first_frame_candidates"
    first_frames.mkdir()
    (first_frames / "first_frame_manifest.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "id": "cold_open_bill",
                        "sample_time": 0.35,
                        "question": "who is affected?",
                        "filename": "cold_open_bill.png",
                    }
                ]
            }
        )
    )
    (root / "EDITORIAL_SCORECARD.json").write_text(
        json.dumps(
            {
                "critical_criteria": [
                    {"id": "first_frame", "label": "First frame makes the human problem clear."},
                    {"id": "machine_reveal", "label": "Machine reveal explains cause and effect."},
                ],
                "choreography_beats": [
                    {
                        "id": "machine_reveal_handoff",
                        "acting_objective": "look from bill to the revealed machine",
                        "primary_motion": "wall slide reveals active gears and lever",
                        "review_frame": "machine_reveal",
                    }
                ],
            }
        )
    )


def test_editorial_screening_guide_maps_watch_points_to_scorecard(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)

    markdown = module.render_screening_guide(tmp_path)

    assert "# Signal Room Editorial Screening Guide" in markdown
    assert "fee_machine_v2_draft_with_audio.mp4" in markdown
    assert "0.35s" in markdown
    assert "cold_open_bill.png" in markdown
    assert "ordinary_bill" in markdown
    assert "machine_reveal" in markdown
    assert "critical_criteria.first_frame" in markdown
    assert "retention_frames.machine_reveal" in markdown
    assert "## Choreography Beat Checks" in markdown
    assert "machine_reveal_handoff" in markdown
    assert "look from bill to the revealed machine" in markdown
    assert "choreography_beats.machine_reveal_handoff" in markdown
    assert "machine reveal does not show cause and effect" in markdown


def test_editorial_screening_guide_writes_output(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)
    out = tmp_path / "EDITORIAL_SCREENING_GUIDE.md"

    result = module.write_screening_guide(tmp_path, out)

    assert result["passed"] is True
    assert result["output"] == str(out)
    assert "Signal Room Editorial Screening Guide" in out.read_text()
