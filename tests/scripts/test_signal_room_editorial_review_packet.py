from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_editorial_review_packet.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_editorial_review_packet", SCRIPT)
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
                "failure_conditions": ["five similar cards"],
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
                        "review_question": "Is cause and effect visible?",
                        "must_show": ["lever", "money flow"],
                    },
                ],
            }
        )
    )
    (scaffold / "audio_cue_sheet.json").write_text(
        json.dumps(
            {
                "duration_seconds": 15,
                "cues": [
                    {"id": "paper_bill_snap", "start": 0.35, "duration": 0.25, "sound": "paper snap"},
                    {"id": "machine_drive_loop", "start": 6.05, "duration": 3.2, "sound": "gear rhythm"},
                ],
            }
        )
    )
    (root / "review_package_index.json").write_text(
        json.dumps(
            {
                "passed": False,
                "blockers": ["Blender or Moho is required"],
                "placeholder_counts": {"audio_assets": 0, "proof_frames": 0},
            }
        )
    )


def test_editorial_review_packet_renders_questions_and_assets(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)

    markdown = module.render_editorial_review_packet(tmp_path)

    assert "# Signal Room Editorial Review Packet" in markdown
    assert "fee_machine_v2_draft_with_audio.mp4" in markdown
    assert "proof_retention_contact_sheet.svg" in markdown
    assert "Blender or Moho is required" in markdown
    assert "five similar cards" in markdown
    assert "ordinary_bill" in markdown
    assert "Who is affected?" in markdown
    assert "foreground character, ordinary bill" in markdown
    assert "paper_bill_snap" in markdown
    assert "0.35s" in markdown
    assert "[ ] Pass" in markdown


def test_editorial_review_packet_writes_output_file(tmp_path: Path) -> None:
    module = load_module()
    write_package(tmp_path)
    out = tmp_path / "EDITORIAL_REVIEW.md"

    result = module.write_editorial_review_packet(tmp_path, out)

    assert result["passed"] is True
    assert result["output"] == str(out)
    assert "Signal Room Editorial Review Packet" in out.read_text()
