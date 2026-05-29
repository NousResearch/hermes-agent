from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_pose_export_intake_pack.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_pose_export_intake_pack", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_pose_export_intake_pack_writes_templates_and_commands(tmp_path: Path) -> None:
    module = load_module()
    out_dir = tmp_path / "pose_export_intake"

    manifest = module.create_pose_export_intake_pack(out_dir, candidate_name="Suit_Male")

    assert manifest["candidate_name"] == "Suit_Male"
    assert manifest["required_frame_count"] == 8
    assert manifest["public_release"] is False
    assert (out_dir / "README.md").exists()
    assert (out_dir / "rig_pass_manifest.template.json").exists()
    assert (out_dir / "required_frames.json").exists()
    assert (out_dir / "validate_pose_export.sh").exists()
    assert (out_dir / "character_frames" / "Suit_Male" / ".gitkeep").exists()

    required_frames = json.loads((out_dir / "required_frames.json").read_text())
    assert "neutral_read.png" in required_frames["frames"]
    assert "mouth_o.png" in required_frames["frames"]

    template = json.loads((out_dir / "rig_pass_manifest.template.json").read_text())
    assert template["render_tool"] == "Blender or Moho"
    assert template["license_status"] == "REQUIRED: source/license notes"
    assert template["public_release"] is False

    readme = (out_dir / "README.md").read_text()
    assert "transparent RGBA PNG" in readme
    assert "python scripts/signal_room_rig_acting_gate.py" in readme
    assert str(out_dir / "character_frames" / "Suit_Male") in readme
    assert "Suit_Male_contact_sheet.svg" in readme

    validate_script = (out_dir / "validate_pose_export.sh").read_text()
    assert str(out_dir / "character_frames" / "Suit_Male") in validate_script
    assert str(out_dir / "rig_acting_scorecard.json") in validate_script


def test_pose_export_intake_pack_refuses_existing_output_without_force(tmp_path: Path) -> None:
    module = load_module()
    out_dir = tmp_path / "pose_export_intake"
    out_dir.mkdir()

    try:
        module.create_pose_export_intake_pack(out_dir)
    except FileExistsError as exc:
        assert exc.args[0] == out_dir
    else:
        raise AssertionError("expected FileExistsError")
