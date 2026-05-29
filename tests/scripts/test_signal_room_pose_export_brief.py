from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_pose_export_brief.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_pose_export_brief", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_intake(root: Path) -> None:
    root.mkdir()
    (root / "required_frames.json").write_text(
        json.dumps(
            {
                "frames": [
                    "neutral_read.png",
                    "bill_shock.png",
                    "skeptical_point.png",
                    "look_to_machine.png",
                    "mouth_closed.png",
                    "mouth_a.png",
                    "mouth_o.png",
                ],
                "format": "transparent RGBA PNG",
                "minimum_height": 1400,
                "review": ["phone-readable face and hands", "no transparent edge halos"],
            }
        )
    )
    (root / "rig_pass_manifest.template.json").write_text(
        json.dumps(
            {
                "title": "Signal Room Adult Investigator Pose Export",
                "candidate_name": "Suit_Male",
                "render_tool": "Blender or Moho",
                "public_release": False,
            }
        )
    )


def test_pose_export_brief_renders_animator_requirements(tmp_path: Path) -> None:
    module = load_module()
    intake = tmp_path / "pose_export_intake"
    write_intake(intake)

    markdown = module.render_pose_export_brief(intake)

    assert "# Signal Room Pose Export Brief" in markdown
    assert "Suit_Male" in markdown
    assert "transparent RGBA PNG" in markdown
    assert "1400px" in markdown
    assert "skeptical_point.png" in markdown
    assert "aims toward the fee machine" in markdown
    assert "phone-readable face and hands" in markdown
    assert "rig_pass_manifest.json" in markdown
    assert "not approved for public release" in markdown


def test_pose_export_brief_writes_output(tmp_path: Path) -> None:
    module = load_module()
    intake = tmp_path / "pose_export_intake"
    write_intake(intake)
    out = tmp_path / "POSE_EXPORT_BRIEF.md"

    result = module.write_pose_export_brief(intake, out)

    assert result["passed"] is True
    assert result["output"] == str(out)
    assert "Signal Room Pose Export Brief" in out.read_text()
