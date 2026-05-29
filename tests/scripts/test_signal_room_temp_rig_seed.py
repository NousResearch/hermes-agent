from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_temp_rig_seed.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_temp_rig_seed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_temp_rig_seed_writes_required_review_only_svg_poses(tmp_path: Path) -> None:
    module = load_module()
    out = tmp_path / "signal_room_adult_investigator_v0"

    manifest = module.create_temp_rig(out)

    assert manifest["status"] == "review-only"
    assert manifest["public_release"] is False
    assert manifest["pose_count"] == 5
    assert (out / "rig_manifest.json").exists()
    assert (out / "poses" / "neutral_read.svg").exists()
    assert (out / "poses" / "bill_shock.svg").exists()
    assert (out / "poses" / "look_to_machine.svg").exists()
    assert (out / "poses" / "skeptical_point.svg").exists()
    assert (out / "poses" / "slight_lean.svg").exists()
    rig_manifest = json.loads((out / "rig_manifest.json").read_text())
    assert rig_manifest["replacement_required"] == "Blender/Moho adult character candidate"
    svg = (out / "poses" / "skeptical_point.svg").read_text()
    assert "<title>skeptical_point</title>" in svg
    assert "review-only temporary vector rig" in svg
    assert "point-arm" in svg


def test_temp_rig_seed_refuses_existing_output_without_force(tmp_path: Path) -> None:
    module = load_module()
    out = tmp_path / "rig"
    out.mkdir()
    (out / "keep.txt").write_text("do not overwrite")

    try:
        module.create_temp_rig(out)
    except FileExistsError as exc:
        assert str(out) in str(exc)
    else:
        raise AssertionError("expected existing rig directory to fail without force")
