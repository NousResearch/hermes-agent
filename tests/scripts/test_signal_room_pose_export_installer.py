from __future__ import annotations

import importlib.util
import json
import struct
import zlib
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_pose_export_installer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_pose_export_installer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def png_bytes(width: int = 1080, height: int = 1500) -> bytes:
    raw = b"".join(b"\x00" + b"\x11\x22\x33\xff" * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def write_candidate(root: Path) -> None:
    for frame in [
        "neutral_read.png",
        "bill_shock.png",
        "skeptical_point.png",
        "look_to_machine.png",
        "lean_weight_shift.png",
        "mouth_closed.png",
        "mouth_a.png",
        "mouth_o.png",
    ]:
        (root / frame).write_bytes(png_bytes())
    (root / "rig_pass_manifest.json").write_text(
        json.dumps(
            {
                "status": "review-only",
                "public_release": False,
                "license_status": "licensed for review",
                "render_tool": "Blender",
                "candidate_name": "Suit_Male",
            }
        )
    )


def write_scaffold(root: Path) -> None:
    poses = root / "assets" / "poses"
    poses.mkdir(parents=True)
    for pose in ["neutral_read", "bill_shock", "look_to_machine", "skeptical_point", "slight_lean"]:
        (poses / f"{pose}.svg").write_text(f"<svg><title>{pose}</title></svg>")
    (root / "index.html").write_text(
        """
<img src="assets/poses/neutral_read.svg">
<img src="assets/poses/bill_shock.svg">
<img src="assets/poses/look_to_machine.svg">
<img src="assets/poses/skeptical_point.svg">
<img src="assets/poses/slight_lean.svg">
"""
    )
    (root / "manifest.json").write_text(json.dumps({"status": "review-only", "public_release": False}))


def test_pose_export_installer_copies_pngs_and_rewrites_scaffold(tmp_path: Path) -> None:
    module = load_module()
    candidate = tmp_path / "candidate"
    scaffold = tmp_path / "scaffold"
    candidate.mkdir()
    scaffold.mkdir()
    write_candidate(candidate)
    write_scaffold(scaffold)

    result = module.install_pose_export(candidate, scaffold)

    assert result["passed"] is True
    assert result["installed_pose_count"] == 5
    assert (scaffold / "assets" / "poses" / "neutral_read.png").exists()
    assert (scaffold / "assets" / "poses" / "slight_lean.png").exists()
    html = (scaffold / "index.html").read_text()
    assert "assets/poses/neutral_read.png" in html
    assert "assets/poses/slight_lean.png" in html
    assert "assets/poses/slight_lean.svg" not in html
    source = json.loads((scaffold / "source_rig_manifest.json").read_text())
    assert source["candidate_name"] == "Suit_Male"
    assert source["installed_from"] == str(candidate)
    manifest = json.loads((scaffold / "manifest.json").read_text())
    assert manifest["approved_pose_export"]["candidate_name"] == "Suit_Male"


def test_pose_export_installer_fails_missing_required_candidate_frame(tmp_path: Path) -> None:
    module = load_module()
    candidate = tmp_path / "candidate"
    scaffold = tmp_path / "scaffold"
    candidate.mkdir()
    scaffold.mkdir()
    write_candidate(candidate)
    write_scaffold(scaffold)
    (candidate / "look_to_machine.png").unlink()

    result = module.install_pose_export(candidate, scaffold)

    assert result["passed"] is False
    assert "missing required candidate frame: look_to_machine.png" in result["errors"]
