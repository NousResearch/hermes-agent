from __future__ import annotations

import importlib.util
import json
import struct
import zlib
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_contact_sheet.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_contact_sheet", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_png(path: Path, *, width: int = 1080, height: int = 1500) -> None:
    raw = b"".join(b"\x00" + (b"\xff" * width * 4) for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def write_candidate(candidate_dir: Path) -> None:
    candidate_dir.mkdir()
    module = load_module()
    for frame in module.REQUIRED_CONTACT_FRAMES:
        write_png(candidate_dir / frame.filename)
    (candidate_dir / "rig_pass_manifest.json").write_text(
        json.dumps(
            {
                "candidate": candidate_dir.name,
                "license_status": "CC0 local license copied in package/licenses/",
                "render_tool": "Blender",
                "public_release": False,
            }
        )
    )


def test_contact_sheet_writes_1080x1920_review_svg_with_all_frames(tmp_path: Path) -> None:
    module = load_module()
    candidate = tmp_path / "Suit_Male"
    write_candidate(candidate)
    out = tmp_path / "contact_sheet.svg"

    result = module.write_contact_sheet(candidate, out)

    assert result["passed"] is True
    assert result["output"] == str(out)
    svg = out.read_text()
    assert '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1920" viewBox="0 0 1080 1920">' in svg
    assert "Signal Room rig acting review" in svg
    assert "review-only / not public" in svg
    assert "neutral_read.png" in svg
    assert "bill shock" in svg
    assert "mouth o" in svg
    assert "license: CC0 local license copied in package/licenses/" in svg


def test_contact_sheet_rejects_failing_candidate(tmp_path: Path) -> None:
    module = load_module()
    candidate = tmp_path / "Suit_Male"
    candidate.mkdir()
    write_png(candidate / "neutral_read.png", height=900)
    (candidate / "rig_pass_manifest.json").write_text(json.dumps({"public_release": True}))

    result = module.write_contact_sheet(candidate, tmp_path / "contact_sheet.svg")

    assert result["passed"] is False
    assert "missing required frame: bill_shock.png" in result["errors"]
    assert "neutral_read.png height 900 below minimum 1400" in result["errors"]
    assert not (tmp_path / "contact_sheet.svg").exists()
