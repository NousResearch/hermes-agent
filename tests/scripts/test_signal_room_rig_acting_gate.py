from __future__ import annotations

import importlib.util
import json
import struct
import zlib
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_rig_acting_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_rig_acting_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_png(path: Path, *, width: int = 1080, height: int = 1500, alpha: bool = True) -> None:
    color_type = 6 if alpha else 2
    channels = 4 if alpha else 3
    raw = b"".join(b"\x00" + (b"\xff" * width * channels) for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def write_manifest(candidate_dir: Path, *, public_release: bool = False) -> None:
    (candidate_dir / "rig_pass_manifest.json").write_text(
        json.dumps(
            {
                "candidate": candidate_dir.name,
                "license_status": "CC0 local license copied in package/licenses/",
                "render_tool": "Blender",
                "public_release": public_release,
            }
        )
    )


def test_candidate_passes_when_required_frames_are_alpha_and_large(tmp_path: Path) -> None:
    module = load_module()
    candidate = tmp_path / "suit_male"
    candidate.mkdir()
    for name in module.REQUIRED_FRAMES:
        write_png(candidate / name)
    write_manifest(candidate)

    result = module.evaluate_candidate_dir(candidate)

    assert result["passed"] is True
    assert result["candidate"] == "suit_male"
    assert result["errors"] == []


def test_candidate_fails_for_missing_small_or_non_alpha_frames(tmp_path: Path) -> None:
    module = load_module()
    candidate = tmp_path / "suit_male"
    candidate.mkdir()
    for name in module.REQUIRED_FRAMES:
        if name == "mouth_o.png":
            continue
        write_png(candidate / name, height=900 if name == "neutral_read.png" else 1500, alpha=name != "bill_shock.png")
    write_manifest(candidate, public_release=True)

    result = module.evaluate_candidate_dir(candidate)

    assert result["passed"] is False
    assert "missing required frame: mouth_o.png" in result["errors"]
    assert "neutral_read.png height 900 below minimum 1400" in result["errors"]
    assert "bill_shock.png is not RGBA/alpha PNG" in result["errors"]
    assert "manifest public_release must be false for review gate" in result["errors"]


def test_find_candidate_dirs_handles_missing_parent(tmp_path: Path) -> None:
    module = load_module()

    assert module.find_candidate_dirs(tmp_path / "missing") == []
