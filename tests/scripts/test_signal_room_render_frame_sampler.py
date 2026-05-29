from __future__ import annotations

import importlib.util
import json
import struct
import zlib
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_render_frame_sampler.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_render_frame_sampler", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def png_bytes(width: int = 1080, height: int = 1920, shade: int = 80) -> bytes:
    raw = b"".join(b"\x00" + bytes([shade, shade, shade]) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def test_render_frame_sampler_exports_plan_frames_with_manifest(tmp_path: Path) -> None:
    module = load_module()
    plan = {
        "frames": [
            {"id": "ordinary_bill", "sample_time": 1.25, "beat": "human problem"},
            {"id": "machine_reveal", "sample_time": 6.5, "beat": "mechanism reveal"},
        ]
    }
    plan_path = tmp_path / "retention_frame_plan.json"
    plan_path.write_text(json.dumps(plan))
    render_path = tmp_path / "draft.mp4"
    render_path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + (b"\x00" * 1500))
    out_dir = tmp_path / "proof_frames"
    commands = []

    def fake_runner(command: list[str]) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(png_bytes(shade=50 + len(commands)))

    manifest = module.sample_render_frames(plan_path, render_path, out_dir, runner=fake_runner)

    assert [frame["id"] for frame in manifest["frames"]] == ["ordinary_bill", "machine_reveal"]
    assert manifest["render_tool"] == "ffmpeg"
    assert manifest["source_render"] == str(render_path)
    assert manifest["frames"][0]["placeholder"] is False
    assert manifest["frames"][0]["sample_time"] == 1.25
    assert (out_dir / "ordinary_bill.png").exists()
    assert (out_dir / "machine_reveal.png").exists()
    assert (out_dir / "proof_frame_manifest.json").exists()
    assert commands[0][:4] == ["ffmpeg", "-y", "-ss", "1.25"]
    assert commands[0][-5:] == ["-frames:v", "1", "-update", "1", str(out_dir / "ordinary_bill.png")]


def test_render_frame_sampler_refuses_existing_output_without_force(tmp_path: Path) -> None:
    module = load_module()
    plan_path = tmp_path / "retention_frame_plan.json"
    plan_path.write_text(json.dumps({"frames": [{"id": "ordinary_bill", "sample_time": 1.25}]}))
    render_path = tmp_path / "draft.mp4"
    render_path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + (b"\x00" * 1500))
    out_dir = tmp_path / "proof_frames"
    out_dir.mkdir()

    try:
        module.sample_render_frames(plan_path, render_path, out_dir)
    except FileExistsError as exc:
        assert exc.args[0] == out_dir
    else:
        raise AssertionError("expected FileExistsError")
