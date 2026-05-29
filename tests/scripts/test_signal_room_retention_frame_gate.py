from __future__ import annotations

import importlib.util
import json
import struct
import zlib
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_retention_frame_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_retention_frame_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_png(path: Path, *, width: int = 1080, height: int = 1920, shade: int = 255) -> None:
    raw = b"".join(b"\x00" + bytes([shade, shade, shade]) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def write_frame_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "review-only",
                "contact_sheet_required": True,
                "frames": [
                    {"id": "ordinary_bill", "sample_time": 1.25},
                    {"id": "number_split", "sample_time": 3.75},
                    {"id": "machine_reveal", "sample_time": 6.5},
                    {"id": "acting_read", "sample_time": 9.25},
                    {"id": "memory_anchor", "sample_time": 13.2},
                ],
            }
        )
    )


def write_manifest(frames_dir: Path, *, public_release: bool = False) -> None:
    (frames_dir / "proof_frame_manifest.json").write_text(
        json.dumps(
            {
                "source_composition": "fee-machine-v2-review",
                "public_release": public_release,
                "render_tool": "hyperframes inspect/render frame export",
            }
        )
    )


def test_retention_frame_gate_passes_for_distinct_exact_size_frames(tmp_path: Path) -> None:
    module = load_module()
    plan = tmp_path / "retention_frame_plan.json"
    frames = tmp_path / "frames"
    frames.mkdir()
    write_frame_plan(plan)
    write_manifest(frames)
    for index, frame_id in enumerate(module.required_frame_ids(plan), start=1):
        write_png(frames / f"{frame_id}.png", shade=220 - index)

    result = module.evaluate_retention_frames(plan, frames)

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["required_frame_ids"] == [
        "ordinary_bill",
        "number_split",
        "machine_reveal",
        "acting_read",
        "memory_anchor",
    ]
    assert result["frames"]["memory_anchor.png"]["width"] == 1080
    assert result["frames"]["memory_anchor.png"]["height"] == 1920


def test_retention_frame_gate_maps_choreography_beats_to_valid_review_frames(tmp_path: Path) -> None:
    module = load_module()
    plan = tmp_path / "retention_frame_plan.json"
    choreography = tmp_path / "scene_choreography.json"
    frames = tmp_path / "frames"
    frames.mkdir()
    write_frame_plan(plan)
    write_manifest(frames)
    for index, frame_id in enumerate(module.required_frame_ids(plan), start=1):
        write_png(frames / f"{frame_id}.png", shade=220 - index)
    choreography.write_text(
        json.dumps(
            {
                "beats": [
                    {"id": "machine_reveal_handoff", "review_frame": "machine_reveal"},
                    {"id": "skeptical_point_drive", "review_frame": "acting_read"},
                ]
            }
        )
    )

    result = module.evaluate_retention_frames(plan, frames, choreography_path=choreography)

    assert result["passed"] is True
    assert result["choreography_frame_coverage"] == [
        {
            "beat_id": "machine_reveal_handoff",
            "review_frame": "machine_reveal",
            "frame_name": "machine_reveal.png",
            "frame_present": True,
            "frame_valid": True,
        },
        {
            "beat_id": "skeptical_point_drive",
            "review_frame": "acting_read",
            "frame_name": "acting_read.png",
            "frame_present": True,
            "frame_valid": True,
        },
    ]


def test_retention_frame_gate_fails_choreography_without_valid_review_frame(tmp_path: Path) -> None:
    module = load_module()
    plan = tmp_path / "retention_frame_plan.json"
    choreography = tmp_path / "scene_choreography.json"
    frames = tmp_path / "frames"
    frames.mkdir()
    write_frame_plan(plan)
    write_manifest(frames)
    for index, frame_id in enumerate(module.required_frame_ids(plan), start=1):
        if frame_id != "acting_read":
            write_png(frames / f"{frame_id}.png", shade=220 - index)
    choreography.write_text(
        json.dumps(
            {
                "beats": [
                    {"id": "machine_reveal_handoff", "review_frame": "missing_frame"},
                    {"id": "skeptical_point_drive", "review_frame": "acting_read"},
                    {"id": "memory_anchor_settle"},
                ]
            }
        )
    )

    result = module.evaluate_retention_frames(plan, frames, choreography_path=choreography)

    assert result["passed"] is False
    assert "choreography beat machine_reveal_handoff review_frame missing_frame is not in retention plan" in result["errors"]
    assert "choreography beat skeptical_point_drive review frame missing: acting_read.png" in result["errors"]
    assert "choreography beat memory_anchor_settle missing review_frame" in result["errors"]


def test_retention_frame_gate_fails_for_missing_wrong_size_or_duplicate_frames(tmp_path: Path) -> None:
    module = load_module()
    plan = tmp_path / "retention_frame_plan.json"
    frames = tmp_path / "frames"
    frames.mkdir()
    write_frame_plan(plan)
    write_manifest(frames, public_release=True)
    write_png(frames / "ordinary_bill.png", width=720, height=1280)
    write_png(frames / "number_split.png", shade=12)
    write_png(frames / "machine_reveal.png", shade=12)
    write_png(frames / "acting_read.png", shade=80)

    result = module.evaluate_retention_frames(plan, frames)

    assert result["passed"] is False
    assert "missing required frame: memory_anchor.png" in result["errors"]
    assert "ordinary_bill.png dimensions 720x1280 must be 1080x1920" in result["errors"]
    assert "frame images must be visually distinct; duplicate file content detected" in result["errors"]
    assert "manifest public_release must be false for review gate" in result["errors"]
