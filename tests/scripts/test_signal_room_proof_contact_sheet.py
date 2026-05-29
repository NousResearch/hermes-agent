from __future__ import annotations

import importlib.util
import json
import struct
import zlib
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_proof_contact_sheet.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_proof_contact_sheet", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_png(path: Path, *, shade: int = 180) -> None:
    width, height = 1080, 1920
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
                "frames": [
                    {
                        "id": "ordinary_bill",
                        "sample_time": 1.25,
                        "beat": "human problem",
                        "review_question": "Who is affected?",
                    },
                    {
                        "id": "number_split",
                        "sample_time": 3.75,
                        "beat": "visible contradiction",
                        "review_question": "Does the total visibly split?",
                    },
                    {
                        "id": "machine_reveal",
                        "sample_time": 6.5,
                        "beat": "mechanism reveal",
                        "review_question": "Is the machine visible?",
                    },
                    {
                        "id": "acting_read",
                        "sample_time": 9.25,
                        "beat": "escalation",
                        "review_question": "Does the gesture clarify action?",
                    },
                    {
                        "id": "memory_anchor",
                        "sample_time": 13.2,
                        "beat": "memory anchor",
                        "review_question": "Can this frame be remembered?",
                    },
                ],
            }
        )
    )


def test_proof_contact_sheet_writes_review_svg_with_all_retention_frames(tmp_path: Path) -> None:
    module = load_module()
    plan = tmp_path / "retention_frame_plan.json"
    frames = tmp_path / "proof_frames"
    out = tmp_path / "proof_contact_sheet.svg"
    frames.mkdir()
    write_frame_plan(plan)
    for index, frame_id in enumerate(module.required_frame_ids(plan), start=1):
        write_png(frames / f"{frame_id}.png", shade=200 - index)

    result = module.write_proof_contact_sheet(plan, frames, out)

    assert result["passed"] is True
    assert result["output"] == str(out)
    svg = out.read_text()
    assert '<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">' in svg
    assert "Signal Room proof retention contact sheet" in svg
    assert "ordinary_bill" in svg
    assert "1.25s" in svg
    assert "Who is affected?" in svg
    assert "memory_anchor.png" in svg


def test_proof_contact_sheet_rejects_missing_frame(tmp_path: Path) -> None:
    module = load_module()
    plan = tmp_path / "retention_frame_plan.json"
    frames = tmp_path / "proof_frames"
    frames.mkdir()
    write_frame_plan(plan)

    result = module.write_proof_contact_sheet(plan, frames, tmp_path / "out.svg")

    assert result["passed"] is False
    assert "missing required frame: ordinary_bill.png" in result["errors"]
    assert not (tmp_path / "out.svg").exists()
