from __future__ import annotations

import importlib.util
import json
import struct
import zlib
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_motion_smoothness_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_motion_smoothness_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def png_bytes(width: int = 4, height: int = 4, shade: int = 80) -> bytes:
    raw = b"".join(b"\x00" + bytes([shade, shade, shade]) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def test_motion_smoothness_gate_passes_metadata_and_empty_diff_report(tmp_path: Path) -> None:
    module = load_module()
    render = tmp_path / "render.mp4"
    render.write_bytes(b"fake mp4")
    metadata = {
        "streams": [{"codec_type": "video", "avg_frame_rate": "30/1", "r_frame_rate": "30/1"}],
        "format": {"duration": "15.0"},
    }
    diff_report = {"static_intervals": []}

    result = module.evaluate_motion_smoothness(render, metadata=metadata, frame_diff_report=diff_report)

    assert result["passed"] is True
    assert result["fps"] == 30.0
    assert result["duration_seconds"] == 15.0
    assert result["errors"] == []
    assert result["warnings"] == []


def test_motion_smoothness_gate_fails_low_fps_and_long_static_interval(tmp_path: Path) -> None:
    module = load_module()
    render = tmp_path / "render.mp4"
    render.write_bytes(b"fake mp4")
    metadata = {
        "streams": [{"codec_type": "video", "avg_frame_rate": "24000/1001"}],
        "format": {"duration": "15.0"},
    }
    diff_report = {"static_intervals": [{"start": 6.0, "end": 8.0, "duration": 2.0}]}

    result = module.evaluate_motion_smoothness(render, metadata=metadata, frame_diff_report=diff_report)

    assert result["passed"] is False
    assert "fps below minimum: 23.98 < 30.0" in result["errors"]
    assert "static interval too long: 2.00s at 6.0-8.0" in result["errors"]


def test_motion_smoothness_gate_requires_motion_in_each_choreography_beat(tmp_path: Path) -> None:
    module = load_module()
    render = tmp_path / "render.mp4"
    render.write_bytes(b"fake mp4")
    metadata = {
        "streams": [{"codec_type": "video", "avg_frame_rate": "30/1"}],
        "format": {"duration": "15.0"},
    }
    diff_report = {
        "sample_fps": 2.0,
        "static_intervals": [],
        "diffs": [
            {"mean_abs_diff": 4.0, "static": False},
            {"mean_abs_diff": 3.0, "static": False},
            {"mean_abs_diff": 0.0, "static": True},
            {"mean_abs_diff": 0.0, "static": True},
        ],
    }
    choreography = {
        "beats": [
            {"id": "ordinary_bill_hold", "start": 0.0, "end": 1.0},
            {"id": "machine_reveal_handoff", "start": 1.0, "end": 2.0},
        ]
    }

    result = module.evaluate_motion_smoothness(
        render,
        metadata=metadata,
        frame_diff_report=diff_report,
        choreography=choreography,
    )

    assert result["passed"] is False
    assert result["beat_motion"][0]["moving_sample_count"] == 2
    assert result["beat_motion"][1]["moving_sample_count"] == 0
    assert "choreography beat machine_reveal_handoff has no sampled motion evidence" in result["errors"]


def test_motion_smoothness_gate_requires_machine_causality_motion(tmp_path: Path) -> None:
    module = load_module()
    render = tmp_path / "render.mp4"
    render.write_bytes(b"fake mp4")
    metadata = {
        "streams": [{"codec_type": "video", "avg_frame_rate": "30/1"}],
        "format": {"duration": "15.0"},
    }
    diff_report = {
        "sample_fps": 2.0,
        "static_intervals": [],
        "diffs": [
            {"mean_abs_diff": 4.0, "static": False},
            {"mean_abs_diff": 3.0, "static": False},
            {"mean_abs_diff": 0.0, "static": True},
            {"mean_abs_diff": 0.0, "static": True},
        ],
    }
    motion_primitives = {
        "primitives": [
            {"id": "machine_causality", "timeline_marker": "SignalRoomMotion.machineCausality(", "start": 1.0, "end": 2.0}
        ]
    }

    result = module.evaluate_motion_smoothness(
        render,
        metadata=metadata,
        frame_diff_report=diff_report,
        motion_primitives=motion_primitives,
    )

    assert result["passed"] is False
    assert result["primitive_motion"][0]["primitive_id"] == "machine_causality"
    assert result["primitive_motion"][0]["moving_sample_count"] == 0
    assert "motion primitive machine_causality has no sampled motion evidence" in result["errors"]


def test_motion_smoothness_gate_writes_scorecard_with_injected_probe(tmp_path: Path) -> None:
    module = load_module()
    render = tmp_path / "render.mp4"
    render.write_bytes(b"fake mp4")
    out = tmp_path / "motion_smoothness_scorecard.json"

    result = module.write_motion_smoothness_scorecard(
        render,
        out,
        metadata_loader=lambda path: {
            "streams": [{"codec_type": "video", "avg_frame_rate": "60/1"}],
            "format": {"duration": "15.0"},
        },
    )

    assert result["passed"] is True
    assert out.exists()
    assert json.loads(out.read_text())["fps"] == 60.0


def test_motion_smoothness_scorecard_loads_choreography_path(tmp_path: Path) -> None:
    module = load_module()
    render = tmp_path / "render.mp4"
    render.write_bytes(b"fake mp4")
    out = tmp_path / "motion_smoothness_scorecard.json"
    choreography_path = tmp_path / "scene_choreography.json"
    choreography_path.write_text(
        json.dumps({"beats": [{"id": "ordinary_bill_hold", "start": 0.0, "end": 1.0}]})
    )

    result = module.write_motion_smoothness_scorecard(
        render,
        out,
        choreography_path=choreography_path,
        metadata_loader=lambda path: {
            "streams": [{"codec_type": "video", "avg_frame_rate": "60/1"}],
            "format": {"duration": "15.0"},
        },
        frame_diff_report_builder=lambda path: {
            "sample_fps": 2.0,
            "static_intervals": [],
            "diffs": [{"mean_abs_diff": 2.0, "static": False}],
        },
    )

    assert result["passed"] is True
    assert json.loads(out.read_text())["beat_motion"][0]["beat_id"] == "ordinary_bill_hold"


def test_motion_smoothness_scorecard_loads_motion_primitives_path(tmp_path: Path) -> None:
    module = load_module()
    render = tmp_path / "render.mp4"
    render.write_bytes(b"fake mp4")
    out = tmp_path / "motion_smoothness_scorecard.json"
    primitives_path = tmp_path / "motion_primitives.json"
    primitives_path.write_text(
        json.dumps(
            {
                "primitives": [
                    {
                        "id": "machine_causality",
                        "timeline_marker": "SignalRoomMotion.machineCausality(",
                        "start": 0.0,
                        "end": 1.0,
                    }
                ]
            }
        )
    )

    result = module.write_motion_smoothness_scorecard(
        render,
        out,
        motion_primitives_path=primitives_path,
        metadata_loader=lambda path: {
            "streams": [{"codec_type": "video", "avg_frame_rate": "60/1"}],
            "format": {"duration": "15.0"},
        },
        frame_diff_report_builder=lambda path: {
            "sample_fps": 2.0,
            "static_intervals": [],
            "diffs": [{"mean_abs_diff": 2.0, "static": False}],
        },
    )

    assert result["passed"] is True
    assert json.loads(out.read_text())["primitive_motion"][0]["primitive_id"] == "machine_causality"


def test_frame_difference_report_detects_static_hold(tmp_path: Path) -> None:
    module = load_module()
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for idx, shade in enumerate([20, 20, 20, 80], start=1):
        (frame_dir / f"frame_{idx:04d}.png").write_bytes(png_bytes(shade=shade))

    report = module.analyze_frame_differences(frame_dir, sample_fps=2.0, static_threshold=0.5)

    assert report["sample_fps"] == 2.0
    assert report["frame_count"] == 4
    assert report["static_intervals"] == [{"start": 0.0, "end": 1.0, "duration": 1.0}]
    assert report["diffs"][0]["mean_abs_diff"] == 0.0
    assert report["diffs"][-1]["mean_abs_diff"] == 60.0


def test_motion_smoothness_scorecard_generates_diff_report_when_missing(tmp_path: Path) -> None:
    module = load_module()
    render = tmp_path / "render.mp4"
    render.write_bytes(b"fake mp4")
    out = tmp_path / "motion_smoothness_scorecard.json"
    diff_out = tmp_path / "motion_frame_diff_report.json"
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for idx, shade in enumerate([10, 30, 50], start=1):
        (frame_dir / f"frame_{idx:04d}.png").write_bytes(png_bytes(shade=shade))

    result = module.write_motion_smoothness_scorecard(
        render,
        out,
        frame_diff_report_path=diff_out,
        metadata_loader=lambda path: {
            "streams": [{"codec_type": "video", "avg_frame_rate": "30/1"}],
            "format": {"duration": "15.0"},
        },
        frame_diff_report_builder=lambda path: module.analyze_frame_differences(frame_dir, sample_fps=2.0),
    )

    assert result["passed"] is True
    assert result["warnings"] == []
    assert diff_out.exists()
    assert json.loads(diff_out.read_text())["frame_count"] == 3
