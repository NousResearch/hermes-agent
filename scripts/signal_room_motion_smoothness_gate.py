#!/usr/bin/env python3
"""Gate Signal Room renders for basic motion smoothness evidence."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any, Callable


MINIMUM_FPS = 30.0
MAX_STATIC_HOLD_SECONDS = 1.25
DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_STATIC_THRESHOLD = 1.0
REQUIRED_PRIMITIVE_MOTION_IDS = {"machine_causality"}

MetadataLoader = Callable[[Path], dict[str, Any]]
FrameDiffReportBuilder = Callable[[Path], dict[str, Any]]


def _parse_rate(value: str | None) -> float | None:
    if not value:
        return None
    if "/" not in value:
        try:
            return float(value)
        except ValueError:
            return None
    numerator, denominator = value.split("/", 1)
    try:
        den = float(denominator)
        if den == 0:
            return None
        return float(numerator) / den
    except ValueError:
        return None


def load_ffprobe_metadata(render_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(render_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"ffprobe_error": completed.stderr.strip() or completed.stdout.strip()}
    return json.loads(completed.stdout)


def extract_motion_frames(render_path: Path, out_dir: Path, *, sample_fps: float = DEFAULT_SAMPLE_FPS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(render_path),
            "-vf",
            f"fps={sample_fps:g},scale=160:-1",
            str(out_dir / "frame_%04d.png"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _png_chunks(path: Path) -> list[tuple[bytes, bytes]]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG: {path}")
    chunks: list[tuple[bytes, bytes]] = []
    offset = 8
    while offset < len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        chunks.append((kind, chunk_data))
        offset += 12 + length
    return chunks


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    pa = abs(estimate - left)
    pb = abs(estimate - up)
    pc = abs(estimate - upper_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return upper_left


def read_png_pixels(path: Path) -> tuple[int, int, bytes, int]:
    chunks = _png_chunks(path)
    ihdr = next(data for kind, data in chunks if kind == b"IHDR")
    width = int.from_bytes(ihdr[0:4], "big")
    height = int.from_bytes(ihdr[4:8], "big")
    bit_depth = ihdr[8]
    color_type = ihdr[9]
    interlace = ihdr[12]
    if bit_depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise ValueError(f"unsupported PNG format: {path}")
    channels = 3 if color_type == 2 else 4
    compressed = b"".join(data for kind, data in chunks if kind == b"IDAT")
    raw = zlib.decompress(compressed)
    stride = width * channels
    rows: list[bytes] = []
    previous = bytearray(stride)
    pos = 0
    for _ in range(height):
        filter_type = raw[pos]
        pos += 1
        scanline = bytearray(raw[pos : pos + stride])
        pos += stride
        for idx, value in enumerate(scanline):
            left = scanline[idx - channels] if idx >= channels else 0
            up = previous[idx]
            upper_left = previous[idx - channels] if idx >= channels else 0
            if filter_type == 1:
                scanline[idx] = (value + left) & 0xFF
            elif filter_type == 2:
                scanline[idx] = (value + up) & 0xFF
            elif filter_type == 3:
                scanline[idx] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                scanline[idx] = (value + _paeth(left, up, upper_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"unsupported PNG filter {filter_type}: {path}")
        rows.append(bytes(scanline))
        previous = scanline
    return width, height, b"".join(rows), channels


def _mean_abs_diff(left_path: Path, right_path: Path) -> float:
    lw, lh, left, lc = read_png_pixels(left_path)
    rw, rh, right, rc = read_png_pixels(right_path)
    if (lw, lh, lc) != (rw, rh, rc):
        raise ValueError(f"frame dimensions differ: {left_path} vs {right_path}")
    total = sum(abs(a - b) for a, b in zip(left, right, strict=True))
    return total / len(left)


def analyze_frame_differences(
    frame_dir: Path,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    static_threshold: float = DEFAULT_STATIC_THRESHOLD,
) -> dict[str, Any]:
    frames = sorted(frame_dir.glob("frame_*.png"))
    diffs: list[dict[str, Any]] = []
    static_intervals: list[dict[str, Any]] = []
    static_start: int | None = None
    for idx, (left, right) in enumerate(zip(frames, frames[1:], strict=False)):
        mean_diff = _mean_abs_diff(left, right)
        is_static = mean_diff <= static_threshold
        diffs.append(
            {
                "from": left.name,
                "to": right.name,
                "mean_abs_diff": round(mean_diff, 3),
                "static": is_static,
            }
        )
        if is_static and static_start is None:
            static_start = idx
        if not is_static and static_start is not None:
            start = static_start / sample_fps
            end = idx / sample_fps
            static_intervals.append({"start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3)})
            static_start = None
    if static_start is not None:
        start = static_start / sample_fps
        end = (len(frames) - 1) / sample_fps
        static_intervals.append({"start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3)})
    return {
        "sample_fps": sample_fps,
        "static_threshold": static_threshold,
        "frame_count": len(frames),
        "diff_count": len(diffs),
        "static_intervals": static_intervals,
        "diffs": diffs,
    }


def build_frame_diff_report(render_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="signal-room-motion-frames-") as temp:
        frame_dir = Path(temp)
        extract_motion_frames(render_path, frame_dir)
        return analyze_frame_differences(frame_dir)


def _video_stream(metadata: dict[str, Any]) -> dict[str, Any]:
    for stream in metadata.get("streams", []):
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            return stream
    return {}


def evaluate_beat_motion(frame_diff_report: dict[str, Any], choreography: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        sample_fps = float(frame_diff_report.get("sample_fps", DEFAULT_SAMPLE_FPS))
    except (TypeError, ValueError):
        sample_fps = DEFAULT_SAMPLE_FPS
    if sample_fps <= 0:
        sample_fps = DEFAULT_SAMPLE_FPS

    beat_results: list[dict[str, Any]] = []
    diffs = [diff for diff in frame_diff_report.get("diffs", []) if isinstance(diff, dict)]
    for beat in choreography.get("beats", []):
        if not isinstance(beat, dict):
            continue
        beat_id = str(beat.get("id", ""))
        try:
            start = float(beat.get("start"))
            end = float(beat.get("end"))
        except (TypeError, ValueError):
            beat_results.append(
                {
                    "beat_id": beat_id,
                    "start": beat.get("start"),
                    "end": beat.get("end"),
                    "sample_count": 0,
                    "moving_sample_count": 0,
                    "passed": False,
                }
            )
            continue
        sample_count = 0
        moving_sample_count = 0
        for idx, diff in enumerate(diffs):
            diff_start = idx / sample_fps
            diff_end = (idx + 1) / sample_fps
            overlaps = diff_start < end and diff_end > start
            if not overlaps:
                continue
            sample_count += 1
            if not diff.get("static"):
                moving_sample_count += 1
        beat_results.append(
            {
                "beat_id": beat_id,
                "start": start,
                "end": end,
                "sample_count": sample_count,
                "moving_sample_count": moving_sample_count,
                "passed": moving_sample_count > 0,
            }
        )
    return beat_results


def _evaluate_motion_windows(
    frame_diff_report: dict[str, Any],
    windows: list[dict[str, Any]],
    *,
    id_key: str,
    result_key: str,
) -> list[dict[str, Any]]:
    try:
        sample_fps = float(frame_diff_report.get("sample_fps", DEFAULT_SAMPLE_FPS))
    except (TypeError, ValueError):
        sample_fps = DEFAULT_SAMPLE_FPS
    if sample_fps <= 0:
        sample_fps = DEFAULT_SAMPLE_FPS

    results: list[dict[str, Any]] = []
    diffs = [diff for diff in frame_diff_report.get("diffs", []) if isinstance(diff, dict)]
    for window in windows:
        if not isinstance(window, dict):
            continue
        window_id = str(window.get(id_key, ""))
        try:
            start = float(window.get("start"))
            end = float(window.get("end"))
        except (TypeError, ValueError):
            results.append(
                {
                    result_key: window_id,
                    "start": window.get("start"),
                    "end": window.get("end"),
                    "sample_count": 0,
                    "moving_sample_count": 0,
                    "passed": False,
                }
            )
            continue
        sample_count = 0
        moving_sample_count = 0
        for idx, diff in enumerate(diffs):
            diff_start = idx / sample_fps
            diff_end = (idx + 1) / sample_fps
            overlaps = diff_start < end and diff_end > start
            if not overlaps:
                continue
            sample_count += 1
            if not diff.get("static"):
                moving_sample_count += 1
        results.append(
            {
                result_key: window_id,
                "start": start,
                "end": end,
                "sample_count": sample_count,
                "moving_sample_count": moving_sample_count,
                "passed": moving_sample_count > 0,
            }
        )
    return results


def evaluate_primitive_motion(
    frame_diff_report: dict[str, Any],
    motion_primitives: dict[str, Any],
) -> list[dict[str, Any]]:
    required_primitives = [
        primitive
        for primitive in motion_primitives.get("primitives", [])
        if isinstance(primitive, dict) and primitive.get("id") in REQUIRED_PRIMITIVE_MOTION_IDS
    ]
    return _evaluate_motion_windows(
        frame_diff_report,
        required_primitives,
        id_key="id",
        result_key="primitive_id",
    )


def evaluate_motion_smoothness(
    render_path: Path,
    *,
    metadata: dict[str, Any],
    frame_diff_report: dict[str, Any] | None = None,
    choreography: dict[str, Any] | None = None,
    motion_primitives: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not render_path.exists():
        errors.append(f"missing render: {render_path}")

    if metadata.get("ffprobe_error"):
        errors.append(f"ffprobe failed: {metadata['ffprobe_error']}")

    stream = _video_stream(metadata)
    fps = _parse_rate(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""))
    if fps is None:
        errors.append("missing video frame rate")
    elif fps + 0.005 < MINIMUM_FPS:
        errors.append(f"fps below minimum: {fps:.2f} < {MINIMUM_FPS:.1f}")

    duration: float | None = None
    try:
        duration = float(metadata.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        errors.append("missing render duration")

    beat_motion: list[dict[str, Any]] = []
    primitive_motion: list[dict[str, Any]] = []
    if frame_diff_report is None:
        warnings.append("frame-difference report not provided; static-hold check not run")
    else:
        for interval in frame_diff_report.get("static_intervals", []):
            try:
                static_duration = float(interval.get("duration", 0))
            except (TypeError, ValueError):
                continue
            if static_duration > MAX_STATIC_HOLD_SECONDS:
                errors.append(
                    "static interval too long: "
                    f"{static_duration:.2f}s at {interval.get('start')}-{interval.get('end')}"
                )
        if choreography:
            beat_motion = evaluate_beat_motion(frame_diff_report, choreography)
            for beat in beat_motion:
                if not beat["passed"]:
                    errors.append(f"choreography beat {beat['beat_id']} has no sampled motion evidence")
        if motion_primitives:
            primitive_motion = evaluate_primitive_motion(frame_diff_report, motion_primitives)
            for primitive in primitive_motion:
                if not primitive["passed"]:
                    errors.append(
                        f"motion primitive {primitive['primitive_id']} has no sampled motion evidence"
                    )

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "render_path": str(render_path),
        "fps": round(fps, 2) if fps is not None else None,
        "duration_seconds": duration,
        "minimum_fps": MINIMUM_FPS,
        "max_static_hold_seconds": MAX_STATIC_HOLD_SECONDS,
        "beat_motion": beat_motion,
        "primitive_motion": primitive_motion,
    }


def write_motion_smoothness_scorecard(
    render_path: Path,
    out: Path,
    *,
    frame_diff_report_path: Path | None = None,
    choreography_path: Path | None = None,
    motion_primitives_path: Path | None = None,
    metadata_loader: MetadataLoader = load_ffprobe_metadata,
    frame_diff_report_builder: FrameDiffReportBuilder = build_frame_diff_report,
) -> dict[str, Any]:
    metadata = metadata_loader(render_path)
    frame_diff_report = None
    if frame_diff_report_path is not None and frame_diff_report_path.exists():
        frame_diff_report = json.loads(frame_diff_report_path.read_text())
    elif frame_diff_report_path is not None:
        frame_diff_report = frame_diff_report_builder(render_path)
        frame_diff_report_path.parent.mkdir(parents=True, exist_ok=True)
        frame_diff_report_path.write_text(json.dumps(frame_diff_report, indent=2) + "\n")
    else:
        try:
            frame_diff_report = frame_diff_report_builder(render_path)
        except (subprocess.SubprocessError, OSError, ValueError, shutil.Error):
            frame_diff_report = None
    choreography = json.loads(choreography_path.read_text()) if choreography_path and choreography_path.exists() else None
    motion_primitives = (
        json.loads(motion_primitives_path.read_text())
        if motion_primitives_path and motion_primitives_path.exists()
        else None
    )
    result = evaluate_motion_smoothness(
        render_path,
        metadata=metadata,
        frame_diff_report=frame_diff_report,
        choreography=choreography,
        motion_primitives=motion_primitives,
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render", type=Path)
    parser.add_argument("--frame-diff-report", type=Path)
    parser.add_argument("--choreography", type=Path)
    parser.add_argument("--motion-primitives", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = write_motion_smoothness_scorecard(
        args.render,
        args.out,
        frame_diff_report_path=args.frame_diff_report,
        choreography_path=args.choreography,
        motion_primitives_path=args.motion_primitives,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
