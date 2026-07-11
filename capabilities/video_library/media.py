"""FFmpeg-backed media probing, scene detection, and clip extraction."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


SCENE_TIME_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


def resolve_ffmpeg(explicit: str | None = None) -> str:
    if explicit and Path(explicit).expanduser().is_file():
        return str(Path(explicit).expanduser().resolve())
    try:
        from capabilities.moneyprinter import adapter

        runtime = adapter._moneyprinter_runtime_status()
        candidate = str(runtime.get("ffmpegPath") or "")
        if candidate and Path(candidate).is_file():
            return candidate
    except Exception:
        pass
    candidate = shutil.which("ffmpeg")
    if not candidate:
        raise RuntimeError("FFmpeg is not available")
    return candidate


def resolve_ffprobe(ffmpeg_path: str | None = None) -> str:
    if ffmpeg_path:
        sibling = Path(ffmpeg_path).with_name("ffprobe" + (".exe" if Path(ffmpeg_path).suffix.lower() == ".exe" else ""))
        if sibling.is_file():
            return str(sibling)
    candidate = shutil.which("ffprobe")
    if not candidate:
        raise RuntimeError("ffprobe is not available")
    return candidate


def _ratio(value: str) -> float:
    numerator, separator, denominator = str(value or "0").partition("/")
    if not separator:
        return float(numerator or 0)
    denominator_value = float(denominator or 0)
    return float(numerator or 0) / denominator_value if denominator_value else 0.0


def probe_media(path: Path | str, *, ffmpeg_path: str | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    completed = subprocess.run(
        [
            resolve_ffprobe(ffmpeg_path),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(source),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or "ffprobe failed")[-2000:])
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    stream = streams[0] if streams and isinstance(streams[0], dict) else {}
    format_data = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    return {
        "duration_seconds": round(float(format_data.get("duration") or 0), 6),
        "fps": round(_ratio(str(stream.get("r_frame_rate") or "0")), 6),
        "height": int(stream.get("height") or 0),
        "width": int(stream.get("width") or 0),
    }


def _fixed_boundaries(duration_seconds: float, clip_seconds: float, max_clips: int) -> list[tuple[float, float]]:
    boundaries = []
    start = 0.0
    while start < duration_seconds and len(boundaries) < max_clips:
        end = min(duration_seconds, start + clip_seconds)
        boundaries.append((round(start, 6), round(end, 6)))
        start = end
    return boundaries


def parse_scene_boundaries(
    stderr: str,
    *,
    duration_seconds: float,
    min_clip_seconds: float = 1.0,
    max_clips: int = 120,
    fallback_clip_seconds: float = 5.0,
) -> list[tuple[float, float]]:
    duration = max(0.0, float(duration_seconds))
    if duration <= 0:
        return []
    candidates = sorted(
        {
            float(match.group(1))
            for match in SCENE_TIME_RE.finditer(stderr or "")
            if 0 < float(match.group(1)) < duration
        }
    )
    kept = [0.0]
    for timestamp in candidates:
        if timestamp - kept[-1] >= min_clip_seconds:
            kept.append(timestamp)
    if duration - kept[-1] < min_clip_seconds and len(kept) > 1:
        kept.pop()
    kept.append(duration)
    boundaries = [
        (round(kept[index], 6), round(kept[index + 1], 6))
        for index in range(len(kept) - 1)
        if kept[index + 1] > kept[index]
    ]
    if len(boundaries) <= 1 and duration > fallback_clip_seconds:
        return _fixed_boundaries(duration, fallback_clip_seconds, max_clips)
    return boundaries[:max_clips]


def detect_scene_boundaries(
    path: Path | str,
    *,
    duration_seconds: float,
    threshold: float = 0.32,
    min_clip_seconds: float = 1.0,
    max_clips: int = 120,
    fallback_clip_seconds: float = 5.0,
    ffmpeg_path: str | None = None,
) -> list[tuple[float, float]]:
    source = Path(path).expanduser().resolve(strict=True)
    ffmpeg = resolve_ffmpeg(ffmpeg_path)
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(source),
            "-filter:v",
            f"select='gt(scene,{max(0.01, min(0.99, threshold))})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=max(60, int(duration_seconds * 3)),
    )
    return parse_scene_boundaries(
        (completed.stderr or "")[-2_000_000:],
        duration_seconds=duration_seconds,
        min_clip_seconds=min_clip_seconds,
        max_clips=max_clips,
        fallback_clip_seconds=fallback_clip_seconds,
    )


def _managed_output(output: Path | str, library_root: Path | str) -> Path:
    root = Path(library_root).expanduser().resolve()
    target = Path(output).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("media output must stay inside the managed library root") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def extract_clip(
    source_path: Path | str,
    output_path: Path | str,
    *,
    start_seconds: float,
    end_seconds: float,
    library_root: Path | str,
    ffmpeg_path: str | None = None,
) -> Path:
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("clip boundaries must have 0 <= start < end")
    source = Path(source_path).expanduser().resolve(strict=True)
    target = _managed_output(output_path, library_root)
    completed = subprocess.run(
        [
            resolve_ffmpeg(ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_seconds:.6f}",
            "-i",
            str(source),
            "-t",
            f"{end_seconds - start_seconds:.6f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(target),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=max(60, int((end_seconds - start_seconds) * 10)),
    )
    if completed.returncode != 0 or not target.is_file():
        raise RuntimeError((completed.stderr or "FFmpeg clip extraction failed")[-4000:])
    return target


def extract_keyframe(
    source_path: Path | str,
    output_path: Path | str,
    *,
    at_seconds: float,
    library_root: Path | str,
    ffmpeg_path: str | None = None,
) -> Path:
    source = Path(source_path).expanduser().resolve(strict=True)
    target = _managed_output(output_path, library_root)
    completed = subprocess.run(
        [
            resolve_ffmpeg(ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, at_seconds):.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(target),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0 or not target.is_file():
        raise RuntimeError((completed.stderr or "FFmpeg keyframe extraction failed")[-4000:])
    return target
