"""Local-only video analysis helpers for gateway inbound media."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gateway.platforms.base import get_video_cache_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoAnalysisLimits:
    max_bytes: int = 20 * 1024 * 1024
    max_seconds: float = 120.0
    frames: int = 4
    timeout_seconds: float = 45.0
    transcribe: bool = False


def _run(cmd: list[str], *, timeout: float, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        timeout=timeout,
    )


def _require_binary(name: str) -> str | None:
    return shutil.which(name)


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return stem[:80] or "video"


def _duration_seconds(probe: dict[str, Any]) -> float:
    try:
        return max(0.0, float(probe.get("format", {}).get("duration") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _has_audio(probe: dict[str, Any]) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))


def _frame_times(duration: float, count: int) -> list[float]:
    count = max(0, min(count, 12))
    if count == 0:
        return []
    if duration <= 0:
        return [0.0]
    if count == 1:
        return [min(duration * 0.5, max(0.0, duration - 0.1))]
    return [min(duration - 0.1, duration * (idx + 1) / (count + 1)) for idx in range(count)]


def _ffprobe(video_path: Path, timeout: float) -> dict[str, Any]:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,duration",
            "-of",
            "json",
            str(video_path),
        ],
        timeout=timeout,
        capture=True,
    )
    return json.loads(result.stdout)


def _extract_frames(video_path: Path, outdir: Path, times: list[float], timeout: float) -> list[str]:
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames: list[str] = []
    for idx, seconds in enumerate(times, start=1):
        frame = frames_dir / f"frame-{idx:02d}-{int(seconds):04d}s.jpg"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{seconds:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame),
            ],
            timeout=timeout,
        )
        frames.append(str(frame))
    return frames


def _extract_audio(video_path: Path, outdir: Path, probe: dict[str, Any], timeout: float) -> str | None:
    if not _has_audio(probe):
        return None
    audio_path = outdir / "audio.wav"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        timeout=timeout,
    )
    return str(audio_path)


def _transcribe_local_only(audio_path: str) -> dict[str, Any]:
    try:
        from tools import transcription_tools as stt
    except Exception as exc:
        return {"success": False, "transcript": "", "error": f"STT unavailable: {exc}"}

    stt_config = stt._load_stt_config()
    provider = str(stt_config.get("provider") or "local").strip().lower()
    if provider not in {"local", "local_command"}:
        return {
            "success": False,
            "transcript": "",
            "error": f"Skipped non-local STT provider: {provider}",
        }
    if provider == "local" and not getattr(stt, "_HAS_FASTER_WHISPER", False):
        if not stt._has_local_command():
            return {
                "success": False,
                "transcript": "",
                "error": "Skipped local STT: faster-whisper/local command unavailable",
            }
    return stt.transcribe_audio(audio_path)


def analyze_video_local(video_path: str, limits: VideoAnalysisLimits | None = None) -> dict[str, Any]:
    """Analyze a cached video without network calls.

    Returns a JSON-serializable dict. Errors are reported in the dict so the
    inbound pipeline can degrade gracefully instead of failing the message.
    """
    limits = limits or VideoAnalysisLimits()
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        return {"status": "skipped", "reason": "video path does not exist"}
    try:
        size = path.stat().st_size
    except OSError as exc:
        return {"status": "skipped", "reason": f"cannot stat video: {exc}"}
    if size > limits.max_bytes:
        return {
            "status": "skipped",
            "reason": f"video too large: {size} bytes > {limits.max_bytes}",
            "input": str(path),
            "size_bytes": size,
        }
    if not _require_binary("ffprobe") or not _require_binary("ffmpeg"):
        return {"status": "skipped", "reason": "ffmpeg/ffprobe unavailable", "input": str(path)}

    outdir = get_video_cache_dir() / "analysis" / f"{_safe_stem(path)}-{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        probe = _ffprobe(path, limits.timeout_seconds)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"status": "skipped", "reason": f"ffprobe failed: {exc}", "input": str(path)}

    duration = _duration_seconds(probe)
    (outdir / "ffprobe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")
    if duration > limits.max_seconds:
        return {
            "status": "skipped",
            "reason": f"video too long: {duration:.1f}s > {limits.max_seconds:.1f}s",
            "input": str(path),
            "outdir": str(outdir),
            "duration_seconds": duration,
            "size_bytes": size,
            "ffprobe": str(outdir / "ffprobe.json"),
        }

    try:
        audio_path = _extract_audio(path, outdir, probe, limits.timeout_seconds)
        frames = _extract_frames(path, outdir, _frame_times(duration, limits.frames), limits.timeout_seconds)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "partial",
            "reason": f"media extraction failed: {exc}",
            "input": str(path),
            "outdir": str(outdir),
            "duration_seconds": duration,
            "size_bytes": size,
            "ffprobe": str(outdir / "ffprobe.json"),
        }

    transcript: str | None = None
    transcript_error: str | None = None
    if limits.transcribe and audio_path:
        stt_result = _transcribe_local_only(audio_path)
        if stt_result.get("success"):
            transcript = str(stt_result.get("transcript") or "").strip() or None
            (outdir / "transcript.json").write_text(
                json.dumps(stt_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            transcript_error = str(stt_result.get("error") or "local STT failed")

    return {
        "status": "ok",
        "input": str(path),
        "outdir": str(outdir),
        "duration_seconds": duration,
        "size_bytes": size,
        "ffprobe": str(outdir / "ffprobe.json"),
        "audio": audio_path,
        "frames": frames,
        "transcript": transcript,
        "transcript_error": transcript_error,
    }


def format_video_analysis_context(result: dict[str, Any]) -> str:
    status = result.get("status")
    if status == "skipped":
        return f"[Telegram video analysis skipped: {result.get('reason', 'unknown reason')}]"

    parts = [
        "[Telegram video local analysis]",
        f"duration_seconds: {result.get('duration_seconds', 0):.1f}",
        f"analysis_dir: {result.get('outdir')}",
    ]
    frames = result.get("frames") or []
    if frames:
        parts.append("vision_frame_paths:")
        parts.extend(f"- {frame}" for frame in frames)
    if result.get("transcript"):
        parts.append("transcript:")
        parts.append(str(result["transcript"]))
    elif result.get("audio"):
        parts.append(f"audio_path: {result.get('audio')}")
        if result.get("transcript_error"):
            parts.append(f"transcript_status: {result.get('transcript_error')}")
    if status == "partial" and result.get("reason"):
        parts.append(f"partial_reason: {result.get('reason')}")
    return "\n".join(parts)
