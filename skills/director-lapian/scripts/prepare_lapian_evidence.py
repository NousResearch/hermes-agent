#!/usr/bin/env python3
"""Prepare evidence assets for director-lapian analysis.

This script creates the standard project archive, samples frames through
video_frame_sampler.py, extracts a 16k mono WAV when audio exists, writes
per-second audio levels, and records a root manifest. ASR is optional because
it may download models or take a long time.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLER = SCRIPT_DIR / "video_frame_sampler.py"
SHOT_DETECTOR = SCRIPT_DIR / "detect_shot_cuts.py"
ILLEGAL_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def run_command(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"missing required tool: {name}")


def sanitize_name(value: str, fallback: str = "未命名视频") -> str:
    value = ILLEGAL_PATH_CHARS.sub("_", value).strip().strip(".")
    value = re.sub(r"\s+", "_", value)
    return value or fallback


def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 1000):
        candidate = path.with_name(f"{path.name}_v{idx:02d}")
        if not candidate.exists():
            return candidate
    raise SystemExit(f"could not create a unique project directory near: {path}")


def parse_ffprobe(video: Path) -> dict[str, Any]:
    result = run_command([
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video),
    ])
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed:\n{result.stderr}")
    data = json.loads(result.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    duration = data.get("format", {}).get("duration") or video_stream.get("duration")
    return {
        "source_path": str(video),
        "source_name": video.name,
        "source_size_bytes": video.stat().st_size,
        "duration_seconds": float(duration) if duration else None,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "video_codec": video_stream.get("codec_name"),
        "avg_frame_rate": video_stream.get("avg_frame_rate"),
        "audio_stream_count": len(audio_streams),
        "audio_streams": [
            {
                "codec": s.get("codec_name"),
                "sample_rate": s.get("sample_rate"),
                "channels": s.get("channels"),
                "duration": s.get("duration"),
            }
            for s in audio_streams
        ],
    }


def make_project_dirs(project_dir: Path) -> dict[str, Path]:
    dirs = {
        "source_info": project_dir / "00_源信息",
        "frames": project_dir / "01_逐秒抽帧",
        "audio": project_dir / "02_音频分析",
        "markdown_reports": project_dir / "03_MD报告",
        "word_delivery": project_dir / "04_飞书Word交付",
        "pdf_preview": project_dir / "05_PDF预览",
        "qa": project_dir / "06_QA审计",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run_frame_sampler(args: argparse.Namespace, video: Path, frames_dir: Path) -> dict[str, Any]:
    if args.skip_frames:
        return {"skipped": True}
    cmd = [
        sys.executable,
        str(SAMPLER),
        str(video),
        "--out-dir",
        str(frames_dir),
        "--interval",
        str(args.interval),
    ]
    if args.start:
        cmd.extend(["--start", args.start])
    if args.duration:
        cmd.extend(["--duration", args.duration])
    if args.max_frames:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.contact_sheet:
        cmd.append("--contact-sheet")
    result = run_command(cmd)
    if result.returncode != 0:
        raise SystemExit(f"frame sampler failed:\n{result.stderr or result.stdout}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw_stdout": result.stdout}


def run_shot_detection(args: argparse.Namespace, video: Path, frames_dir: Path, title: str) -> dict[str, Any]:
    if args.skip_shot_detect:
        return {"skipped": True, "reason": "skip requested"}
    if not SHOT_DETECTOR.exists():
        return {"skipped": True, "reason": f"missing detector script: {SHOT_DETECTOR}"}
    out_path = frames_dir / f"{title}_镜头切分.json"
    cmd = [
        sys.executable,
        str(SHOT_DETECTOR),
        str(video),
        "--out",
        str(out_path),
        "--threshold",
        str(args.scene_threshold),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        # shot detection is an upgrade, not a prerequisite; never abort evidence prep
        return {"skipped": True, "reason": f"shot detection failed: {(result.stderr or result.stdout).strip()[:500]}"}
    try:
        summary = json.loads(result.stdout)
    except json.JSONDecodeError:
        summary = {"raw_stdout": result.stdout}
    summary["cuts_json"] = str(out_path)
    return summary


def extract_audio(video: Path, out_wav: Path, start: str | None, duration: str | None) -> dict[str, Any]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if start:
        cmd.extend(["-ss", start])
    cmd.extend([
        "-i",
        str(video),
    ])
    if duration:
        cmd.extend(["-t", duration])
    cmd.extend([
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ])
    result = run_command(cmd)
    if result.returncode != 0:
        raise SystemExit(f"audio extraction failed:\n{result.stderr}")
    return {"wav": str(out_wav)}


def write_audio_levels(wav_path: Path, csv_path: Path, window_seconds: float = 1.0) -> dict[str, Any]:
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        if sample_width != 2:
            raise SystemExit(f"unsupported WAV sample width for level analysis: {sample_width}")
        samples_per_window = max(1, int(sample_rate * window_seconds))
        max_int = 32768.0
        rows = []
        index = 0
        while True:
            raw = wf.readframes(samples_per_window)
            if not raw:
                break
            values = [
                int.from_bytes(raw[i : i + 2], "little", signed=True)
                for i in range(0, len(raw), 2)
            ]
            if not values:
                break
            peak = max(abs(v) for v in values) / max_int
            rms = math.sqrt(sum(v * v for v in values) / len(values)) / max_int
            rows.append({
                "second": index,
                "start_seconds": round(index * window_seconds, 3),
                "end_seconds": round(index * window_seconds + len(values) / sample_rate / channels, 3),
                "rms": round(rms, 6),
                "peak": round(peak, 6),
            })
            index += 1
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["second", "start_seconds", "end_seconds", "rms", "peak"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "levels_csv": str(csv_path),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "level_rows": len(rows),
    }


def run_optional_asr(args: argparse.Namespace, wav_path: Path, audio_dir: Path, title: str) -> dict[str, Any]:
    if not args.asr_model:
        return {"skipped": True, "reason": "asr disabled"}
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {"skipped": True, "reason": "faster_whisper is not installed"}

    model = WhisperModel(args.asr_model, device=args.asr_device, compute_type=args.asr_compute_type)
    segments_iter, info = model.transcribe(
        str(wav_path),
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    segments = []
    for segment in segments_iter:
        segments.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
        })
    base = audio_dir / f"{title}_asr_{args.asr_model}"
    json_path = base.with_suffix(".json")
    txt_path = base.with_suffix(".txt")
    jsonl_path = base.with_suffix(".jsonl")
    payload = {
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "model": args.asr_model,
        "segments": segments,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text("\n".join(f"[{s['start']:.3f}-{s['end']:.3f}] {s['text']}" for s in segments), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for segment in segments:
            f.write(json.dumps(segment, ensure_ascii=False) + "\n")
    return {
        "model": args.asr_model,
        "segment_count": len(segments),
        "json": str(json_path),
        "txt": str(txt_path),
        "jsonl": str(jsonl_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare standard evidence archive for director-lapian.")
    parser.add_argument("video", type=Path, help="input video path")
    parser.add_argument("--out-root", type=Path, default=Path("08拉片输出"), help="root output folder")
    parser.add_argument("--project-name", help="project folder name; defaults to video stem")
    parser.add_argument("--task-name", default="导演级拉片", help="task suffix used in project folder")
    parser.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"), help="archive date folder, YYYYMMDD")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between sampled frames")
    parser.add_argument("--start", help="optional start time, seconds or hh:mm:ss")
    parser.add_argument("--duration", help="optional duration, seconds or hh:mm:ss")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--contact-sheet", action="store_true", default=True, help="create contact sheets")
    parser.add_argument("--no-contact-sheet", action="store_false", dest="contact_sheet")
    parser.add_argument("--skip-frames", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-shot-detect", action="store_true", help="skip ffmpeg scene-change shot detection")
    parser.add_argument("--scene-threshold", type=float, default=0.3, help="scene-change score threshold for shot detection")
    parser.add_argument("--asr-model", help="optional faster-whisper model name, e.g. base or small")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--asr-compute-type", default="int8")
    return parser.parse_args()


def main() -> int:
    # Windows GBK consoles cannot print replacement chars from subprocess output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    require_tool("ffmpeg")
    require_tool("ffprobe")
    if not SAMPLER.exists():
        raise SystemExit(f"missing frame sampler: {SAMPLER}")
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than 0")

    video = args.video.expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    title = sanitize_name(args.project_name or video.stem)
    task = sanitize_name(args.task_name, "导演级拉片")
    project_name = title if not task else f"{title}_{task}"
    project_dir = unique_dir((args.out_root / args.date / project_name).resolve())
    project_dir.mkdir(parents=True, exist_ok=False)
    dirs = make_project_dirs(project_dir)

    metadata = parse_ffprobe(video)
    (dirs["source_info"] / "ffprobe_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    frame_result = run_frame_sampler(args, video, dirs["frames"])
    shot_result = run_shot_detection(args, video, dirs["frames"], title)

    audio_result: dict[str, Any]
    if args.skip_audio or metadata["audio_stream_count"] == 0:
        audio_result = {"skipped": True, "reason": "skip requested or no audio stream"}
    else:
        wav_path = dirs["audio"] / f"{title}_audio_16k_mono.wav"
        levels_path = dirs["audio"] / f"{title}_audio_levels.csv"
        audio_result = extract_audio(video, wav_path, args.start, args.duration)
        audio_result.update(write_audio_levels(wav_path, levels_path))
        audio_result["asr"] = run_optional_asr(args, wav_path, dirs["audio"], title)

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(project_dir),
        "source": metadata,
        "directories": {key: str(value) for key, value in dirs.items()},
        "frame_sampling": frame_result,
        "shot_detection": shot_result,
        "audio": audio_result,
        "reports": {
            "markdown_main": None,
            "markdown_second_by_second": None,
            "docx_embedded": None,
            "pdf_preview": None,
            "qa": None,
        },
        "limitations": [],
    }
    if not args.asr_model:
        manifest["limitations"].append("ASR not run by default; use --asr-model when transcription is needed.")
    if shot_result.get("skipped"):
        manifest["limitations"].append(f"shot detection skipped: {shot_result.get('reason', 'unknown')}")
    manifest_path = project_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "project_dir": str(project_dir),
        "manifest": str(manifest_path),
        "frames_dir": str(dirs["frames"]),
        "audio_dir": str(dirs["audio"]),
        "qa_dir": str(dirs["qa"]),
        "frame_count": frame_result.get("frame_count"),
        "shot_detection": shot_result,
        "audio": audio_result,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
