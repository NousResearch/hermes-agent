#!/usr/bin/env python3
"""Validate a presenter-video job and write a machine-readable preflight report."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,pix_fmt,sample_rate,channels,r_frame_rate:format=duration,size,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def require_file(value: str, label: str, errors: list[str]) -> Path | None:
    if not value:
        errors.append(f"missing {label}")
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        errors.append(f"{label} is not a file: {path}")
        return None
    return path


def write_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: preflight.py ~/Videos/my-presenter-video/job.json", file=sys.stderr)
        return 64
    if not shutil.which("ffprobe"):
        print("ERROR: ffprobe is required", file=sys.stderr)
        return 2

    job_path = Path(sys.argv[1]).expanduser().resolve()
    if not job_path.is_file():
        print(f"ERROR: job manifest does not exist: {job_path}", file=sys.stderr)
        return 2
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_dir = job_path.parent
    errors: list[str] = []
    remote_blockers: list[str] = []
    warnings: list[str] = []
    media: dict[str, Any] = {}

    input_data = job.get("input", {})
    topic = str(input_data.get("topic", "")).strip()
    script_text = str(input_data.get("script_path", "")).strip()
    if not topic and not script_text:
        errors.append("topic or script_path is required")
    if script_text:
        require_file(script_text, "script", errors)

    image = require_file(str(input_data.get("presenter_image", "")), "presenter image", errors)
    if image:
        try:
            media["presenter_image"] = probe(image)
            streams = [
                stream
                for stream in media["presenter_image"].get("streams", [])
                if stream.get("codec_type") == "video"
            ]
            if not streams:
                errors.append("presenter image has no decodable image/video stream")
            else:
                width = int(streams[0].get("width") or 0)
                height = int(streams[0].get("height") or 0)
                if min(width, height) < 512:
                    warnings.append(f"presenter image is low resolution: {width}x{height}")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            errors.append(f"could not decode presenter image: {exc}")

    voice_value = str(input_data.get("voice_sample", "")).strip()
    if voice_value:
        voice = require_file(voice_value, "voice sample", errors)
        if voice:
            try:
                media["voice_sample"] = probe(voice)
                streams = [
                    stream
                    for stream in media["voice_sample"].get("streams", [])
                    if stream.get("codec_type") == "audio"
                ]
                if not streams:
                    errors.append("voice sample has no audio stream")
                duration = float(media["voice_sample"].get("format", {}).get("duration") or 0)
                if duration < 4:
                    warnings.append(f"voice sample is short: {duration:.3f}s")
                if duration > 60:
                    warnings.append(f"voice sample is unusually long: {duration:.3f}s")
            except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"could not decode voice sample: {exc}")
        if not input_data.get("voice_clone_approved"):
            remote_blockers.append("voice_clone_approved must be true before voice cloning")
    else:
        warnings.append("no voice sample supplied; use and record a stock voice")

    supporting_reports = []
    for value in input_data.get("supporting_media", []):
        path = require_file(str(value), "supporting media", errors)
        if path:
            try:
                supporting_reports.append({"file": path.name, "probe": probe(path)})
            except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
                errors.append(f"could not decode supporting media {path}: {exc}")
    media["supporting_media"] = supporting_reports

    if not input_data.get("rights_confirmed"):
        remote_blockers.append("rights_confirmed must be true before presenter synthesis")
    if not input_data.get("adult_presenter_confirmed"):
        remote_blockers.append("adult_presenter_confirmed must be true before presenter synthesis")
    if not input_data.get("remote_upload_approved"):
        remote_blockers.append("remote_upload_approved must be true before remote generation")

    manual = job.get("manual_input_review", {})
    for key in ("image_viewed", "single_clear_face", "image_has_no_unwanted_text"):
        if not manual.get(key):
            errors.append(f"manual_input_review.{key} must be true")
    if voice_value:
        for key in ("voice_sample_listened", "single_clear_speaker"):
            if not manual.get(key):
                errors.append(f"manual_input_review.{key} must be true")

    creative = job.get("creative", {})
    duration = float(creative.get("duration_target_s") or 0)
    if not 5 <= duration <= 1800:
        errors.append("creative.duration_target_s must be between 5 and 1800")
    width = int(creative.get("width") or 0)
    height = int(creative.get("height") or 0)
    if min(width, height) < 256 or max(width, height) > 7680:
        errors.append("creative width/height must be between 256 and 7680")
    if int(creative.get("fps") or 0) not in (24, 25, 30, 50, 60):
        errors.append("creative.fps must be one of 24, 25, 30, 50, or 60")

    report_path = job_dir / "qa" / "reports" / "preflight.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "ok": not errors,
        "remote_ready": not errors and not remote_blockers,
        "job": job_path.name,
        "errors": errors,
        "remote_blockers": remote_blockers,
        "warnings": warnings,
        "media": media,
    }
    write_atomic(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    job.setdefault("qa", {})["preflight_report"] = "qa/reports/preflight.json"
    write_atomic(job_path, json.dumps(job, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
