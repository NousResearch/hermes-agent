#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def fail(msg: str, code: int = 1):
    print(json.dumps({"success": False, "error": msg}, ensure_ascii=False))
    raise SystemExit(code)


def check_deps():
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        missing.append("yt-dlp (python package)")
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        missing.append("faster-whisper")
    if missing:
        fail("Missing dependencies: " + ", ".join(missing))


def sanitize_name(name: str) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    out = "".join("_" if c in bad else c for c in name).strip()
    return out[:180] or "video"


def download_audio(url: str, outdir: Path):
    from yt_dlp import YoutubeDL

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(outdir / "%(title).160s [%(id)s].%(ext)s"),
        "noplaylist": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        return info, Path(filepath)


def to_wav_16k_mono(src: Path, dst: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", "16000", "-vn",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        fail(f"ffmpeg conversion failed: {proc.stderr[-1200:]}")


def transcribe(wav_path: Path, model_size: str, task: str, language: str | None):
    from faster_whisper import WhisperModel

    # Force CPU for maximum portability (avoids CUDA runtime dependency on hosts without libcublas)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(wav_path),
        task=task,
        language=language,
        vad_filter=True,
        beam_size=5,
    )

    segs = []
    full_text_parts = []
    for s in segments:
        text = (s.text or "").strip()
        seg = {
            "start": round(float(s.start), 3),
            "end": round(float(s.end), 3),
            "text": text,
        }
        segs.append(seg)
        if text:
            full_text_parts.append(text)

    full_text = "\n".join(full_text_parts).strip()

    info_obj = {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "duration_after_vad": getattr(info, "duration_after_vad", None),
    }
    return full_text, segs, info_obj


def main():
    p = argparse.ArgumentParser(description="Video Intel Pipeline: yt-dlp + ffmpeg + faster-whisper")
    p.add_argument("--url", required=True, help="Video URL")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--model", default="small", help="Whisper model size (tiny/base/small/medium/large-v3)")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="ASR task")
    p.add_argument("--language", default=None, help="Force source language, e.g. es")
    args = p.parse_args()

    check_deps()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    info, downloaded_audio = download_audio(args.url, outdir)

    title = info.get("title") or "video"
    video_id = info.get("id")
    safe_base = sanitize_name(f"{title} [{video_id}]" if video_id else title)

    workdir = outdir / safe_base
    workdir.mkdir(parents=True, exist_ok=True)

    audio_dst = workdir / "source_audio" / downloaded_audio.name
    audio_dst.parent.mkdir(parents=True, exist_ok=True)
    if downloaded_audio.resolve() != audio_dst.resolve():
        audio_dst.write_bytes(downloaded_audio.read_bytes())

    wav_path = workdir / "audio_16k_mono.wav"
    to_wav_16k_mono(audio_dst, wav_path)

    full_text, segments, asr_info = transcribe(
        wav_path=wav_path,
        model_size=args.model,
        task=args.task,
        language=args.language,
    )

    metadata = {
        "id": info.get("id"),
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "channel": info.get("channel"),
        "webpage_url": info.get("webpage_url"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "description": info.get("description"),
        "extractor": info.get("extractor"),
    }

    (workdir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (workdir / "transcript.txt").write_text(full_text + "\n", encoding="utf-8")
    (workdir / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "success": True,
        "input_url": args.url,
        "output_dir": str(workdir),
        "files": {
            "metadata": str(workdir / "metadata.json"),
            "transcript": str(workdir / "transcript.txt"),
            "segments": str(workdir / "segments.json"),
            "wav": str(wav_path),
        },
        "video": metadata,
        "asr": {
            "task": args.task,
            "model": args.model,
            "forced_language": args.language,
            "detected_language": asr_info.get("language"),
            "detected_language_probability": asr_info.get("language_probability"),
            "duration_seconds": asr_info.get("duration"),
        },
        "segments_count": len(segments),
        "transcript_chars": len(full_text),
    }

    (workdir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Interrupted by user", code=130)
