#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def check_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_doctor() -> int:
    checks = [
        ("python3", check_cmd("python3")),
        ("ffmpeg", check_cmd("ffmpeg")),
        ("yt-dlp", check_cmd("yt-dlp")),
    ]

    for name, ok in checks:
        print(f"{'OK ' if ok else 'MISS'} {name}")

    try:
        import yt_dlp  # noqa: F401
    except Exception as exc:
        print(f"MISS yt_dlp python package ({exc})", file=sys.stderr)
        return 1

    try:
        import faster_whisper  # noqa: F401
    except Exception as exc:
        print(f"MISS faster-whisper ({exc})", file=sys.stderr)
        return 1

    print("OK  yt_dlp python package")
    print("OK  faster-whisper")

    missing = [name for name, ok in checks if not ok]
    if missing:
        print(f"Missing system dependencies: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


def extract_info(url: str, cookies_from_browser: str | None = None) -> dict[str, Any]:
    import yt_dlp

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "id": info.get("id"),
        "title": info.get("title") or "untitled",
        "duration": info.get("duration"),
        "extractor_key": info.get("extractor_key") or info.get("extractor") or "unknown",
        "webpage_url": info.get("webpage_url") or url,
    }


def download_audio_source(url: str, cookies_from_browser: str | None = None) -> Path:
    import yt_dlp

    out_dir = Path(tempfile.mkdtemp(prefix="hermes_vts_"))
    outtmpl = str(out_dir / "source.%(ext)s")

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "restrictfilenames": True,
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    return Path(filename)


def normalize_to_wav16k(input_path: Path) -> Path:
    out_dir = Path(tempfile.mkdtemp(prefix="hermes_vts_audio_"))
    out_path = out_dir / "audio_16k_mono.wav"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def transcribe_audio(audio_path: Path, model_size: str, language: str | None, word_timestamps: bool) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, compute_type="int8")
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        word_timestamps=word_timestamps,
        beam_size=5,
    )

    segments: list[dict[str, Any]] = []
    transcript_parts: list[str] = []

    for idx, seg in enumerate(segments_iter):
        text = seg.text.strip()
        if text:
            transcript_parts.append(text)

        words = []
        if word_timestamps and seg.words:
            for w in seg.words:
                words.append(
                    {
                        "word": w.word,
                        "start": float(w.start) if w.start is not None else None,
                        "end": float(w.end) if w.end is not None else None,
                    }
                )

        segments.append(
            {
                "id": idx,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
                "speaker": None,
                "words": words,
            }
        )

    return {
        "language": getattr(info, "language", None),
        "segments": segments,
        "transcript": " ".join(transcript_parts).strip(),
    }


def platform_name(extractor_key: str) -> str:
    key = (extractor_key or "unknown").lower()
    if "youtube" in key:
        return "youtube"
    if "twitter" in key or key == "x":
        return "x"
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe a media URL into timestamped JSON")
    parser.add_argument("url", nargs="?", default="")
    parser.add_argument("--language", default=None)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--no-word-timestamps", action="store_true")
    parser.add_argument("--persist-media", action="store_true")
    parser.add_argument("--cookies-from-browser", default=None)
    parser.add_argument(
        "--allow-personal-cookies",
        action="store_true",
        help="Explicit opt-in for browser cookies (may prompt keychain access).",
    )
    parser.add_argument("--doctor", action="store_true")

    args = parser.parse_args()

    if args.doctor:
        return run_doctor()

    if not args.url:
        parser.error("url is required unless --doctor is used")

    if args.cookies_from_browser and not args.allow_personal_cookies:
        parser.error(
            "--cookies-from-browser requires --allow-personal-cookies; default mode avoids personal browser/keychain data"
        )

    source_path: Path | None = None
    audio_path: Path | None = None
    try:
        info = extract_info(args.url, cookies_from_browser=args.cookies_from_browser)
        source_path = download_audio_source(args.url, cookies_from_browser=args.cookies_from_browser)
        audio_path = normalize_to_wav16k(source_path)
        asr = transcribe_audio(
            audio_path,
            model_size=args.model_size,
            language=args.language,
            word_timestamps=not args.no_word_timestamps,
        )

        payload = {
            "source_url": info.get("webpage_url") or args.url,
            "platform": platform_name(info.get("extractor_key") or "unknown"),
            "title": info.get("title") or "untitled",
            "duration_sec": info.get("duration"),
            "language": asr.get("language"),
            "transcript": asr.get("transcript", ""),
            "segments": asr.get("segments", []),
            "metadata": {
                "extractor_key": info.get("extractor_key"),
                "video_id": info.get("id"),
                "model_size": args.model_size,
            },
            "status": "completed",
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"transcription failed at ffmpeg stage: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"transcription failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if not args.persist_media:
            if source_path and source_path.exists():
                shutil.rmtree(source_path.parent, ignore_errors=True)
            if audio_path and audio_path.exists():
                shutil.rmtree(audio_path.parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
