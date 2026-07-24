#!/usr/bin/env python3
"""
Fetch a transcript for a live-streaming clip or VOD (Twitch, Kick, Rumble, ...) and
output it as structured JSON.

Live-streaming platforms are inconsistent about captions: Twitch and Kick serve no
caption track at all, while some Rumble videos expose an auto-generated one. So this
script:
  1. reads a served caption track directly when the source exposes one (cheap — no
     download, no transcription), otherwise
  2. downloads the audio with yt-dlp and transcribes it with the shared
     ``transcribe_audio`` tool (faster-whisper / Groq / OpenAI, whichever the
     environment is configured for).

Usage:
    python fetch_transcript.py <url> [--text-only] [--keep-audio]

Output (JSON):
    {
        "url": "...",
        "extractor": "twitch",
        "uploader": "...",
        "title": "...",
        "duration": 40,
        "provider": "served-captions" | "openai" | ...,
        "full_text": "complete transcript as plain text"
    }

Dependencies: yt-dlp (pip), ffmpeg (on PATH, only needed for the transcribe path).
Requires Python 3.10+ — older interpreters silently fail Twitch's GraphQL, so run
inside the Hermes venv.
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


def _find_hermes_root(start):
    """Walk up from this script to the repo root (the directory holding ``tools/``)."""
    for parent in [start, *start.parents]:
        if (parent / "tools" / "transcription_tools.py").exists():
            return parent
    return None


def _find_ffmpeg():
    """Locate ffmpeg's directory even when it isn't on PATH — non-interactive shells
    routinely omit /opt/homebrew/bin or /usr/local/bin. Returns the dir, or None."""
    found = shutil.which("ffmpeg")
    if found:
        return os.path.dirname(found)
    for d in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/snap/bin"):
        if os.path.exists(os.path.join(d, "ffmpeg")):
            return d
    return None


def _captions_to_text(path):
    """Flatten a .vtt/.srt caption file to plain text: drop the WEBVTT header, cue
    numbers, timestamp lines, and inline timing tags, and collapse the consecutive
    duplicate lines that auto-generated captions emit."""
    lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if not s or s == "WEBVTT" or "-->" in s or s.isdigit():
                continue
            if s.startswith(("NOTE", "Kind:", "Language:")):
                continue
            s = re.sub(r"<[^>]+>", "", s)  # strip <00:00:00.000>-style inline tags
            if s:
                lines.append(s)
    out = []
    for ln in lines:
        if not out or out[-1] != ln:
            out.append(ln)
    return " ".join(out).strip()


def fetch_served_captions(url, dest_dir):
    """If the source already exposes a caption/subtitle track, fetch and return
    (info, text) — skipping the whole audio download + transcription pass. Returns
    (info, None) when there's no usable track (Twitch, Kick, ...), so the caller
    falls back to download + transcribe. Prefers a manual track over auto-generated,
    and English over other languages."""
    try:
        import yt_dlp
    except ImportError:
        return None, None

    probe = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(probe) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None, None

    subs = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    def _pick(tracks):
        if not tracks:
            return None
        for lang in tracks:
            if lang.lower().startswith("en"):
                return lang
        return next(iter(tracks))

    manual = _pick(subs)
    lang = manual or _pick(auto)
    if not lang:
        return info, None

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": bool(manual),
        "writeautomaticsub": not bool(manual),
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt/srt/best",
        "outtmpl": os.path.join(dest_dir, "caption.%(ext)s"),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception:
        return info, None

    found = sorted(Path(dest_dir).glob("caption*.vtt")) + sorted(Path(dest_dir).glob("caption*.srt"))
    if not found:
        found = sorted(Path(dest_dir).glob("*.vtt")) + sorted(Path(dest_dir).glob("*.srt"))
    if not found:
        return info, None

    text = _captions_to_text(str(found[0]))
    return info, (text or None)


def download_audio(url, dest_dir):
    """Download a single clip/VOD's audio. Returns (info, audio_path, error)."""
    try:
        import yt_dlp
    except ImportError:
        return None, "", "yt-dlp not installed. Run: pip install yt-dlp"

    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(dest_dir, "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "5"}
        ],
    }
    ffmpeg_dir = _find_ffmpeg()
    if ffmpeg_dir:
        opts["ffmpeg_location"] = ffmpeg_dir   # so audio extraction works off-PATH

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "no video could be found" in low:
            # X/Twitter (and similar) image/text/link-only posts: valid URL, no media.
            return None, "", "no playable video in this post (image, text, or link-only - nothing to transcribe)"
        if "ffmpeg" in low or "ffprobe" in low:
            msg = ("ffmpeg/ffprobe not found. Install ffmpeg "
                   "(brew install ffmpeg / apt install ffmpeg).")
        return None, "", "download failed: %s" % msg

    audio_path = os.path.join(dest_dir, "audio.mp3")
    if not os.path.exists(audio_path):
        candidates = sorted(Path(dest_dir).glob("audio.*"))
        audio_path = str(candidates[0]) if candidates else ""
    return info, audio_path, ""


def transcribe(audio_path):
    """Transcribe via the shared transcription tool. Returns its result dict."""
    root = _find_hermes_root(Path(__file__).resolve())
    if root and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from tools.transcription_tools import transcribe_audio
    except ImportError as e:
        return {"success": False, "error": "could not import transcribe_audio: %s" % e}
    return transcribe_audio(audio_path)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a Twitch/Kick/Rumble/X clip or VOD as JSON")
    parser.add_argument("url", help="Clip or VOD URL")
    parser.add_argument("--text-only", action="store_true",
                        help="Print plain transcript text instead of JSON")
    parser.add_argument("--keep-audio", action="store_true",
                        help="Keep the downloaded audio file")
    args = parser.parse_args()

    work_dir = tempfile.mkdtemp(prefix="streaming-content-")
    try:
        # 1) If the source already serves a caption track (e.g. some Rumble videos),
        #    read it directly — no audio download, no transcription.
        info, cap_text = fetch_served_captions(args.url, work_dir)
        if cap_text:
            text = cap_text
            provider = "served-captions"
        else:
            # 2) No caption track (Twitch, Kick, ...) — download the audio + transcribe.
            dl_info, audio_path, err = download_audio(args.url, work_dir)
            info = info or dl_info
            if err or not audio_path:
                print(json.dumps({"error": err or "audio download failed"}))
                sys.exit(1)

            result = transcribe(audio_path)
            if not result.get("success"):
                print(json.dumps({"error": result.get("error", "transcription failed")}))
                sys.exit(1)

            text = result.get("transcript") or result.get("text") or ""
            provider = result.get("provider")

        info = info or {}
        text = (text or "").strip()

        if args.text_only:
            print(text)
            return

        print(json.dumps({
            "url": args.url,
            "extractor": info.get("extractor_key") or info.get("extractor"),
            "uploader": info.get("uploader") or info.get("channel"),
            "title": info.get("title"),
            "duration": info.get("duration"),
            "provider": provider,
            "full_text": text,
        }, ensure_ascii=False, indent=2))
    finally:
        if not args.keep_audio:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
