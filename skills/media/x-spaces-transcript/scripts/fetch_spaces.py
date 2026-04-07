#!/usr/bin/env python3
"""
X Spaces → Transcript pipeline.

Downloads audio from an X/Twitter Spaces URL using yt-dlp,
then transcribes it using OpenAI Whisper.

Usage:
    python3 fetch_spaces.py "https://x.com/i/spaces/1yxBeMYdqgnJN"
    python3 fetch_spaces.py "https://x.com/i/spaces/1yxBeMYdqgnJN" --model small
    python3 fetch_spaces.py "https://x.com/i/spaces/1yxBeMYdqgnJN" --text-only
    python3 fetch_spaces.py "https://x.com/i/spaces/1yxBeMYdqgnJN" --output /path/to/transcript.txt
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse


# Allowed domains for X Spaces URLs
ALLOWED_DOMAINS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}

# Regex for valid X Spaces URL path
SPACES_PATH_RE = re.compile(r"^/i/spaces/[A-Za-z0-9]+$")

# Regex to strip Whisper timestamps
TIMESTAMP_RE = re.compile(r"\[\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}\.\d{3}\]\s*")

# Regex to strip control characters from metadata
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def log(msg: str):
    print(f"[x-spaces] {msg}", file=sys.stderr)


def validate_url(url: str) -> str:
    """Validate that the URL is a proper X Spaces link. Returns cleaned URL."""
    parsed = urlparse(url)

    if parsed.scheme not in ("https", "http"):
        print(f"ERROR: Only https:// URLs are supported, got: {parsed.scheme}://", file=sys.stderr)
        sys.exit(1)

    if parsed.hostname not in ALLOWED_DOMAINS:
        print(f"ERROR: URL must be from x.com or twitter.com, got: {parsed.hostname}", file=sys.stderr)
        sys.exit(1)

    # Strip query params and fragments for clean matching
    path = parsed.path.rstrip("/")
    if not SPACES_PATH_RE.match(path):
        print(f"ERROR: Not a valid X Spaces URL. Expected: https://x.com/i/spaces/<ID>", file=sys.stderr)
        print(f"  Got path: {path}", file=sys.stderr)
        sys.exit(1)

    # Return URL without tracking params
    return f"https://x.com{path}"


def sanitize_metadata_value(value: str) -> str:
    """Strip control characters and limit length from metadata values."""
    if not value:
        return ""
    cleaned = CONTROL_CHAR_RE.sub("", str(value))
    # Limit to 200 chars to prevent absurd output
    return cleaned[:200].strip()


def download_audio(url: str, output_dir: str) -> str:
    """Download X Spaces audio via yt-dlp. Returns path to mp3 file."""
    output_template = os.path.join(output_dir, "spaces_audio.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "-o", output_template,
        url,
    ]

    log(f"Downloading audio from: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    # yt-dlp may report error about .m4a.part but the .mp3 still exists
    mp3_path = os.path.join(output_dir, "spaces_audio.mp3")
    if not os.path.exists(mp3_path):
        # Check for any audio file with that prefix (must be in our output_dir only)
        try:
            for f in os.listdir(output_dir):
                candidate = os.path.join(output_dir, f)
                if (os.path.isfile(candidate)
                        and f.startswith("spaces_audio.")
                        and f.endswith((".mp3", ".m4a", ".ogg"))):
                    mp3_path = candidate
                    break
        except OSError:
            pass

    if not os.path.exists(mp3_path):
        print("ERROR: Audio file not found after download.", file=sys.stderr)
        print(f"yt-dlp stderr:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(mp3_path) / (1024 * 1024)
    log(f"Audio downloaded: {mp3_path} ({size_mb:.1f} MB)")
    return mp3_path


def extract_metadata(url: str) -> dict:
    """Extract space metadata using yt-dlp's JSON dump."""
    cmd = [
        "yt-dlp",
        "-j",  # dump JSON
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return {
                "title": sanitize_metadata_value(data.get("title", "Unknown")),
                "duration": data.get("duration"),
                "duration_string": sanitize_metadata_value(data.get("duration_string", "")),
                "uploader": sanitize_metadata_value(data.get("uploader", "")),
                "url": url,
            }
    except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError):
        pass
    return {"title": "Unknown", "url": url}


def transcribe(audio_path: str, model: str = "base", language: str = "en",
               output_dir: str = "/tmp") -> str:
    """Transcribe audio using Whisper. Returns path to .txt file."""
    cmd = [
        "whisper",
        audio_path,
        "--model", model,
        "--language", language,
        "--output_format", "txt",
        "--output_dir", output_dir,
    ]

    log(f"Transcribing with Whisper (model={model}, lang={language})...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print("ERROR: Whisper failed.", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Whisper outputs <filename_without_ext>.txt — look only for exact match
    base = Path(audio_path).stem
    txt_path = os.path.join(output_dir, f"{base}.txt")

    if not os.path.exists(txt_path):
        print(f"ERROR: Expected transcript at {txt_path} not found.", file=sys.stderr)
        sys.exit(1)

    log(f"Transcript saved: {txt_path}")
    return txt_path


def clean_transcript(text: str) -> str:
    """Remove Whisper timestamp lines for clean output."""
    cleaned = TIMESTAMP_RE.sub("", text)
    # Collapse multiple blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def main():
    parser = argparse.ArgumentParser(description="Extract and transcribe X Spaces recordings")
    parser.add_argument("url", help="X Spaces URL (https://x.com/i/spaces/...)")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument("--text-only", action="store_true", help="Output clean text without metadata header")
    parser.add_argument("--audio-only", action="store_true", help="Only download audio, skip transcription")
    parser.add_argument("--keep-audio", action="store_true", help="Keep downloaded audio file")
    parser.add_argument("--work-dir", help="Working directory for temp files")

    args = parser.parse_args()

    # Validate URL before doing anything
    clean_url = validate_url(args.url)

    # Set up work directory — use tempfile if not specified
    work_dir = args.work_dir
    is_temp_dir = work_dir is None
    if is_temp_dir:
        work_dir = tempfile.mkdtemp(prefix="xspaces_")
    else:
        os.makedirs(work_dir, exist_ok=True)

    try:
        # Step 1: Extract metadata
        meta = extract_metadata(clean_url)

        # Step 2: Download audio
        audio_path = download_audio(clean_url, work_dir)

        if args.audio_only:
            print(audio_path)
            return

        # Step 3: Transcribe
        txt_path = transcribe(audio_path, model=args.model, language=args.language, output_dir=work_dir)

        with open(txt_path, "r") as f:
            raw_text = f.read()

        transcript = clean_transcript(raw_text)

        # Step 4: Output
        if args.text_only:
            output = transcript
        else:
            header_lines = [
                f"# {meta.get('title', 'X Space')}",
                f"URL: {meta['url']}",
            ]
            if meta.get("duration_string"):
                header_lines.append(f"Duration: {meta['duration_string']}")
            if meta.get("uploader"):
                header_lines.append(f"Host: {meta['uploader']}")
            header_lines.append("")
            header_lines.append("---")
            header_lines.append("")
            output = "\n".join(header_lines) + transcript

        if args.output:
            # Validate output path is not a symlink (prevent overwrite attacks)
            output_path = Path(args.output).resolve()
            if output_path.is_symlink():
                print(f"ERROR: Output path is a symlink, refusing to write: {args.output}", file=sys.stderr)
                sys.exit(1)
            with open(output_path, "w") as f:
                f.write(output)
            log(f"Output written to: {output_path}")
        else:
            print(output)

        # Cleanup audio (unless --keep-audio)
        if not args.keep_audio:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    finally:
        # Clean up temp directory if we created it
        if is_temp_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
