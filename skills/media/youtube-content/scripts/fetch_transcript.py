#!/usr/bin/env python3
"""
Fetch a YouTube video transcript and output it as structured JSON.

Usage:
    python fetch_transcript.py <url_or_video_id> [--language en,tr] [--timestamps]

Output (JSON):
    {
        "video_id": "...",
        "language": "en",
        "segments": [{"text": "...", "start": 0.0, "duration": 2.5}, ...],
        "full_text": "complete transcript as plain text",
        "timestamped_text": "00:00 first line\n00:05 second line\n..."
    }

Install dependency:  pip install youtube-transcript-api
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


SUPADATA_BASE_URL = os.getenv("SUPADATA_BASE_URL", "https://api.supadata.ai/v1").rstrip("/")
SUPADATA_TIMEOUT_SECONDS = 120
SUPADATA_JOB_TIMEOUT_SECONDS = 180
SUPADATA_JOB_POLL_SECONDS = 3


def extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character video ID from various YouTube URL formats."""
    url_or_id = url_or_id.strip()
    patterns = [
        r'(?:v=|youtu\.be/|shorts/|embed/|live/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fetch_transcript(video_id: str, languages: list = None):
    """Fetch transcript segments from YouTube.

    Returns a list of dicts with 'text', 'start', and 'duration' keys.
    Compatible with youtube-transcript-api v1.x.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("Error: youtube-transcript-api not installed. Run: pip install youtube-transcript-api",
              file=sys.stderr)
        sys.exit(1)

    api = YouTubeTranscriptApi()
    if languages:
        result = api.fetch(video_id, languages=languages)
    else:
        result = api.fetch(video_id)

    # v1.x returns FetchedTranscriptSnippet objects; normalize to dicts
    return [
        {"text": seg.text, "start": seg.start, "duration": seg.duration}
        for seg in result
    ]


def youtube_url_for(video_id_or_url: str) -> str:
    """Return a URL suitable for Supadata from a YouTube URL or raw video ID."""
    video_id = extract_video_id(video_id_or_url)
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", video_id):
        return f"https://www.youtube.com/watch?v={video_id}"
    return video_id_or_url


def _supadata_get(path: str, params: dict | None = None) -> dict:
    api_key = os.getenv("SUPADATA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SUPADATA_API_KEY is not set")

    query = urllib.parse.urlencode(params or {})
    url = f"{SUPADATA_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"

    # Supadata is behind Cloudflare and currently rejects Python's default
    # urllib User-Agent with HTTP 403 / error code 1010. Use a normal CLI UA.
    request = urllib.request.Request(
        url,
        headers={
            "x-api-key": api_key,
            "User-Agent": "curl/8.5.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=SUPADATA_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(detail)
            detail = data.get("error") or data.get("message") or detail
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"Supadata API error HTTP {exc.code}: {detail}") from exc


def _normalize_supadata_segments(data: dict) -> list[dict]:
    content = data.get("content", [])
    if isinstance(content, str):
        return [{"text": content.strip(), "start": 0.0, "duration": 0.0}]

    segments = []
    for item in content:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        offset_ms = float(item.get("offset", 0) or 0)
        duration_ms = float(item.get("duration", 0) or 0)
        segments.append({
            "text": text,
            "start": offset_ms / 1000,
            "duration": duration_ms / 1000,
        })
    return segments


def _poll_supadata_job(job_id: str) -> dict:
    deadline = time.monotonic() + SUPADATA_JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        data = _supadata_get(f"/transcript/{urllib.parse.quote(job_id)}")
        status = str(data.get("status", "")).lower()
        if data.get("content"):
            return data
        if status in {"completed", "succeeded", "success"}:
            return data
        if status in {"failed", "error"}:
            raise RuntimeError(f"Supadata transcript job failed: {data}")
        time.sleep(SUPADATA_JOB_POLL_SECONDS)
    raise TimeoutError(f"Supadata transcript job timed out: {job_id}")


def fetch_supadata_transcript(url_or_id: str, languages: list = None):
    """Fetch transcript segments from Supadata's universal transcript API."""
    params = {
        "url": youtube_url_for(url_or_id),
        "mode": os.getenv("SUPADATA_TRANSCRIPT_MODE", "auto"),
    }
    if languages:
        params["lang"] = languages[0]

    data = _supadata_get("/transcript", params)
    job_id = data.get("jobId") or data.get("job_id")
    if job_id:
        data = _poll_supadata_job(str(job_id))

    segments = _normalize_supadata_segments(data)
    if not segments:
        raise RuntimeError("Supadata returned an empty transcript")
    return segments


def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube transcript as JSON")
    parser.add_argument("url", help="YouTube URL or video ID")
    parser.add_argument("--language", "-l", default=None,
                        help="Comma-separated language codes (e.g. en,tr). Default: auto")
    parser.add_argument("--timestamps", "-t", action="store_true",
                        help="Include timestamped text in output")
    parser.add_argument("--text-only", action="store_true",
                        help="Output plain text instead of JSON")
    parser.add_argument("--provider", choices=("auto", "youtube", "supadata"),
                        default="auto",
                        help="Transcript provider. Default: YouTube first, then Supadata if SUPADATA_API_KEY is set")
    args = parser.parse_args()

    video_id = extract_video_id(args.url)
    languages = [l.strip() for l in args.language.split(",")] if args.language else None

    try:
        if args.provider == "supadata":
            segments = fetch_supadata_transcript(args.url, languages)
        else:
            try:
                segments = fetch_transcript(video_id, languages)
            except Exception:
                if args.provider != "auto" or not os.getenv("SUPADATA_API_KEY"):
                    raise
                segments = fetch_supadata_transcript(args.url, languages)
    except Exception as e:
        error_msg = str(e)
        if "disabled" in error_msg.lower():
            print(json.dumps({"error": "Transcripts are disabled for this video."}))
        elif "no transcript" in error_msg.lower():
            print(json.dumps({"error": f"No transcript found. Try specifying a language with --language."}))
        else:
            print(json.dumps({"error": error_msg}))
        sys.exit(1)

    full_text = " ".join(seg["text"] for seg in segments)
    timestamped = "\n".join(
        f"{format_timestamp(seg['start'])} {seg['text']}" for seg in segments
    )

    if args.text_only:
        print(timestamped if args.timestamps else full_text)
        return

    result = {
        "video_id": video_id,
        "segment_count": len(segments),
        "duration": format_timestamp(segments[-1]["start"] + segments[-1]["duration"]) if segments else "0:00",
        "full_text": full_text,
    }
    if args.timestamps:
        result["timestamped_text"] = timestamped

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
