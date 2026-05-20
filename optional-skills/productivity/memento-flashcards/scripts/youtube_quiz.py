#!/usr/bin/env python3
"""Fetch YouTube transcripts for Memento quiz generation.

Requires: pip install youtube-transcript-api
The quiz question *generation* is done by the agent's LLM — this script only fetches transcripts.
"""

import argparse
import json
import re
import sys


def _out(obj: object) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _normalize_segments(segments: list) -> str:
    parts = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if text:
            parts.append(text)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def cmd_fetch(args: argparse.Namespace) -> None:
    try:
        import youtube_transcript_api  # noqa: F811
    except ImportError:
        _out({
            "ok": False,
            "error": "missing_dependency",
            "message": "Run: pip install youtube-transcript-api",
        })
        sys.exit(1)

    video_id = args.video_id
    languages = ["en", "en-US", "en-GB", "en-CA", "en-AU"]

    api = youtube_transcript_api.YouTubeTranscriptApi()
    try:
        raw = api.fetch(video_id, languages=languages)
    except Exception as exc:
        error_type = type(exc).__name__
        _out({
            "ok": False,
            "error": "transcript_unavailable",
            "error_type": error_type,
            "message": f"Could not fetch transcript for {video_id}: {exc}",
        })
        sys.exit(1)

    segments = raw
    if hasattr(raw, "to_raw_data"):
        segments = raw.to_raw_data()

    text = _normalize_segments(segments)
    if not text:
        _out({
            "ok": False,
            "error": "empty_transcript",
            "message": f"Transcript for {video_id} contained no usable text.",
        })
        sys.exit(1)

    _out({
        "ok": True,
        "video_id": video_id,
        "transcript": text,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Memento YouTube transcript fetcher")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch transcript for a video")
    p_fetch.add_argument("video_id", help="YouTube video ID")

    args = parser.parse_args()
    if args.command == "fetch":
        cmd_fetch(args)


if __name__ == "__main__":
    main()
