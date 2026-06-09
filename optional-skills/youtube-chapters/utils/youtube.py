"""YouTube URL and video ID parsing."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
VIDEO_ID_IN_TEXT_RE = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")
INVALID_VIDEO_ID_PLACEHOLDERS = {"invalid-url"}
SUPPORTED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


def normalize_video_id(candidate: str) -> str:
    """Return a valid YouTube video ID or raise ValueError."""
    value = candidate.strip()
    if not VIDEO_ID_RE.fullmatch(value) or value.lower() in INVALID_VIDEO_ID_PLACEHOLDERS:
        raise ValueError("YouTube video IDs must contain exactly 11 URL-safe characters")
    return value


def extract_video_id(user_input: str) -> str:
    """Extract a YouTube video ID from a URL, raw ID, or surrounding text."""
    value = user_input.strip()
    if VIDEO_ID_RE.fullmatch(value):
        return normalize_video_id(value)

    for token in value.split():
        candidate = _extract_from_url(token.strip("<>()[]{}.,"))
        if candidate:
            return candidate

    match = VIDEO_ID_IN_TEXT_RE.search(value)
    if match:
        return normalize_video_id(match.group(1))
    raise ValueError("No valid YouTube URL or video ID found")


def _extract_from_url(value: str) -> str | None:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    if parsed.hostname not in SUPPORTED_HOSTS:
        return None

    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/", 1)[0]
    elif parsed.path == "/watch":
        candidate = parse_qs(parsed.query).get("v", [""])[0]
    elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
        candidate = parsed.path.strip("/").split("/", 1)[1].split("/", 1)[0]
    else:
        return None

    try:
        return normalize_video_id(candidate)
    except ValueError:
        return None
