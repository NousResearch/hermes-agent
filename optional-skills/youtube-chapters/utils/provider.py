"""Optional live transcript provider backed by youtube-transcript-api."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .transcript import normalize_segments
from .youtube import normalize_video_id

UNAVAILABLE_EXCEPTIONS = {
    "NoTranscriptFound",
    "TranscriptsDisabled",
    "VideoUnavailable",
}


class TranscriptProviderError(Exception):
    """A stable provider failure suitable for structured CLI output."""

    def __init__(self, error: str, message: str) -> None:
        super().__init__(message)
        self.error = error
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"error": self.error, "message": self.message}


def fetch_transcript(
    video_id: str,
    *,
    languages: Sequence[str] | None = None,
    api_factory: Callable[[], Any] | None = None,
) -> list[dict[str, float | str]]:
    """Fetch and normalize captions for a YouTube video ID."""
    try:
        video_id = normalize_video_id(video_id)
    except ValueError as exc:
        raise TranscriptProviderError(
            "InvalidYouTubeURL",
            "The provided input is not a valid YouTube URL or video ID.",
        ) from exc

    if api_factory is None:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as exc:
            raise TranscriptProviderError(
                "DependencyUnavailable",
                "youtube-transcript-api is not installed. Install the skill requirements first.",
            ) from exc
        api_factory = YouTubeTranscriptApi

    try:
        fetched = api_factory().fetch(video_id, languages=list(languages or ("en",)))
        raw_segments = _to_raw_segments(fetched)
        normalized = normalize_segments(raw_segments)
    except TranscriptProviderError:
        raise
    except Exception as exc:
        if type(exc).__name__ in UNAVAILABLE_EXCEPTIONS:
            raise TranscriptProviderError(
                "TranscriptUnavailable",
                "No captions or transcript could be retrieved for this video.",
            ) from exc
        raise TranscriptProviderError(
            "TranscriptFetchFailed",
            "Transcript retrieval failed. Try again later or provide a timestamped transcript.",
        ) from exc

    if not normalized:
        raise TranscriptProviderError(
            "TranscriptUnavailable",
            "No captions or transcript could be retrieved for this video.",
        )
    return normalized


def _to_raw_segments(fetched: Any) -> Iterable[Mapping[str, Any]]:
    """Convert current provider objects or raw mappings into normalization input."""
    if hasattr(fetched, "to_raw_data"):
        return fetched.to_raw_data()
    return [
        segment
        if isinstance(segment, Mapping)
        else {
            "start": segment.start,
            "duration": segment.duration,
            "text": segment.text,
        }
        for segment in fetched
    ]
