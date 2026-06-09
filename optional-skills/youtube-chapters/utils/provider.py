"""Optional live transcript provider backed by youtube-transcript-api."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .transcript import normalize_segments
from .youtube import normalize_video_id

DEFAULT_LANGUAGES = ("tr", "en")
UNAVAILABLE_EXCEPTIONS = {
    "NoTranscriptFound",
    "TranscriptsDisabled",
    "VideoUnavailable",
}


class TranscriptProviderError(Exception):
    """A stable provider failure suitable for structured CLI output."""

    def __init__(self, error: str, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.error = error
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        result = {"error": self.error, "message": self.message}
        if self.detail:
            result["detail"] = self.detail
        return result


def fetch_transcript(
    video_id: str,
    *,
    languages: Sequence[str] | None = None,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
    api_factory: Callable[[], Any] | None = None,
) -> list[dict[str, float | str]]:
    """Fetch and normalize captions for a YouTube video ID."""
    video_id = _validate_video_id(video_id)
    api = _create_api(api_factory, cookies=cookies, cookies_from_browser=cookies_from_browser)
    requested_languages = tuple(languages or DEFAULT_LANGUAGES)

    try:
        transcripts = api.list(video_id)
        track = _select_track(transcripts, requested_languages)
        fetched = track.fetch()
        raw_segments = _to_raw_segments(fetched)
        normalized = normalize_segments(raw_segments)
    except TranscriptProviderError:
        raise
    except Exception as exc:
        if type(exc).__name__ in UNAVAILABLE_EXCEPTIONS:
            raise TranscriptProviderError(
                "TranscriptUnavailable",
                "No captions or transcript could be retrieved for this video.",
                type(exc).__name__,
            ) from exc
        raise TranscriptProviderError(
            "TranscriptFetchFailed",
            "Transcript retrieval failed. Try again later or provide a timestamped transcript.",
            type(exc).__name__,
        ) from exc

    if not normalized:
        raise TranscriptProviderError(
            "TranscriptUnavailable",
            "No captions or transcript could be retrieved for this video.",
            "EmptyTranscript",
        )
    return normalized


def list_transcripts(
    video_id: str,
    *,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
    api_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """List available transcript tracks for a YouTube video ID."""
    video_id = _validate_video_id(video_id)
    api = _create_api(api_factory, cookies=cookies, cookies_from_browser=cookies_from_browser)
    try:
        transcripts = api.list(video_id)
        tracks = [
            {
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "is_translatable": transcript.is_translatable,
            }
            for transcript in transcripts
        ]
    except TranscriptProviderError:
        raise
    except Exception as exc:
        raise TranscriptProviderError(
            "TranscriptListUnavailable",
            "Could not list available transcripts for this video.",
            type(exc).__name__,
        ) from exc
    return {"video_id": video_id, "transcripts": tracks}


def _validate_video_id(video_id: str) -> str:
    try:
        return normalize_video_id(video_id)
    except ValueError as exc:
        raise TranscriptProviderError(
            "InvalidYouTubeURL",
            "The provided input is not a valid YouTube URL or video ID.",
        ) from exc


def _create_api(
    api_factory: Callable[[], Any] | None,
    *,
    cookies: str | None,
    cookies_from_browser: str | None,
) -> Any:
    if cookies or cookies_from_browser:
        raise TranscriptProviderError(
            "CookiesUnsupported",
            "The installed transcript provider does not support cookie-based transcript retrieval.",
        )
    if api_factory is None:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as exc:
            raise TranscriptProviderError(
                "DependencyUnavailable",
                "youtube-transcript-api is not installed. Install the skill requirements first.",
            ) from exc
        api_factory = YouTubeTranscriptApi
    return api_factory()


def _select_track(transcripts: Any, languages: Sequence[str]) -> Any:
    try:
        return transcripts.find_transcript(languages)
    except Exception as exact_error:
        for transcript in transcripts:
            if transcript.is_translatable:
                try:
                    return transcript.translate(languages[0])
                except Exception:
                    continue
        raise exact_error


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
