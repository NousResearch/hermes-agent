from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from scripts.fetch_transcript import main, parse_languages, run
from utils.provider import TranscriptProviderError, fetch_transcript, list_transcripts


class FakeFetchedTranscript:
    def to_raw_data(self) -> list[dict[str, float | str]]:
        return [
            {"start": 0.0, "duration": 1.5, "text": "  Hello   world  "},
            {"start": 1.5, "duration": 2.0, "text": "Next topic"},
        ]


class NoTranscriptFound(Exception):
    pass


class FakeTrack:
    def __init__(
        self,
        language: str = "Turkish",
        language_code: str = "tr",
        *,
        is_generated: bool = True,
        is_translatable: bool = True,
    ) -> None:
        self.language = language
        self.language_code = language_code
        self.is_generated = is_generated
        self.is_translatable = is_translatable

    def fetch(self) -> FakeFetchedTranscript:
        return FakeFetchedTranscript()

    def translate(self, language_code: str) -> FakeTrack:
        return FakeTrack(language_code, language_code, is_generated=self.is_generated)


class FakeTranscriptList:
    def __init__(self, tracks: list[FakeTrack] | None = None) -> None:
        self.tracks = tracks or [FakeTrack()]

    def __iter__(self):
        return iter(self.tracks)

    def find_transcript(self, languages: list[str] | tuple[str, ...]) -> FakeTrack:
        for language in languages:
            for track in self.tracks:
                if track.language_code == language:
                    return track
        raise NoTranscriptFound()


class FakeApi:
    def list(self, video_id: str) -> FakeTranscriptList:
        return FakeTranscriptList()


class TranscriptsDisabled(Exception):
    pass


class DisabledApi:
    def list(self, video_id: str) -> None:
        raise TranscriptsDisabled()


class ProviderTests(unittest.TestCase):
    def test_fetches_and_normalizes_mocked_provider_output(self) -> None:
        transcript = fetch_transcript("abcdefghijk", api_factory=FakeApi)
        self.assertEqual(
            transcript,
            [
                {"start": 0.0, "end": 1.5, "text": "Hello world"},
                {"start": 1.5, "end": 3.5, "text": "Next topic"},
            ],
        )

    def test_disabled_transcript_returns_stable_provider_error(self) -> None:
        with self.assertRaises(TranscriptProviderError) as context:
            fetch_transcript("abcdefghijk", api_factory=DisabledApi)
        self.assertEqual(
            context.exception.as_dict(),
            {
                "error": "TranscriptUnavailable",
                "message": "No captions or transcript could be retrieved for this video.",
                "detail": "TranscriptsDisabled",
            },
        )

    def test_translates_available_track_to_first_requested_language(self) -> None:
        class EnglishApi:
            def list(self, video_id: str) -> FakeTranscriptList:
                return FakeTranscriptList([FakeTrack("English", "en")])

        transcript = fetch_transcript("abcdefghijk", languages=["tr"], api_factory=EnglishApi)
        self.assertEqual(transcript[0]["text"], "Hello world")

    def test_lists_mocked_transcript_tracks(self) -> None:
        self.assertEqual(
            list_transcripts("abcdefghijk", api_factory=FakeApi),
            {
                "video_id": "abcdefghijk",
                "transcripts": [
                    {
                        "language": "Turkish",
                        "language_code": "tr",
                        "is_generated": True,
                        "is_translatable": True,
                    }
                ],
            },
        )

    def test_list_failure_returns_stable_provider_error_with_detail(self) -> None:
        with self.assertRaises(TranscriptProviderError) as context:
            list_transcripts("abcdefghijk", api_factory=DisabledApi)
        self.assertEqual(
            context.exception.as_dict(),
            {
                "error": "TranscriptListUnavailable",
                "message": "Could not list available transcripts for this video.",
                "detail": "TranscriptsDisabled",
            },
        )

    def test_cookie_option_is_rejected_before_provider_construction(self) -> None:
        provider_constructed = False

        def api_factory() -> FakeApi:
            nonlocal provider_constructed
            provider_constructed = True
            return FakeApi()

        with self.assertRaises(TranscriptProviderError) as context:
            fetch_transcript("abcdefghijk", cookies="private-cookies.txt", api_factory=api_factory)
        self.assertFalse(provider_constructed)
        self.assertEqual(context.exception.error, "CookiesUnsupported")

    def test_invalid_video_id_is_rejected_before_provider_construction(self) -> None:
        provider_constructed = False

        def api_factory() -> FakeApi:
            nonlocal provider_constructed
            provider_constructed = True
            return FakeApi()

        with self.assertRaises(TranscriptProviderError) as context:
            fetch_transcript("too-short", api_factory=api_factory)

        self.assertFalse(provider_constructed)
        self.assertEqual(
            context.exception.as_dict(),
            {
                "error": "InvalidYouTubeURL",
                "message": "The provided input is not a valid YouTube URL or video ID.",
            },
        )

    def test_runner_rejects_invalid_youtube_url_without_calling_provider(self) -> None:
        def unexpected_provider(video_id: str, **kwargs: object) -> list[dict[str, float | str]]:
            self.fail("provider must not be called for invalid input")

        self.assertEqual(
            run("https://example.com/video", provider=unexpected_provider),
            {
                "error": "InvalidYouTubeURL",
                "message": "The provided input is not a valid YouTube URL or video ID.",
            },
        )

    def test_runner_rejects_invalid_video_id_without_calling_provider(self) -> None:
        def unexpected_provider(video_id: str, **kwargs: object) -> list[dict[str, float | str]]:
            self.fail("provider must not be called for invalid input")

        self.assertEqual(
            run("invalid-url", provider=unexpected_provider),
            {
                "error": "InvalidYouTubeURL",
                "message": "The provided input is not a valid YouTube URL or video ID.",
            },
        )

    def test_valid_youtube_url_reaches_provider_failure_path(self) -> None:
        def unavailable_provider(video_id: str, **kwargs: object) -> list[dict[str, float | str]]:
            self.assertEqual(video_id, "dQw4w9WgXcQ")
            raise TranscriptProviderError("TranscriptUnavailable", "Unavailable")

        self.assertEqual(
            run("https://www.youtube.com/watch?v=dQw4w9WgXcQ", provider=unavailable_provider),
            {"error": "TranscriptUnavailable", "message": "Unavailable"},
        )

    def test_runner_lists_transcripts_with_mocked_provider(self) -> None:
        def listing_provider(video_id: str, **kwargs: object) -> dict[str, object]:
            return {"video_id": video_id, "transcripts": []}

        self.assertEqual(
            run("abcdefghijk", list_only=True, list_provider=listing_provider),
            {"video_id": "abcdefghijk", "transcripts": []},
        )

    def test_languages_cli_argument_parsing(self) -> None:
        self.assertEqual(parse_languages("tr,en"), ["tr", "en"])
        self.assertEqual(parse_languages(" de, en ,"), ["de", "en"])
        self.assertEqual(parse_languages(None), ["tr", "en"])

        with patch("scripts.fetch_transcript.run", return_value=[]) as mocked_run, redirect_stdout(StringIO()):
            self.assertEqual(main(["abcdefghijk", "--languages", "tr,en"]), 0)
        mocked_run.assert_called_once_with(
            "abcdefghijk",
            languages=["tr", "en"],
            list_only=False,
            cookies=None,
            cookies_from_browser=None,
        )


if __name__ == "__main__":
    unittest.main()
