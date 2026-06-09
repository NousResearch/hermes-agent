from __future__ import annotations

import sys
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from scripts.fetch_transcript import run
from utils.provider import TranscriptProviderError, fetch_transcript


class FakeFetchedTranscript:
    def to_raw_data(self) -> list[dict[str, float | str]]:
        return [
            {"start": 0.0, "duration": 1.5, "text": "  Hello   world  "},
            {"start": 1.5, "duration": 2.0, "text": "Next topic"},
        ]


class FakeApi:
    def fetch(self, video_id: str, languages: list[str]) -> FakeFetchedTranscript:
        return FakeFetchedTranscript()


class TranscriptsDisabled(Exception):
    pass


class DisabledApi:
    def fetch(self, video_id: str, languages: list[str]) -> None:
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
            },
        )

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


if __name__ == "__main__":
    unittest.main()
