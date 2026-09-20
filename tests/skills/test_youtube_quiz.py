"""Tests for optional-skills/productivity/memento-flashcards/scripts/youtube_quiz.py"""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "productivity" / "memento-flashcards" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import youtube_quiz


def _run(capsys, argv: list[str]) -> dict:
    """Run main() with given argv and return parsed JSON output."""
    with mock.patch("sys.argv", ["youtube_quiz"] + argv):
        youtube_quiz.main()
    captured = capsys.readouterr()
    return json.loads(captured.out)


class TestNormalizeSegments:
    def test_basic(self):
        segments = [{"text": "hello "}, {"text": " world"}]
        assert youtube_quiz._normalize_segments(segments) == "hello world"


    def test_whitespace_only(self):
        assert youtube_quiz._normalize_segments([{"text": "   "}, {"text": "  "}]) == ""

    def test_collapses_multiple_spaces(self):
        segments = [{"text": "a   b"}, {"text": "c  d"}]
        assert youtube_quiz._normalize_segments(segments) == "a b c d"




class TestFetchWithMockedAPI:
    def _make_mock_module(self, segments=None, raise_exc=None):
        """Create a mock youtube_transcript_api module."""
        mock_module = mock.MagicMock()

        mock_api_instance = mock.MagicMock()
        mock_module.YouTubeTranscriptApi.return_value = mock_api_instance

        if raise_exc:
            mock_api_instance.fetch.side_effect = raise_exc
        else:
            raw_data = segments or [{"text": "Hello world"}]
            result = mock.MagicMock()
            result.to_raw_data.return_value = raw_data
            mock_api_instance.fetch.return_value = result

        return mock_module

    def test_successful_fetch(self, capsys):
        mock_mod = self._make_mock_module(
            segments=[{"text": "This is a test"}, {"text": "transcript segment"}]
        )
        with mock.patch.dict("sys.modules", {"youtube_transcript_api": mock_mod}):
            result = _run(capsys, ["fetch", "abc123"])

        assert result["ok"] is True
        assert result["video_id"] == "abc123"
        assert "This is a test" in result["transcript"]
        assert "transcript segment" in result["transcript"]

    def test_fetch_error(self, capsys):
        mock_mod = self._make_mock_module(raise_exc=Exception("Video unavailable"))
        with mock.patch.dict("sys.modules", {"youtube_transcript_api": mock_mod}):
            with pytest.raises(SystemExit):
                _run(capsys, ["fetch", "bad_id"])

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["ok"] is False
        assert result["error"] == "transcript_unavailable"


