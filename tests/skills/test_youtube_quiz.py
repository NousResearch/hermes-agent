"""Tests for optional-skills/productivity/memento-flashcards/scripts/youtube_quiz.py"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
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

    def test_empty_segments(self):
        assert youtube_quiz._normalize_segments([]) == ""

    def test_whitespace_only(self):
        assert youtube_quiz._normalize_segments([{"text": "   "}, {"text": "  "}]) == ""

    def test_collapses_multiple_spaces(self):
        segments = [{"text": "a   b"}, {"text": "c  d"}]
        assert youtube_quiz._normalize_segments(segments) == "a b c d"


class TestFetchMissingDependency:
    def test_missing_youtube_transcript_api(self, capsys, monkeypatch):
        """When youtube-transcript-api is not installed, report the error."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "youtube_transcript_api":
                raise ImportError("No module named 'youtube_transcript_api'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(SystemExit) as exc_info:
            _run(capsys, ["fetch", "test123"])

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["ok"] is False
        assert result["error"] == "missing_dependency"
        assert "pip install" in result["message"]


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

    def test_empty_transcript(self, capsys):
        mock_mod = self._make_mock_module(segments=[{"text": ""}, {"text": "   "}])
        with mock.patch.dict("sys.modules", {"youtube_transcript_api": mock_mod}):
            with pytest.raises(SystemExit):
                _run(capsys, ["fetch", "empty_vid"])

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["ok"] is False
        assert result["error"] == "empty_transcript"

    def test_segments_without_to_raw_data(self, capsys):
        """Handle plain list segments (no to_raw_data method)."""
        mock_mod = mock.MagicMock()
        mock_api = mock.MagicMock()
        mock_mod.YouTubeTranscriptApi.return_value = mock_api
        # Return a plain list (no to_raw_data attribute)
        mock_api.fetch.return_value = [{"text": "plain list"}]

        with mock.patch.dict("sys.modules", {"youtube_transcript_api": mock_mod}):
            result = _run(capsys, ["fetch", "plain123"])

        assert result["ok"] is True
        assert result["transcript"] == "plain list"
