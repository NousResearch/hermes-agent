"""Tests for skills/media/youtube-content/scripts/fetch_transcript.py (issues #22243 and #36522)."""

import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "skills" / "media" / "youtube-content" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fetch_transcript


class TestExtractVideoId:
    def test_standard_watch_url(self):
        assert fetch_transcript.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert fetch_transcript.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        assert fetch_transcript.extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_with_extra_params(self):
        assert fetch_transcript.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42") == "dQw4w9WgXcQ"

    @pytest.mark.parametrize(
        "value",
        [
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/live/dQw4w9WgXcQ",
            "  dQw4w9WgXcQ  ",
        ],
    )
    def test_other_supported_inputs(self, value):
        assert fetch_transcript.extract_video_id(value) == "dQw4w9WgXcQ"


class TestFormatTimestamp:
    def test_seconds_only(self):
        assert fetch_transcript.format_timestamp(90) == "1:30"

    def test_zero(self):
        assert fetch_transcript.format_timestamp(0) == "0:00"

    def test_minutes_only(self):
        assert fetch_transcript.format_timestamp(600) == "10:00"

    def test_hours(self):
        assert fetch_transcript.format_timestamp(3661) == "1:01:01"


class TestFetchTranscript:
    def test_fetches_and_normalizes_segments_with_languages(self):
        calls = []

        class FakeApi:
            def fetch(self, video_id, **kwargs):
                calls.append((video_id, kwargs))
                return [
                    types.SimpleNamespace(text="hello", start=0.0, duration=1.5),
                    types.SimpleNamespace(text="world", start=1.5, duration=2.0),
                ]

        fake_module = types.SimpleNamespace(YouTubeTranscriptApi=FakeApi)
        with mock.patch.dict(sys.modules, {"youtube_transcript_api": fake_module}):
            result = fetch_transcript.fetch_transcript("video-id", ["en", "tr"])

        assert calls == [("video-id", {"languages": ["en", "tr"]})]
        assert result == [
            {"text": "hello", "start": 0.0, "duration": 1.5},
            {"text": "world", "start": 1.5, "duration": 2.0},
        ]

    def test_fetches_without_languages(self):
        calls = []

        class FakeApi:
            def fetch(self, video_id, **kwargs):
                calls.append((video_id, kwargs))
                return []

        fake_module = types.SimpleNamespace(YouTubeTranscriptApi=FakeApi)
        with mock.patch.dict(sys.modules, {"youtube_transcript_api": fake_module}):
            assert fetch_transcript.fetch_transcript("video-id") == []

        assert calls == [("video-id", {})]

    def test_missing_dependency_exits_with_actionable_error(self, capsys):
        with mock.patch.dict(sys.modules, {"youtube_transcript_api": None}):
            with pytest.raises(SystemExit) as exc_info:
                fetch_transcript.fetch_transcript("video-id")

        assert exc_info.value.code == 1
        assert "uv pip install youtube-transcript-api" in capsys.readouterr().err


class TestMain:
    def test_json_output_includes_duration_and_full_text(self, capsys):
        segments = [
            {"text": "hello", "start": 0.0, "duration": 1.5},
            {"text": "world", "start": 60.0, "duration": 2.0},
        ]
        with mock.patch.object(fetch_transcript, "fetch_transcript", return_value=segments) as fetch_mock:
            with mock.patch.object(sys, "argv", ["fetch_transcript.py", "dQw4w9WgXcQ"]):
                fetch_transcript.main()

        output = json.loads(capsys.readouterr().out)
        fetch_mock.assert_called_once_with("dQw4w9WgXcQ", None)
        assert output == {
            "video_id": "dQw4w9WgXcQ",
            "segment_count": 2,
            "duration": "1:02",
            "full_text": "hello world",
        }

    def test_json_timestamps_and_language_options(self, capsys):
        segments = [{"text": "hello", "start": 0.0, "duration": 1.5}]
        with mock.patch.object(fetch_transcript, "fetch_transcript", return_value=segments) as fetch_mock:
            with mock.patch.object(
                sys,
                "argv",
                ["fetch_transcript.py", "https://youtu.be/dQw4w9WgXcQ", "--language", "tr,en", "--timestamps"],
            ):
                fetch_transcript.main()

        output = json.loads(capsys.readouterr().out)
        fetch_mock.assert_called_once_with("dQw4w9WgXcQ", ["tr", "en"])
        assert output["timestamped_text"] == "0:00 hello"

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            (["--text-only"], "hello world\n"),
            (["--text-only", "--timestamps"], "0:00 hello\n1:00 world\n"),
        ],
    )
    def test_text_only_output(self, arguments, expected, capsys):
        segments = [
            {"text": "hello", "start": 0.0, "duration": 1.5},
            {"text": "world", "start": 60.0, "duration": 2.0},
        ]
        with mock.patch.object(fetch_transcript, "fetch_transcript", return_value=segments):
            with mock.patch.object(sys, "argv", ["fetch_transcript.py", "video-id", *arguments]):
                fetch_transcript.main()

        assert capsys.readouterr().out == expected

    def test_empty_transcript_uses_zero_duration(self, capsys):
        with mock.patch.object(fetch_transcript, "fetch_transcript", return_value=[]):
            with mock.patch.object(sys, "argv", ["fetch_transcript.py", "video-id"]):
                fetch_transcript.main()

        output = json.loads(capsys.readouterr().out)
        assert output["duration"] == "0:00"
        assert output["full_text"] == ""

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            ("captions disabled", {"error": "Transcripts are disabled for this video."}),
            ("no transcript available", {"error": "No transcript found. Try specifying a language with --language."}),
            ("network unavailable", {"error": "network unavailable"}),
        ],
    )
    def test_fetch_errors_are_reported_as_json(self, error, expected, capsys):
        with mock.patch.object(fetch_transcript, "fetch_transcript", side_effect=RuntimeError(error)):
            with mock.patch.object(sys, "argv", ["fetch_transcript.py", "video-id"]):
                with pytest.raises(SystemExit) as exc_info:
                    fetch_transcript.main()

        assert exc_info.value.code == 1
        assert json.loads(capsys.readouterr().out) == expected


class TestPyprojectDeclaresYoutubeExtra:
    def test_youtube_extra_declared_in_pyproject(self):
        """youtube-transcript-api must be listed in pyproject.toml [youtube] extra (issue #22243)."""
        import tomllib
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        extras = data.get("project", {}).get("optional-dependencies", {})
        assert "youtube" in extras, "Missing [youtube] extra in pyproject.toml"
        youtube_deps = " ".join(extras["youtube"])
        assert "youtube-transcript-api" in youtube_deps
