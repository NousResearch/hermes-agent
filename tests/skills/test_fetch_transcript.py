"""Tests for skills/media/youtube-content/scripts/fetch_transcript.py (issue #22243)."""

import sys
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


class TestFormatTimestamp:
    def test_seconds_only(self):
        assert fetch_transcript.format_timestamp(90) == "1:30"


    def test_zero(self):
        assert fetch_transcript.format_timestamp(0) == "0:00"

    def test_minutes_only(self):
        assert fetch_transcript.format_timestamp(600) == "10:00"




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

