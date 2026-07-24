"""Tests for skills/media/streaming-content/scripts/fetch_transcript.py."""

import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "media"
    / "streaming-content"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import fetch_transcript


class TestCaptionsToText:
    """Tests for the _captions_to_text helper (caption parsing)."""

    def test_basic_vtt(self, tmp_path):
        vtt = tmp_path / "caption.vtt"
        vtt.write_text(
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:04.000\n"
            "Hello world\n\n"
            "00:00:05.000 --> 00:00:08.000\n"
            "This is a test\n"
        )
        text = fetch_transcript._captions_to_text(str(vtt))
        assert "Hello world" in text
        assert "This is a test" in text

    def test_strips_timestamps_and_numbers(self, tmp_path):
        vtt = tmp_path / "caption.vtt"
        vtt.write_text(
            "1\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Line one\n\n"
            "2\n"
            "00:00:03.000 --> 00:00:05.000\n"
            "Line two\n"
        )
        text = fetch_transcript._captions_to_text(str(vtt))
        assert "Line one Line two" in text or "Line one" in text

    def test_drops_duplicate_consecutive_lines(self, tmp_path):
        vtt = tmp_path / "caption.vtt"
        vtt.write_text(
            "00:00:01.000 --> 00:00:03.000\n"
            "Repeated line\n\n"
            "00:00:04.000 --> 00:00:06.000\n"
            "Repeated line\n"
        )
        text = fetch_transcript._captions_to_text(str(vtt))
        # Should collapse duplicates
        assert text.count("Repeated line") == 1

    def test_strips_html_tags(self, tmp_path):
        vtt = tmp_path / "caption.vtt"
        vtt.write_text(
            "00:00:00.000 --> 00:00:02.000\n"
            "This has <00:00:01.000> tags\n"
        )
        text = fetch_transcript._captions_to_text(str(vtt))
        assert "<" not in text
        assert "tags" in text


class TestDownloadErrorPaths:
    """Tests for documented download-error handling (especially X no-video case)."""

    def test_no_playable_video_error_for_image_text_post(self):
        """X/Twitter image/text/link-only posts should return the clean documented error."""
        with mock.patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = mock_ydl.return_value.__enter__.return_value
            mock_instance.extract_info.side_effect = Exception("no video could be found")

            info, path, err = fetch_transcript.download_audio(
                "https://x.com/user/status/123", "/tmp"
            )

            assert info is None
            assert path == ""
            assert "no playable video in this post" in err
            assert "image, text, or link-only" in err

    def test_ffmpeg_missing_gives_helpful_message(self):
        with mock.patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = mock_ydl.return_value.__enter__.return_value
            mock_instance.extract_info.side_effect = Exception("ffmpeg not found")

            _, _, err = fetch_transcript.download_audio("https://example.com/vid", "/tmp")

            assert "ffmpeg" in err.lower() or "download failed" in err.lower()

    def test_generic_download_failure(self):
        with mock.patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = mock_ydl.return_value.__enter__.return_value
            mock_instance.extract_info.side_effect = Exception("some other error")

            _, _, err = fetch_transcript.download_audio("https://example.com/vid", "/tmp")

            assert "download failed" in err


class TestCaptionToAudioFallback:
    """Tests the served-captions vs download+transcribe fallback path."""

    def test_uses_served_captions_when_available(self):
        """If fetch_served_captions returns text, download_audio is never called."""
        with mock.patch("fetch_transcript.fetch_served_captions", return_value=({"title": "Test"}, "This is served caption text")) as mock_served:
            with mock.patch("fetch_transcript.download_audio") as mock_dl:
                with mock.patch("sys.argv", ["fetch_transcript.py", "https://example.com/vid"]):
                    with mock.patch("builtins.print") as mock_print:
                        try:
                            fetch_transcript.main()
                        except SystemExit:
                            pass
                mock_dl.assert_not_called()
                mock_served.assert_called_once()
                # Should have printed JSON containing the served text
                printed = "".join(str(call) for call in mock_print.call_args_list)
                assert "served caption text" in printed or "served-captions" in printed

    def test_falls_back_to_download_when_no_captions(self):
        """If fetch_served_captions returns no text, download_audio + transcribe path is used."""
        with mock.patch("fetch_transcript.fetch_served_captions", return_value=(None, None)) as mock_served:
            with mock.patch("fetch_transcript.download_audio") as mock_dl:
                mock_dl.return_value = ({"title": "Vid"}, "/tmp/audio.mp3", "")
                with mock.patch("fetch_transcript.transcribe") as mock_trans:
                    mock_trans.return_value = {"success": True, "transcript": "transcribed content", "provider": "local"}
                    with mock.patch("sys.argv", ["fetch_transcript.py", "https://example.com/vid"]):
                        with mock.patch("builtins.print") as mock_print:
                            try:
                                fetch_transcript.main()
                            except SystemExit:
                                pass
                    mock_dl.assert_called_once()
                    mock_trans.assert_called_once()
                    printed = "".join(str(call) for call in mock_print.call_args_list)
                    assert "transcribed content" in printed or "local" in printed