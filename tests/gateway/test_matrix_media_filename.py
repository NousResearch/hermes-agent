"""Tests for Matrix adapter media-filename blanking.

Matrix ``m.image`` events were already handled (PR #16821, issue #13482).
These tests verify that ``m.audio``, ``m.file``, and ``m.video`` events
also have their transport filename stripped from ``body`` when no caption
is present, so the filename doesn't leak into ``event.text`` and get fed
to the model as user message text.
"""

import pytest

from plugins.platforms.matrix.adapter import (
    _looks_like_matrix_image_filename,
    _looks_like_matrix_media_filename,
)


class TestLooksLikeMatrixMediaFilename:
    """Unit tests for _looks_like_matrix_media_filename()."""

    @pytest.mark.parametrize(
        "text",
        [
            "voice-20260725-114233.ogg",
            "recording.wav",
            "clip.opus",
            "audio.amr",
            "data.m4a",
            "clip.webm",
            "video.mkv",
            "video.mov",
            "CaPtIoN.mp3",
        ],
    )
    def test_bare_media_filenames_detected(self, text):
        assert _looks_like_matrix_media_filename(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "remind me to call the dentist",
            "voice note.m4a",
            "document.pdf",
            "file.txt",
            "noextension",
            "data.csv",
            "",
            "  ",
            None,
        ],
    )
    def test_non_filenames_rejected(self, text):
        assert _looks_like_matrix_media_filename(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "/tmp/voice.ogg",
            "./audio.mp3",
        ],
    )
    def test_paths_with_separators_rejected(self, text):
        assert _looks_like_matrix_media_filename(text) is False

    def test_trailing_whitespace_stripped(self):
        assert _looks_like_matrix_media_filename("audio.ogg\n") is True
        assert _looks_like_matrix_media_filename("  photo.mp3  ") is True

    def test_multi_token_filename_rejected(self):
        """A filename with a space is treated as a caption, not a bare token."""
        assert _looks_like_matrix_media_filename("my recording.ogg") is False

    def test_newline_in_text_rejected(self):
        """Multi-line text is never a bare filename."""
        assert _looks_like_matrix_media_filename("voice.ogg\nmore text") is False

    def test_image_extension_not_detected_by_media_function(self):
        """Image extensions are handled by the image function, not the media function."""
        assert _looks_like_matrix_media_filename("photo.jpg") is False
        assert _looks_like_matrix_image_filename("photo.jpg") is True

    def test_audio_extension_not_detected_by_image_function(self):
        """Audio extensions are handled by the media function, not the image function."""
        assert _looks_like_matrix_image_filename("voice.ogg") is False
        assert _looks_like_matrix_media_filename("voice.ogg") is True

    def test_mimetypes_audio_detected(self):
        """mimetypes.guess_type catches audio types not in the hardcoded set."""
        # .aiff is audio/x-aiff, NOT in _MATRIX_MEDIA_FILENAME_EXTS
        assert _looks_like_matrix_media_filename("song.aiff") is True

    def test_mimetypes_video_detected(self):
        """mimetypes.guess_type catches video types."""
        assert _looks_like_matrix_media_filename("clip.avi") is True
