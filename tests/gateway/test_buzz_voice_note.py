"""Tests for the Buzz voice-note packaging and inbound classification module.

These bind to the production seams: the ffmpeg argument vector that produces the
canonical Buzz envelope, and the classification decision the adapter uses to
decide whether an inbound attachment is speech or video.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from plugins.platforms.buzz.voice_note import (
    VOICE_NOTE_FILENAME_PREFIX,
    VOICE_NOTE_MIME,
    classify_inbound_voice_note,
    ffmpeg_path,
    is_voice_note_filename,
    package_voice_note,
)

ffmpeg_required = pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="ffmpeg is required to package a voice note"
)


def _top_level_atoms(path: Path) -> list[str]:
    """The ordered top-level atom (box) types of an MP4 file."""
    import struct

    atoms: list[str] = []
    data = path.read_bytes()
    offset = 0
    while offset + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[offset:offset + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[offset + 8:offset + 16])[0]
            header = 16
        if size == 0:
            break
        atoms.append(kind.decode("latin1"))
        offset += size
    return atoms


class TestVoiceNoteFilename:
    def test_buzz_client_convention_is_recognised(self):
        assert is_voice_note_filename("voice-note-123.mp4") is True
        assert is_voice_note_filename("voice-note-123.wav") is True
        assert is_voice_note_filename("VOICE-NOTE-123.MP4") is True

    def test_an_ordinary_media_file_is_not_a_voice_note(self):
        assert is_voice_note_filename("screen-recording.mp4") is False
        assert is_voice_note_filename("clip.mp4") is False
        assert is_voice_note_filename("voice-note-123.txt") is False

    def test_missing_filename_is_not_a_voice_note(self):
        assert is_voice_note_filename(None) is False
        assert is_voice_note_filename("") is False


class TestInboundClassification:
    def test_voice_note_uploaded_as_video_mp4_is_reported_as_audio(self):
        """The relay has no audio allowlist, so Buzz posts voice notes as video/mp4."""
        is_voice_note, reported = classify_inbound_voice_note("video/mp4", "", "16x16")
        assert is_voice_note is True
        assert reported == VOICE_NOTE_MIME

    def test_voice_note_without_any_filename_is_recognised_by_dimensions(self):
        """Buzz attachments carry NO filename field — the dim tag is the only signal.

        Verified against the live relay: zero attachments in the database have a
        ``filename`` field, so a classifier that depends on the filename never
        fires for a real Buzz voice note.
        """
        is_voice_note, reported = classify_inbound_voice_note("video/mp4", "", "16x16")
        assert is_voice_note is True
        assert reported == VOICE_NOTE_MIME

    def test_voice_note_filename_still_works_when_a_client_supplies_one(self):
        is_voice_note, reported = classify_inbound_voice_note("video/mp4", "voice-note-1.mp4", "")
        assert is_voice_note is True
        assert reported == VOICE_NOTE_MIME

    def test_a_real_video_stays_a_video(self):
        """A screen recording must never be mistaken for speech."""
        is_voice_note, reported = classify_inbound_voice_note("video/mp4", "screen-recording.mp4", "1920x1080")
        assert is_voice_note is False
        assert reported == "video/mp4"

    def test_a_real_video_without_dimensions_stays_a_video(self):
        is_voice_note, reported = classify_inbound_voice_note("video/mp4", "", "")
        assert is_voice_note is False
        assert reported == "video/mp4"

    def test_an_audio_attachment_is_speech(self):
        is_voice_note, reported = classify_inbound_voice_note("audio/mpeg", "clip.mp3", "")
        assert is_voice_note is True
        assert reported == "audio/mpeg"

    def test_a_document_is_neither(self):
        is_voice_note, reported = classify_inbound_voice_note("application/pdf", "notes.pdf", "")
        assert is_voice_note is False
        assert reported == "application/pdf"


class TestPackaging:
    def test_ffmpeg_is_discoverable(self):
        assert ffmpeg_path() is not None

    @ffmpeg_required
    def test_packaging_produces_a_canonical_voice_note(self, tmp_path):
        source = tmp_path / "speech.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             str(source)],
            check=True, timeout=60,
        )

        packaged = package_voice_note(source)
        assert packaged is not None
        try:
            assert packaged.is_file()
            assert packaged.stat().st_size > 0
            assert packaged.suffix == ".mp4"

            # The relay's validator requires: moov before mdat (fast-start), one
            # H.264 video track, one AAC audio track, and no metadata boxes.
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "stream=codec_type,codec_name", "-of", "csv=p=0", str(packaged)],
                capture_output=True, text=True, timeout=60,
            )
            codecs = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
            assert "h264,video" in codecs
            assert "aac,audio" in codecs

            # Fast-start: the moov atom must precede mdat. Walk the top-level
            # atom chain rather than scanning raw bytes (mdat is not in the head).
            atoms = _top_level_atoms(packaged)
            assert "moov" in atoms and "mdat" in atoms
            assert atoms.index("moov") < atoms.index("mdat"), (
                f"moov must precede mdat for the relay's fast-start check; got {atoms}"
            )

            # The relay forbids metadata boxes (udta/meta/ilst) and fragmented
            # containers (moof/mfra). A fragmented MP4 keeps moov before mdat but
            # still trips "media contains metadata or a non-canonical metadata
            # channel" on the relay, so the atom set itself must be checked.
            forbidden = {"udta", "meta", "ilst", "moof", "mfra"}
            present = forbidden.intersection(atoms)
            assert not present, (
                f"the relay rejects these atoms in a voice note; got {sorted(present)} in {atoms}"
            )
        finally:
            packaged.unlink(missing_ok=True)

    @ffmpeg_required
    def test_packaging_is_deterministic_for_the_same_source(self, tmp_path):
        """The same audio must produce the same bytes, so uploads are cacheable."""
        source = tmp_path / "speech.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             str(source)],
            check=True, timeout=60,
        )
        first = package_voice_note(source)
        second = package_voice_note(source)
        assert first is not None and second is not None
        try:
            assert first.read_bytes() == second.read_bytes()
        finally:
            first.unlink(missing_ok=True)
            second.unlink(missing_ok=True)

    def test_missing_source_returns_none(self, tmp_path):
        assert package_voice_note(tmp_path / "does-not-exist.mp3") is None

    def test_unsupported_source_type_returns_none(self, tmp_path):
        source = tmp_path / "notes.txt"
        source.write_text("not audio")
        assert package_voice_note(source) is None

    @ffmpeg_required
    def test_undecodable_audio_returns_none_rather_than_raising(self, tmp_path):
        source = tmp_path / "broken.mp3"
        source.write_bytes(b"this is not real audio")
        assert package_voice_note(source) is None

    @ffmpeg_required
    def test_packaged_filename_uses_the_client_voice_note_convention(self, tmp_path):
        """The packaged file must be named like a Buzz client's voice note.

        The desktop renderer only treats an attachment as a voice note when the
        imeta ``filename`` starts with ``voice-note-``
        (desktop/src/features/messages/lib/audioAttachment.ts). A
        ``buzz-voice-note-*.mp4`` name renders as a video player instead of a
        playable audio bubble, so the packaging name is a client-visible
        contract, not a cosmetic detail.
        """
        source = tmp_path / "speech.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             str(source)],
            check=True, timeout=60,
        )
        packaged = package_voice_note(source)
        assert packaged is not None
        try:
            assert packaged.name.startswith(VOICE_NOTE_FILENAME_PREFIX), (
                f"packaged voice note must start with {VOICE_NOTE_FILENAME_PREFIX!r}; "
                f"got {packaged.name!r}"
            )
            # And it must be recognised by our own inbound classifier.
            assert is_voice_note_filename(packaged.name) is True
        finally:
            packaged.unlink(missing_ok=True)

    @ffmpeg_required
    def test_explicit_destination_is_honoured(self, tmp_path):
        source = tmp_path / "speech.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             str(source)],
            check=True, timeout=60,
        )
        destination = tmp_path / "voice-note-out.mp4"
        packaged = package_voice_note(source, destination)
        assert packaged == destination
        assert destination.is_file()
        destination.unlink(missing_ok=True)
