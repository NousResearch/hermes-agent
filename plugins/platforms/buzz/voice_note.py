"""Buzz voice-note packaging and inbound audio classification.

Buzz has no audio upload path. The relay accepts **video/mp4** only, and a voice
note is a video/mp4 whose single H.264 track is a 16x16 black frame whose only
user-facing content is the AAC audio track. That envelope is what the Buzz
desktop app produces (``desktop/src-tauri/src/commands/media_transcode.rs``) and
what the relay's media validator enforces.

Three pieces live here:

* :func:`package_voice_note` — wrap an audio file in that canonical envelope.
* :func:`is_voice_note_filename` / :func:`voice_note_media_type` — recognise an
  inbound voice note so the gateway's STT gate treats it as speech, not video.
* :func:`classify_inbound_voice_note` — the single decision used by the adapter:
  given an inbound attachment's declared MIME and filename, decide whether it is
  a voice note (and therefore STT input) and which MIME to report.

The relay's own rules, mirrored here:

* exactly one H.264 video track, at most one AAC audio track
* ``moov`` before ``mdat`` (fast-start)
* no ``udta``/``meta``/``ilst`` metadata boxes
* duration <= 600 s, short edge <= 2160, long edge <= 3840
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

#: Filename convention the Buzz clients use for a recorded voice note.
VOICE_NOTE_FILENAME_PREFIX = "voice-note-"

#: Video dimensions of a Buzz voice note. A voice note is a video/mp4 whose video
#: track is a 16x16 black still, so the relay records ``dim 16x16`` on the
#: ``imeta`` tag. This is the most reliable voice-note signal, because the
#: relay is content-addressed and never learns a filename — the ``filename``
#: field is a client-side convention the Buzz desktop emits
#: (``voice-note-<ts>.mp4``) but which older CLI and adapter sends omitted.
#: Detect by dimensions FIRST so a note with no filename is still recognised.
VOICE_NOTE_DIMENSIONS = "16x16"

#: MIME the gateway must see for a Buzz voice note to enter the STT pipeline.
#: Buzz posts voice notes as ``video/mp4`` (the relay has no audio allowlist), so
#: the adapter rewrites the reported MIME to ``audio/mp4``.
VOICE_NOTE_MIME = "audio/mp4"

#: MIME Buzz's clients actually declare for a packaged voice note.
VOICE_NOTE_DECLARED_MIME = "video/mp4"

#: ffmpeg timeout for one packaging run.
_FFMPEG_TIMEOUT_SECONDS = 120

#: Extensions we can feed to ffmpeg as a voice-note source.
_AUDIO_SOURCE_EXTS = frozenset({".mp3", ".ogg", ".opus", ".wav", ".m4a", ".aac", ".flac", ".mp4", ".webm"})

#: The ffmpeg arguments that produce the canonical Buzz voice-note MP4, split
#: into the input side and the output side because the source input must sit
#: between them on the command line.
#:
#: Mirrors the desktop app's ``transcode_voice_note_to_mp4_with_cancellation``:
#: a 16x16 black H.264 still satisfies the relay's video validator while the AAC
#: track carries the speech. The ``+bitexact`` flags and ``-metadata encoder=``
#: are REQUIRED — without them ffmpeg writes a ``udta``/``meta``/``ilst`` box and
#: the relay rejects the upload with "media contains metadata or a non-canonical
#: metadata channel".
_FFMPEG_INPUT_ARGS = (
    "-y",
    "-nostdin",
    "-loglevel", "error",
    "-f", "lavfi", "-i", "color=c=black:s=16x16:r=1",
)

_FFMPEG_OUTPUT_ARGS = (
    "-map", "0:v:0", "-map", "1:a:0", "-shortest",
    "-map_metadata", "-1", "-map_chapters", "-1", "-sn", "-dn",
    "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "96k",
    "-movflags", "+faststart",
    "-metadata", "encoder=",
)


def ffmpeg_path() -> Optional[str]:
    """Absolute path to an ``ffmpeg`` binary, or ``None`` when unavailable."""
    return shutil.which("ffmpeg")


def is_voice_note_filename(filename: Optional[str]) -> bool:
    """True when *filename* follows Buzz's voice-note naming convention.

    Buzz's desktop and mobile clients both name a recorded voice note
    ``voice-note-<id>.mp4``; the pre-packaging source is ``voice-note-<id>.wav``.
    """
    if not filename:
        return False
    lower = filename.strip().lower()
    if not lower.startswith(VOICE_NOTE_FILENAME_PREFIX):
        return False
    return os.path.splitext(lower)[1] in (".mp4", ".wav", ".m4a", ".mp3", ".ogg", ".opus")


def voice_note_media_type(declared_mime: str, filename: Optional[str], dimensions: str = "") -> str:
    """The MIME the gateway should report for an inbound Buzz voice note.

    Buzz declares a packaged voice note as ``video/mp4`` because the relay has no
    audio allowlist. The gateway's STT gate keys on ``audio/``, so a voice note
    reported as ``video/mp4`` is treated as a video and never transcribed. This
    rewrites the reported MIME to ``audio/mp4`` for a recognised voice note and
    otherwise returns *declared_mime* unchanged.
    """
    if is_voice_note_attachment(declared_mime, filename, dimensions):
        return VOICE_NOTE_MIME
    return declared_mime


def is_voice_note_attachment(declared_mime: str, filename: Optional[str], dimensions: str = "") -> bool:
    """True when an inbound attachment is a Buzz voice note.

    Two independent signals, either of which is sufficient:

    1. **Dimensions** — a Buzz voice note's video track is a 16x16 black still, so
       the relay records ``dim 16x16``. This is the reliable signal, because Buzz
       attachments carry NO ``filename`` field at all.
    2. **Filename** — the ``voice-note-*`` convention, when a client does supply one.
    """
    if (dimensions or "").strip().lower() == VOICE_NOTE_DIMENSIONS:
        return True
    return is_voice_note_filename(filename)


def classify_inbound_voice_note(
    declared_mime: str, filename: Optional[str], dimensions: str = ""
) -> Tuple[bool, str]:
    """Decide whether an inbound attachment is a voice note.

    Returns ``(is_voice_note, reported_mime)``.

    A voice note is recognised two ways:

    1. By **dimensions** (``dim 16x16`` on the ``imeta`` tag) — the reliable
       signal, because Buzz attachments carry no ``filename`` field.
    2. By **filename** — ``voice-note-*.mp4`` — the convention both Buzz clients
       use when one is supplied.

    A real ``video/mp4`` (a screen recording, a clip) is NOT a voice note: its
    dimensions are ordinary video dimensions, so it stays a video and never
    enters STT.
    """
    mime = (declared_mime or "").strip().lower()
    if mime.startswith("audio/"):
        # An audio attachment is speech by definition. The one exception is a
        # caller that mixed kinds, which the adapter already handles separately.
        return True, (mime or VOICE_NOTE_MIME)
    if is_voice_note_attachment(declared_mime, filename, dimensions):
        return True, VOICE_NOTE_MIME
    return False, mime


def package_voice_note(source: Path, destination: Optional[Path] = None) -> Optional[Path]:
    """Package an audio *source* into Buzz's canonical voice-note MP4.

    Returns the path to the packaged MP4, or ``None`` when the audio could not
    be packaged (missing ffmpeg, bad input, ffmpeg failure). The caller owns the
    returned file and must delete it.

    The result is accepted by the Buzz relay's media validator: one 16x16 black
    H.264 track, one AAC track, fast-start, and no metadata boxes.
    """
    source = Path(source)
    if not source.is_file():
        logger.warning("Buzz: cannot package voice note — source not found: %s", source.name)
        return None
    if source.suffix.lower() not in _AUDIO_SOURCE_EXTS:
        logger.warning("Buzz: cannot package voice note — unsupported source type: %s", source.suffix)
        return None
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        logger.warning("Buzz: cannot package voice note — ffmpeg is not installed")
        return None

    if destination is None:
        # Name the packaged file with the SAME convention the Buzz desktop uses
        # (`voice-note-<id>.mp4`). The desktop's renderer only treats an
        # attachment as a voice note when the imeta `filename` starts with
        # `voice-note-` (desktop/src/features/messages/lib/audioAttachment.ts),
        # so a `buzz-voice-note-*.mp4` name renders as a video player instead
        # of a playable audio bubble. The relay is content-addressed, so the
        # name carries no storage meaning — it is purely a client-side label.
        destination = Path(
            tempfile.mkstemp(prefix=VOICE_NOTE_FILENAME_PREFIX, suffix=".mp4")[1]
        )
    else:
        destination = Path(destination)

    cmd = [ffmpeg, *_FFMPEG_INPUT_ARGS, "-i", str(source), *_FFMPEG_OUTPUT_ARGS, str(destination)]
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Buzz: voice-note packaging failed for %s: %s", source.name, exc)
        _unlink(destination)
        return None
    if completed.returncode != 0:
        stderr = (completed.stderr or b"").decode("utf-8", "replace").strip()
        logger.warning("Buzz: voice-note packaging failed for %s (rc=%d): %s",
                       source.name, completed.returncode, stderr[:300])
        _unlink(destination)
        return None
    if not destination.is_file() or destination.stat().st_size == 0:
        logger.warning("Buzz: voice-note packaging produced an empty file for %s", source.name)
        _unlink(destination)
        return None
    return destination


def _unlink(path: Path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass
