"""Off-loop preparation and temporary-file ownership for native voice uploads."""

import asyncio
import logging
import os
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_platform import resolver
from tools.audio_container import sniff_audio_ext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedAttachment:
    path: str
    filename: str
    content_type: str
    cleanup: bool = False
    native_voice: bool = False


def convert_audio_to_caf(audio_path: str) -> Optional[str]:
    """Normalize sources, including generated MP3, to mono 24 kHz Opus CAF."""
    afconvert = resolver.locate_command("afconvert").command
    if not afconvert:
        return None
    caf_path = normalized_path = None
    keep_caf = False
    try:
        with tempfile.NamedTemporaryFile(prefix="hermes-bluebubbles-voice-", suffix=".caf", delete=False) as caf:
            caf_path = caf.name
        ffmpeg = resolver.locate_command("ffmpeg").command
        if not ffmpeg:
            return None
        with tempfile.NamedTemporaryFile(prefix="hermes-bluebubbles-voice-src-", suffix=".wav", delete=False) as wav:
            normalized_path = wav.name
        options: dict[str, Any] = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=60, creationflags=windows_hide_flags())
        subprocess.run([*ffmpeg, "-y", "-i", audio_path, "-vn", "-ac", "1", "-ar", "24000", normalized_path], check=True, **options)
        if not os.path.getsize(normalized_path):
            return None
        subprocess.run([*afconvert, "-f", "caff", "-d", "opus@24000", "-c", "1", normalized_path, caf_path], check=True, **options)
        with open(caf_path, "rb") as caf_output:
            if caf_output.read(4) == b"caff":
                keep_caf = True
                return caf_path
    except (OSError, subprocess.SubprocessError):
        logger.warning("BlueBubbles native voice conversion failed; using ordinary audio")
    finally:
        for path in (normalized_path, caf_path if not keep_caf else None):
            if path:
                with suppress(OSError):
                    os.unlink(path)
    return None


def prepare_voice_attachment(file_path: str, filename: Optional[str] = None) -> PreparedAttachment:
    with open(file_path, "rb") as source:
        header = source.read(12)
    if header.startswith(b"caff"):
        return PreparedAttachment(file_path, "Audio Message.caf", "audio/x-caf", native_voice=True)
    converted = convert_audio_to_caf(file_path)
    if converted:
        return PreparedAttachment(converted, "Audio Message.caf", "audio/x-caf", cleanup=True, native_voice=True)
    requested_name = Path(filename or file_path)
    # Providers may ignore the requested suffix; reuse the shared container sniffer.
    extension = sniff_audio_ext(header, requested_name.suffix.lower() or ".bin")
    audio_types = {
        ".caf": "audio/x-caf", ".mp3": "audio/mpeg", ".m2a": "audio/mpeg",
        ".m4a": "audio/mp4", ".aac": "audio/aac", ".wav": "audio/wav",
        ".ogg": "audio/ogg", ".opus": "audio/opus", ".flac": "audio/flac",
        ".webm": "audio/webm",
    }
    logger.warning("BlueBubbles native voice conversion unavailable; sending ordinary audio")
    return PreparedAttachment(file_path, requested_name.with_suffix(extension).name,
                              audio_types.get(extension, "application/octet-stream"))


def cleanup_attachment(prepared: PreparedAttachment) -> None:
    if prepared.cleanup:
        with suppress(OSError):
            os.unlink(prepared.path)


def _prepare_upload(file_path: str, filename: Optional[str]) -> tuple[PreparedAttachment, bytes]:
    # Keep creation, reading, and disposal in the same thread: even cancellation
    # of the asyncio wrapper during shutdown cannot strand an owned CAF.
    prepared = prepare_voice_attachment(file_path, filename)
    try:
        return prepared, Path(prepared.path).read_bytes()
    finally:
        cleanup_attachment(prepared)


def _consume_worker_result(task: asyncio.Task) -> None:
    with suppress(Exception, asyncio.CancelledError):
        task.result()


async def prepare_voice_upload(file_path: str, filename: Optional[str] = None) -> tuple[PreparedAttachment, bytes]:
    task = asyncio.create_task(asyncio.to_thread(_prepare_upload, file_path, filename))
    try:
        # Unlike shield(), wait() does not log a worker exception after cancellation
        # on Python 3.14. This path or its callback owns the result, not the files.
        await asyncio.wait({task})
        return task.result()
    except asyncio.CancelledError:
        task.add_done_callback(_consume_worker_result)
        raise
