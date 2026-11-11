"""ffmpeg discovery for Discord voice: ``tools.transcription_audio`` owns the shared lookup
(PATH + Homebrew/local prefixes); this layers an explicit ``FFMPEG_PATH`` override and a
Windows winget fallback (installs that never touch PATH) on top."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from pathlib import Path


def _shared_find_ffmpeg():
    """Delegate to the repo-wide ffmpeg discovery helper when importable."""
    try:
        from tools.transcription_audio import _find_ffmpeg_binary
    except ImportError:  # standalone plugin import (tests / sandboxes)
        return shutil.which("ffmpeg")
    return _find_ffmpeg_binary()


def resolve_ffmpeg_executable() -> str:
    """Return an ffmpeg command that also covers common Windows installs."""
    explicit = (os.getenv("FFMPEG_PATH") or "").strip()
    if explicit:
        return os.path.expandvars(os.path.expanduser(explicit))
    if discovered := _shared_find_ffmpeg():
        return discovered
    if local_appdata := os.getenv("LOCALAPPDATA"):
        candidates = sorted((Path(local_appdata) / "Microsoft" / "WinGet" / "Packages").glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe"))
        if candidates:
            return str(candidates[-1])
    return "ffmpeg"


async def _finish_encoder(spawn):
    # Spawn can finish after the caller is cancelled. Own that child too, and
    # drain its pipe after killing it so asyncio's subprocess transport closes.
    proc = await spawn
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    await proc.communicate()


async def _encode_voice_audio(audio_path: str) -> bytes | None:
    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
        resolve_ffmpeg_executable(), "-v", "error", "-i", audio_path,
        "-acodec", "libopus", "-ac", "1", "-b:a", "48k", "-vbr", "on",
        "-application", "voip", "-compression_level", "10", "-f", "ogg", "pipe:1",
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL))
    cancelled = False
    try:
        proc = await asyncio.shield(spawn)
        data, _ = await proc.communicate()
        return data if proc.returncode == 0 and data else None
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        cleanup = asyncio.create_task(_finish_encoder(spawn))
        try:
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
        finally:
            # Even a failed spawn must not turn cancellation into file fallback.
            if cancelled:
                raise asyncio.CancelledError


async def transcode_voice_audio(audio_path: str) -> bytes | None:
    """Own ffmpeg until it is reaped, including cancellation during spawn.

    Pipe the Ogg result instead of leaving a temp file behind on cancellation.
    Missing ffmpeg, failed encodes and the encoder deadline retain file fallback.
    """
    try:
        async with asyncio.timeout(30):
            return await _encode_voice_audio(audio_path)
    except (OSError, TimeoutError):
        return None
