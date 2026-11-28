"""Locate FFmpeg for Signal audio and encode outgoing iPhone voice notes."""
import asyncio
import os
import shutil
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile

SIGNAL_MAX_ATTACHMENT_SIZE = 100 * 1024 * 1024


def find_ffmpeg() -> str | None:
    # Fall back to common Homebrew/local prefixes on macOS dev hosts.
    return shutil.which("ffmpeg") or next(
        (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
         if os.path.isfile(p) and os.access(p, os.X_OK)), None)


async def convert_to_voice_note(path: Path, container: str, cleanup):
    """Return a converted file whose lifetime extends through all send attempts."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    # signal-cli opens the path in its own process. Keep it beside the shared input.
    with NamedTemporaryFile(dir=path.parent, prefix=".hermes-voice-", suffix=".m4a", delete=False) as tmp:
        target = Path(tmp.name)
        cleanup.callback(target.unlink, missing_ok=True)
    codec = ["-c:a", "copy"] if container == "aac" else ["-c:a", "aac", "-b:a", "48k"]
    process = await asyncio.create_subprocess_exec(
        ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-protocol_whitelist", "file", "-i", str(path),
        "-map", "0:a:0", *codec, "-movflags", "+faststart", str(target),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        status = await asyncio.wait_for(process.wait(), timeout=60)
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await asyncio.wait_for(process.wait(), timeout=10)
    if status == 0 and 0 < (await asyncio.to_thread(target.stat)).st_size <= SIGNAL_MAX_ATTACHMENT_SIZE:
        return str(target)
    return None
