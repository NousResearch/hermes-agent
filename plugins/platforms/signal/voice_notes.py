"""Prepare outbound voice attachments for the Signal send RPC."""
import asyncio
import logging
from contextlib import ExitStack, asynccontextmanager, suppress
from pathlib import Path
from tempfile import NamedTemporaryFile

from hermes_platform.resolver import locate_command
from tools.audio_container import sniff_container

logger = logging.getLogger(__name__)
SIGNAL_MAX_ATTACHMENT_SIZE = 100 * 1024 * 1024


def _read_container(path):
    with path.open("rb") as audio:
        return sniff_container(audio.read(16))


async def _prepare_voice_file(source, cleanup):
    path = Path(source)
    container = await asyncio.to_thread(_read_container, path)
    if container == "m4a" and path.suffix.lower() == ".m4a":
        return source
    if container is None:
        return None
    ffmpeg = locate_command("ffmpeg", known_dirs=("/opt/homebrew/bin", "/usr/local/bin")).command
    if not ffmpeg:
        return None
    # signal-cli opens the path in its own process. Keep it beside the shared input.
    with NamedTemporaryFile(dir=path.parent, prefix=".hermes-voice-", suffix=".m4a", delete=False) as tmp:
        target = Path(tmp.name)
        cleanup.callback(target.unlink, missing_ok=True)
    codec = ["-c:a", "copy"] if container == "aac" else ["-c:a", "aac", "-b:a", "48k"]
    process = await asyncio.create_subprocess_exec(
        *ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-protocol_whitelist", "file", "-i", source,
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


@asynccontextmanager
async def prepare_voice_batch(media):
    """Keep converted files until the RPC (including retries) finishes.

    signal-cli applies voiceNote to the entire message. On a mixed batch or a
    failed conversion, deliver every original in order as an ordinary attachment.
    """
    originals = [path for path, _ in media]
    if not media or not all(voice for _, voice in media):
        yield originals, False
        return
    with ExitStack() as cleanup:
        prepared = []
        for path in originals:
            try:
                output = await _prepare_voice_file(path, cleanup)
            except (OSError, TimeoutError) as exc:
                logger.debug("Signal voice-note preparation failed: %s", exc)
                output = None
            if output is None:
                yield originals, False
                return
            prepared.append(output)
        yield prepared, True
