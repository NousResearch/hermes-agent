"""Local audio preparation for DingTalk's sampleAudio template.

STT helpers encode AAC and only probe duration; native DingTalk voice needs an
AMR-NB codec check as well as cancellable subprocesses, so keep this at the edge.
"""

import asyncio
import json
import math
import os
import tempfile
from pathlib import Path

from hermes_cli._subprocess_compat import windows_hide_flags

VOICE_MAX_BYTES = 2 * 1024 * 1024


async def run_audio_tool(*args: str, timeout: float) -> bytes:
    """Never expose tool stderr (which can contain local paths) to chat/logs."""
    tool = args[0]
    try:
        process = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            creationflags=windows_hide_flags(),
        )
    except FileNotFoundError:
        raise ValueError(f"{tool} is required for DingTalk native voice but was not found.") from None
    except OSError:
        raise ValueError(f"{tool} could not be started for DingTalk native voice.") from None
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass  # Exited between the returncode check and kill.
        await asyncio.shield(process.communicate())
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise ValueError(f"{tool} timed out preparing DingTalk native voice.") from None
    if process.returncode:
        if tool == "ffmpeg" and b"Unknown encoder" in stderr and b"libopencore_amrnb" in stderr:
            raise ValueError("ffmpeg needs the libopencore_amrnb encoder for DingTalk native voice.")
        raise ValueError(f"{tool} failed preparing DingTalk native voice.")
    return stdout


def _read_voice(path: Path) -> bytes:
    # Bound the actual read, not merely stat(): input can change between them.
    with path.open("rb") as audio:
        data = audio.read(VOICE_MAX_BYTES + 1)
    if len(data) > VOICE_MAX_BYTES:
        raise ValueError("DingTalk voice media exceeds the 2 MB upload limit.")
    if not data.startswith(b"#!AMR\n"):
        raise ValueError("DingTalk voice media must be an AMR-NB file.")
    return data


async def prepare_voice(audio_path: str) -> tuple[bytes, int]:
    """Return immutable AMR-NB bytes and probed milliseconds; never edit the input."""
    try:
        source = Path(audio_path).resolve()
        if not source.is_file():
            raise ValueError("Audio file not found.")
        with tempfile.TemporaryDirectory(prefix="hermes-dingtalk-") as directory:
            target = Path(directory) / "voice.amr"
            if source.suffix.lower() == ".amr":
                media = await asyncio.to_thread(_read_voice, source)
                # Probe the exact snapshot uploaded, not a mutable caller-owned file.
                await asyncio.to_thread(target.write_bytes, media)
            else:
                await run_audio_tool(
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", os.fspath(source), "-vn", "-ac", "1", "-ar", "8000",
                    "-c:a", "libopencore_amrnb", "-b:a", "12.2k", os.fspath(target),
                    timeout=60.0,
                )
                media = await asyncio.to_thread(_read_voice, target)
            output = await run_audio_tool(
                "ffprobe", "-v", "error", "-show_entries",
                "stream=codec_name,sample_rate,channels:format=duration", "-of", "json",
                os.fspath(target), timeout=10.0,
            )
            try:
                probe = json.loads(output)
                streams = probe["streams"]
                stream = streams[0]
                duration = float(probe["format"]["duration"])
                valid = (len(streams) == 1 and stream["codec_name"] == "amr_nb"
                         and int(stream["sample_rate"]) == 8000 and stream["channels"] == 1
                         and math.isfinite(duration) and duration > 0)
            except (ValueError, KeyError, IndexError, TypeError):
                valid = False
            if not valid:
                raise ValueError("DingTalk voice requires AMR-NB, 8 kHz, mono and a valid duration.")
            return media, max(1, round(duration * 1000))
    except (OSError, TypeError):
        raise ValueError("Audio file could not be read or prepared.") from None
