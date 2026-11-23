"""ffmpeg transcoding into the containers native voice bubbles require."""
import contextlib
import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)

# container -> (ffmpeg muxer, encoder args). Opus is voip-tuned; AAC/MP4 is what iMessage plays.
_VOICE_ENCODERS = {
    "ogg": ("ogg", ["-acodec", "libopus", "-vbr", "on", "-application", "voip",
                    "-compression_level", "10"]),
    "m4a": ("ipod", ["-acodec", "aac"]),
}


def _transcode(path: str, container: str, *, bitrate: str, timeout: int,
               output_path: str | None) -> str | None:
    """Shared encode behind the per-container helpers below."""
    import shutil as _shutil
    ffmpeg = _shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    muxer, encoder_args = _VOICE_ENCODERS[container]
    suffix = f".{container}"
    if output_path is None:
        fd, out_path = tempfile.mkstemp(prefix="voice_transcode_", suffix=suffix)
        os.close(fd)
    else:
        out_path = output_path
    in_place = os.path.abspath(str(path)) == os.path.abspath(out_path)
    work_path = out_path + ".tmp" + suffix if in_place else out_path
    try:
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(path), *encoder_args,
             "-ac", "1", "-b:a", bitrate, "-f", muxer, work_path],
            capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL, check=False)
        if result.returncode == 0 and os.path.getsize(work_path) > 0:
            if in_place:
                os.replace(work_path, out_path)
            return out_path
        logger.warning("ffmpeg %s transcode of %s failed (returncode=%s): %s", container, path,
                       result.returncode, (result.stderr or b"").decode("utf-8", errors="replace")[:500])
    except Exception:
        logger.warning("voice transcode to %s failed for %s", container, path, exc_info=True)
    with contextlib.suppress(OSError):
        os.unlink(work_path)
    return None


def transcode_to_ogg_opus(path: str, *, bitrate: str = "32k", timeout: int = 60,
                          output_path: str | None = None) -> str | None:
    """Best-effort ffmpeg transcode to Ogg/Opus (voip-tuned) for native voice bubbles: the written
    ``.ogg`` path (a NEW temp file unless ``output_path`` is given; caller cleans up), or None when
    ffmpeg is missing/fails. ``output_path`` may equal ``path`` (in-place container repair) — the
    encode goes through a sidecar so a failed run never truncates the source. Blocking (to_thread)."""
    return _transcode(path, "ogg", bitrate=bitrate, timeout=timeout, output_path=output_path)


def transcode_to_m4a(path: str, *, bitrate: str = "64k", timeout: int = 60,
                     output_path: str | None = None) -> str | None:
    """``transcode_to_ogg_opus``'s twin for AAC in MP4, the container iMessage voice notes need."""
    return _transcode(path, "m4a", bitrate=bitrate, timeout=timeout, output_path=output_path)
