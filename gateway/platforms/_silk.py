"""Shared audio decoding helpers for gateway platform adapters.

Tencent's IM products (Weixin/WeChat, QQ) deliver voice messages in SILK
v3, a proprietary codec ffmpeg cannot decode. The community standard for
decoding SILK in Python is ``pilk`` (wraps Tencent's official SILK SDK).

This module provides:

* :func:`looks_like_silk` — magic-byte detection
* :func:`silk_to_wav`    — SILK → 16kHz mono WAV via ``pilk``
* :func:`ffmpeg_to_wav`  — generic audio → 16kHz mono WAV via ``ffmpeg``
* :func:`ensure_wav`     — try SILK first if applicable, fall back to ffmpeg

Both helpers return ``None`` on failure (never raise) so callers can chain
fallbacks naturally. Used by ``weixin.py`` (inbound voice STT) and
``qqbot/adapter.py`` (inbound voice STT).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_SILK_MAGIC_PREFIXES = (
    b"#!SILK_V3",
    b"#!SILK",
    b"\x02!",
    b"\x02#!SILK",  # Weixin variant: 0x02 framing byte before #!SILK
)

# Minimum size in bytes for a WAV file to count as "non-empty" (header + samples)
_MIN_VALID_WAV_BYTES = 45  # 44-byte RIFF header + at least 1 sample byte


def looks_like_silk(data: bytes) -> bool:
    """Return True if ``data`` starts with a known SILK v3 magic prefix."""
    if not data:
        return False
    return any(data.startswith(prefix) for prefix in _SILK_MAGIC_PREFIXES)


def silk_to_wav(
    src_path: str,
    wav_path: str,
    *,
    rate: int = 16000,
    log_tag: str = "",
) -> Optional[str]:
    """Convert a SILK audio file to WAV using the ``pilk`` library.

    pilk inspects the file extension. If ``src_path`` doesn't end in
    ``.silk`` we copy it to a temporary ``.silk`` path before invoking
    pilk, then clean up.

    Returns ``wav_path`` on success, ``None`` on any failure (missing
    ``pilk``, decode error, empty output). Never raises.
    """
    try:
        import pilk  # type: ignore
    except ImportError:
        logger.warning(
            "%spilk not installed — cannot decode SILK audio. "
            "Run: pip install pilk",
            f"[{log_tag}] " if log_tag else "",
        )
        return None

    src = Path(src_path)
    if not src.exists():
        return None

    # Try as-is first (cheap path when src already ends in .silk)
    try:
        pilk.silk_to_wav(str(src), wav_path, rate=rate)
        if _wav_nonempty(wav_path):
            return wav_path
    except Exception as exc:
        logger.debug(
            "%spilk direct conversion failed for %s: %s",
            f"[{log_tag}] " if log_tag else "",
            src.name,
            exc,
        )

    # Fall back: copy to .silk-suffixed path so pilk's extension check passes
    if src.suffix.lower() != ".silk":
        silk_path = str(src.with_suffix(".silk"))
        try:
            shutil.copy2(str(src), silk_path)
            pilk.silk_to_wav(silk_path, wav_path, rate=rate)
            if _wav_nonempty(wav_path):
                return wav_path
        except Exception as exc:
            logger.debug(
                "%spilk .silk-suffixed conversion failed for %s: %s",
                f"[{log_tag}] " if log_tag else "",
                src.name,
                exc,
            )
        finally:
            try:
                os.unlink(silk_path)
            except OSError:
                pass

    return None


async def ffmpeg_to_wav(
    src_path: str,
    wav_path: str,
    *,
    rate: int = 16000,
    timeout: float = 30.0,
    log_tag: str = "",
) -> Optional[str]:
    """Convert any ffmpeg-decodable audio file to 16kHz mono WAV.

    Returns ``wav_path`` on success, ``None`` on any failure (missing
    ffmpeg binary, decode error, timeout, empty output). Never raises.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i", src_path,
            "-ar", str(rate),
            "-ac", "1",
            wav_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning(
            "%sffmpeg not installed — cannot decode audio via ffmpeg fallback",
            f"[{log_tag}] " if log_tag else "",
        )
        return None

    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "%sffmpeg timed out converting %s",
            f"[{log_tag}] " if log_tag else "",
            Path(src_path).name,
        )
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        return None

    if proc.returncode != 0:
        stderr = b""
        if proc.stderr is not None:
            try:
                stderr = await proc.stderr.read()
            except Exception:
                pass
        logger.warning(
            "%sffmpeg failed for %s: %s",
            f"[{log_tag}] " if log_tag else "",
            Path(src_path).name,
            stderr[:200].decode(errors="replace"),
        )
        return None

    if not _wav_nonempty(wav_path):
        logger.warning(
            "%sffmpeg produced no/empty output for %s",
            f"[{log_tag}] " if log_tag else "",
            Path(src_path).name,
        )
        return None

    return wav_path


async def ensure_wav(
    src_path: str,
    *,
    wav_path: Optional[str] = None,
    sniff_bytes: Optional[bytes] = None,
    rate: int = 16000,
    log_tag: str = "",
) -> Optional[str]:
    """Produce a 16kHz mono WAV from ``src_path``, trying SILK then ffmpeg.

    Detection order:

    1. If ``sniff_bytes`` starts with a SILK magic prefix, or the file
       extension is ``.silk``, try ``pilk`` first.
    2. Otherwise (or if pilk fails), try ``ffmpeg``.

    ``wav_path`` defaults to ``<src_path stem>.wav`` next to the source.

    Returns the wav path on success, ``None`` if every backend failed.
    Callers can fall back to handing the original audio to a cloud STT
    provider that may accept the source format natively.
    """
    src = Path(src_path)
    if not src.exists():
        return None

    if wav_path is None:
        wav_path = str(src.with_suffix(".wav"))

    is_silk = False
    if sniff_bytes is not None and looks_like_silk(sniff_bytes):
        is_silk = True
    elif src.suffix.lower() == ".silk":
        is_silk = True
    else:
        # Cheap header sniff if no bytes given
        try:
            with open(src_path, "rb") as fh:
                head = fh.read(16)
            is_silk = looks_like_silk(head)
        except OSError:
            pass

    if is_silk:
        result = await asyncio.to_thread(
            silk_to_wav, src_path, wav_path, rate=rate, log_tag=log_tag,
        )
        if result:
            return result

    result = await ffmpeg_to_wav(
        src_path, wav_path, rate=rate, log_tag=log_tag,
    )
    return result


def _wav_nonempty(wav_path: str) -> bool:
    try:
        return Path(wav_path).exists() and Path(wav_path).stat().st_size > _MIN_VALID_WAV_BYTES
    except OSError:
        return False
