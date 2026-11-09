"""Size-scaled budgets for one Telegram media upload, shared by the gateway adapter and the
standalone sender (#133093, #133164).

A Bot API server answers a ``sendDocument``/``sendVideo`` only after the whole body has reached
Telegram, so for a self-hosted server (``base_url``) on a slow uplink the time-to-first-byte is the
upstream transfer itself. Fixed budgets cancel such an upload while the server keeps draining it and
delivers the file minutes later, so the client reports a failure the user retries into a duplicate.
Payloads with an on-disk size get the transfer time at ``MIN_UPLOAD_RATE`` added to the read and
write budgets and 1.5x of it to the wall-clock deadline; URLs, file ids and in-memory buffers keep
the fixed budgets.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

# sendVideo transcodes before answering, outlasting the 20s general read timeout.
MEDIA_READ_TIMEOUT = 60.0
MEDIA_WRITE_TIMEOUT = 60.0  # matches telegram_network.ptb_request_kwargs()["media_write_timeout"]
MIN_UPLOAD_RATE = 128 * 1024  # B/s: the slowest uplink the budgets still cover
DEADLINE_MAX = 3600.0
_MEDIA_KEYS = ("animation", "audio", "document", "photo", "sticker", "video", "video_note", "voice", "thumbnail")


def _size_of(value: Any) -> int:
    if isinstance(value, (str, os.PathLike)):
        try:
            return os.stat(value).st_size
        except (OSError, ValueError):
            return 0  # a URL or file id
    try:  # an open file handle (the adapter and the standalone sender stream open files)
        return os.fstat(value.fileno()).st_size
    except (AttributeError, OSError, ValueError):
        return 0


def media_sources(media_kwargs: Mapping[str, Any]) -> list:
    """The payload values of a Bot API media call (never its caption or filename)."""
    return [media_kwargs.get(key) for key in _MEDIA_KEYS]


def scaled_budgets(sources: Iterable[Any], deadline: float, read_timeout: float = MEDIA_READ_TIMEOUT) -> tuple[dict, float]:
    """``(per-request timeout kwargs, wall-clock deadline)`` for one upload of ``sources`` (paths,
    open files, URLs, file ids, bytes): the on-disk total at ``MIN_UPLOAD_RATE`` widens both."""
    extra = sum(_size_of(source) for source in sources) / MIN_UPLOAD_RATE
    if extra <= 0:
        return {"read_timeout": read_timeout}, deadline
    return ({"read_timeout": read_timeout + extra, "write_timeout": MEDIA_WRITE_TIMEOUT + extra},
            max(deadline, min(DEADLINE_MAX, deadline + 1.5 * extra)))
