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

from typing import Any, Iterable, Mapping

from hermes_agent.gateway.platforms.helpers import (
    MEDIA_UPLOAD_DEADLINE_MAX as DEADLINE_MAX, MIN_MEDIA_UPLOAD_RATE as MIN_UPLOAD_RATE, media_upload_seconds,
    scaled_upload_deadline)

# sendVideo transcodes before answering, outlasting the 20s general read timeout.
MEDIA_READ_TIMEOUT = 60.0
MEDIA_WRITE_TIMEOUT = 60.0  # matches telegram_network.ptb_request_kwargs()["media_write_timeout"]
_MEDIA_KEYS = ("animation", "audio", "document", "photo", "sticker", "video", "video_note", "voice", "thumbnail")
__all__ = ["DEADLINE_MAX", "MIN_UPLOAD_RATE", "media_sources", "scaled_budgets"]


def media_sources(media_kwargs: Mapping[str, Any]) -> list:
    """The payload values of a Bot API media call (never its caption or filename)."""
    return [media_kwargs.get(key) for key in _MEDIA_KEYS]


def scaled_budgets(sources: Iterable[Any], deadline: float, read_timeout: float = MEDIA_READ_TIMEOUT) -> tuple[dict, float]:
    """``(per-request timeout kwargs, wall-clock deadline)`` for one upload of ``sources`` (paths,
    open files, URLs, file ids, bytes): the on-disk total at ``MIN_UPLOAD_RATE`` widens both."""
    extra = media_upload_seconds(sources)
    if extra <= 0:
        return {"read_timeout": read_timeout}, deadline
    return ({"read_timeout": read_timeout + extra, "write_timeout": MEDIA_WRITE_TIMEOUT + extra},
            scaled_upload_deadline(deadline, extra))
