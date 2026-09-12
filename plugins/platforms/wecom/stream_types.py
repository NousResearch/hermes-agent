"""Backward-compat re-export shim for the fork's ``stream_types`` module.

The stream-protocol data types now live in ``streaming.py`` (merged there to follow the upstream
mixin layout). Tests and any external code that imported them from ``plugins.platforms.wecom.stream_types``
keep working via this shim."""
from __future__ import annotations

from plugins.platforms.wecom.streaming import (  # noqa: F401
    StreamFrameResult,
    StreamSendOutcome,
    StreamTurn,
    WeComStreamExpiredError,
    STREAM_EXPIRED_ERRCODE,
)

__all__ = [
    "StreamFrameResult",
    "StreamSendOutcome",
    "StreamTurn",
    "WeComStreamExpiredError",
    "STREAM_EXPIRED_ERRCODE",
]
