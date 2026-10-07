"""Best-effort accessors for the single-writer stream fence.

The fence lives on ``AIAgent`` (``_claim_stream_writer`` / ``_stream_writer_is_current``) but is
used from other streaming modules. Calling it directly would turn an *additive* safety net into a
fatal AttributeError on a partially-updated checkout, hot-reloaded gateway, duck-typed agent, or
test double (a cron job died this way). The fence may only drop a *provably* superseded stream,
never the sole writer, so when it is unavailable or raises the degradation is "no fence".
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def claim_stream_writer(agent: Any) -> int:
    """Claim the delta sink for this stream attempt; ``0`` (never fenced) when the agent lacks the fence or the claim raised."""
    return _fence_call(agent, "_claim_stream_writer", int, 0, "claim failed; proceeding unfenced")


def stream_writer_is_current(agent: Any, token: int) -> bool:
    """True when ``token`` is still the active writer; a falsy token or a fence-less agent cannot prove supersession, so True."""
    if not token:
        return True
    return _fence_call(agent, "_stream_writer_is_current", bool, True, "is_current check failed; treating as current", token)


def _fence_call(agent: Any, name: str, cast, fallback, failure_note: str, *args):
    """Call ``agent.<name>(*args)`` when it exists; ``fallback`` when missing or raising (logged at debug)."""
    fn = getattr(agent, name, None)
    if callable(fn):
        try:
            return cast(fn(*args))
        except Exception:
            logger.debug("stream single-writer: %s", failure_note, exc_info=True)
    return fallback
