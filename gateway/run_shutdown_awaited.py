"""Inbound requests whose caller is blocked on the reply, counted for the stop drain.

An adapter opts in with a ``pending_reply_count()`` hook (A2A: ``message/send``, ``message/stream``
and forwards to another profile). Such a turn sits in ``_running_agents`` like a chat turn, but it
does not resume for its caller: a killed one is a failed task for the remote agent, as a killed /v1
run is (#132989), so the drain gives it the cron floor instead of the 0s chat budget.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("gateway.run")


def awaited_reply_count(runner: Any) -> int:
    """Sum of ``pending_reply_count()`` over the runner's adapters, primary and served profiles'."""
    adapters = list((getattr(runner, "adapters", None) or {}).values())
    for profile_adapters in (getattr(runner, "_profile_adapters", None) or {}).values():
        adapters.extend(profile_adapters.values())
    total = 0
    for adapter in {id(a): a for a in adapters}.values():
        hook = getattr(adapter, "pending_reply_count", None)
        if not callable(hook):
            continue
        try:
            count = hook()
        except Exception:  # a broken adapter must not crash or stall the stop drain
            logger.debug("pending_reply_count failed on %r", adapter, exc_info=True)
            continue
        if isinstance(count, int) and not isinstance(count, bool):
            total += max(0, count)
    return total
