"""Async-delegation ledger calls from the gateway loop, run on a worker thread (#146507).

Split out of ``gateway/run_notifications.py``. Every ledger op opens ``state.db`` under the module
``_DB_LOCK`` and waits out its 10 s busy timeout, so on the event loop a busy ``state.db`` freezes
every adapter heartbeat. ``asyncio.to_thread`` copies the context, so the profile scope that picked
this ledger's ``state.db`` follows the hop.

A submitted op is never abandoned: on a cancel the caller waits for it to land and settles what it
claimed before re-raising, so a cancelled delivery cannot leave a row claimed until its lease ends.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Iterable

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

Settle = Callable[[str, str, str], None]


async def _landed(future: asyncio.Future) -> Any:
    """The result of a worker op the caller was cancelled out of, once it lands; None if it raised."""
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        raise
    # health: allow BLE001 -- the cancelled caller only needs to know whether the op claimed a row
    except Exception:
        logger.debug("Ledger op failed after its caller was cancelled", exc_info=True)
        return None


async def settle_claims(settle: Settle, ops: Iterable[tuple[str, str, str]]) -> None:
    """Run ``settle(kind, delegation_id, claim_id)`` for every op in one worker hop."""
    ops = [op for op in ops if op[2]]
    if not ops:
        return
    future = asyncio.ensure_future(asyncio.to_thread(lambda: [settle(*op) for op in ops]))
    try:
        await asyncio.shield(future)
    except asyncio.CancelledError:
        await _landed(future)
        raise


async def claim(settle: Settle, fn: Callable[[], Any], delegation_id: str, claim_id: str = "") -> Any:
    """Run one ledger claim off the loop and return its result. Truthy means claimed; the claim id is
    ``claim_id``, else the result itself. A cancel while it runs defers a row it claimed (refunding the
    attempt: nothing was delivered), then re-raises."""
    future = asyncio.ensure_future(asyncio.to_thread(fn))
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        if result := await _landed(future):
            await settle_claims(settle, [("defer", delegation_id, claim_id or str(result))])
        raise
