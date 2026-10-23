"""Socket read timeouts derived from the stale-stream budget (streaming and inline non-streaming).

A stale kill can only ``shutdown()`` the worker's socket from the monitor thread: ``close()`` from a
stranger thread recycles a live descriptor under the owner's SSL BIO (#29507). On Windows a
``shutdown()`` never wakes a blocked ``recv`` (only ``close()`` does), so the read timeout is the only
thing that ends a hung read there, and it has to expire within the stale budget.
"""

import math


def _inline_nonstream_hard_timeout(stale_timeout: float):
    """Socket-level backstop for inline non-streaming calls (#85252): the keepalive
    client uses ``read=None`` and the stranger-thread abort must not ``close()`` the
    FD (#29507), so a hung provider otherwise waits until TCP dies. Returns an
    ``httpx.Timeout`` with read == stale budget, a float if httpx is unavailable,
    or ``None`` when the watchdog is disarmed (non-finite budget)."""
    if not math.isfinite(stale_timeout) or stale_timeout <= 0:
        return None
    conn_cap = min(stale_timeout, 60.0)
    try:
        import httpx as _httpx
        return _httpx.Timeout(connect=conn_cap, read=stale_timeout, write=conn_cap, pool=conn_cap)
    except Exception:
        return stale_timeout


def _read_within_stale_budget(read: float, stale: float | None) -> float:
    """Cap a stream's socket read timeout at the stale detector's budget, so a stale kill whose
    ``shutdown()`` cannot wake the reader (Windows) still retries on time instead of waiting out
    the longer read timeout. A disarmed detector (non-finite) leaves *read* alone."""
    if stale is None or not math.isfinite(stale) or stale <= 0:
        return read
    return min(read, stale)
