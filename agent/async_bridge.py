"""Sync-to-async bridge for tool handlers.

Single source of truth for running async coroutines from synchronous code.
Extracted from model_tools.py to break the circular import between
model_tools.py and tools/registry.py.

Dependency direction:
    model_tools.py ──> agent/async_bridge.py <── tools/registry.py
    (No cycle. Both depend on this module; this module depends on neither.)
"""

import asyncio


def run_async(coro):
    """Run an async coroutine from a sync context.

    If the current thread already has a running event loop (e.g., inside
    the gateway's async stack or Atropos's event loop), we spin up a
    disposable thread so asyncio.run() can create its own loop without
    conflicting.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=300)
    return asyncio.run(coro)
