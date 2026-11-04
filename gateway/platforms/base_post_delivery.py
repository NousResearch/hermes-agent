"""Running a turn's one-shot post-delivery callback (``BasePlatformAdapter.register_post_delivery_callback``).

Shared by the adapter task that delivered a messaging reply (``_process_message_background``) and the
session authority's drain, which settles every turn no adapter task brackets (``session_ingress``).
"""
import asyncio
import contextlib
import inspect
from typing import Any


async def run_post_delivery_callback(callback: Any, timeout: float) -> None:
    """Run *callback* (a no-op when None): bounded, errors swallowed, so a stuck or failing callback
    never holds up the session whose reply it follows."""
    if callable(callback):
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            result = callback()
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=timeout)


async def run_processing_hook(adapter, hook_name, *args, **kwargs):
    """Isolate hook failures and claim an opt-in native execution's completion once."""
    if hook_name == 'on_processing_complete' and args:
        event = args[0]
        if getattr(event, '_native_completion_owner', None) is False:
            return
        actual = getattr(event, '_native_execution_event', event)
        if getattr(actual, '_native_execution_lifecycle', False):
            if getattr(actual, '_native_completion_reported', False):
                return
            actual._native_completion_reported = True
    hook = getattr(adapter, hook_name, None)
    if not callable(hook):
        return
    try:
        await hook(*args, **kwargs)
    except Exception as exc:
        import logging
        logging.getLogger('gateway.platforms.base').warning(
            '[%s] %s hook failed: %s', adapter.name, hook_name, exc, exc_info=True)
