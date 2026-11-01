"""Regression test for the Discord ``stop_typing`` blocking-delivery bug.

``stop_typing`` previously did an unbounded ``await task`` after cancelling the
typing loop. Cancellation is only delivered at the loop's next await point; if
that await is an HTTP POST that ignores cancellation until the request
completes (the failure mode described in #64874 — a stuck ``/typing`` request
that only clears at the transport timeout), ``stop_typing`` blocks for the full
transport timeout and holds up message delivery.

The fix reaps the cancelled task with a bounded timeout
(``_TYPING_STOP_REAP_TIMEOUT``) and detaches if it does not finish promptly,
mirroring ``base._stop_typing_refresh``.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


def _make_adapter() -> DiscordAdapter:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=None))
    return adapter


@pytest.mark.asyncio
async def test_stop_typing_does_not_block_on_cancellation_immune_request() -> None:
    """``stop_typing`` returns promptly even when the loop ignores cancellation.

    Emulates a transport that swallows ``CancelledError`` mid-flight (the
    #64874 failure mode). ``asyncio.wait`` (not ``wait_for``) is used so the
    *test* still fails fast on the buggy code — wrapping ``stop_typing`` in
    ``wait_for`` would itself deadlock, because ``stop_typing``'s unbounded
    ``await task`` is stuck on the cancellation-immune loop and cannot be
    cancelled cleanly either.
    """
    adapter = _make_adapter()
    chat_id = "123456789012345678"

    entered = asyncio.Event()
    release = asyncio.Event()

    async def _cancellation_immune_request(_route) -> None:
        entered.set()
        # Swallow cancellation until the request "completes" (release set),
        # emulating an HTTP transport that ignores task cancellation.
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    adapter._client.http.request = _cancellation_immune_request

    await adapter.send_typing(chat_id)
    loop = adapter._typing_tasks[chat_id]
    assert loop is not None

    # Let the loop reach the cancellation-immune request.
    await asyncio.wait_for(entered.wait(), timeout=1.0)

    stop_task = asyncio.create_task(adapter.stop_typing(chat_id))
    done, _pending = await asyncio.wait({stop_task}, timeout=1.0)

    try:
        # stop_typing must detach within the reap bound (0.5s); the 1.0s
        # window here leaves margin while still failing fast on a regression.
        assert stop_task in done, (
            "stop_typing blocked on the cancellation-immune loop instead of "
            "detaching after the reap timeout"
        )
    finally:
        # Free the request so every in-flight task can unwind, then reap them.
        release.set()
        with suppress(asyncio.CancelledError, asyncio.TimeoutError, Exception):
            await asyncio.wait_for(stop_task, timeout=1.0)
        with suppress(asyncio.CancelledError, asyncio.TimeoutError, Exception):
            await asyncio.wait_for(loop, timeout=1.0)

    assert chat_id not in adapter._typing_tasks


@pytest.mark.asyncio
async def test_stop_typing_clears_registry_in_normal_path() -> None:
    """The non-hung path still clears the registry entry immediately."""
    adapter = _make_adapter()
    chat_id = "123456789012345678"

    async def _request(_route) -> None:
        return None

    adapter._client.http.request = _request

    await adapter.send_typing(chat_id)
    assert chat_id in adapter._typing_tasks

    await adapter.stop_typing(chat_id)
    assert chat_id not in adapter._typing_tasks
