"""The test helpers themselves: a helper that hides a hung reporter would let every test above pass."""

from __future__ import annotations

import asyncio

import pytest

from tests.gateway.status_report.support import settle

pytestmark = pytest.mark.asyncio


async def test_settle_fails_when_a_task_is_still_running_at_its_timeout():
    hung = asyncio.get_running_loop().create_task(asyncio.Event().wait())
    try:
        with pytest.raises(AssertionError, match="still running"):
            await settle(timeout=0.05)
    finally:
        hung.cancel()
        await asyncio.gather(hung, return_exceptions=True)


async def test_settle_returns_once_every_task_is_done():
    done = asyncio.get_running_loop().create_task(asyncio.sleep(0))

    await settle(timeout=5)

    assert done.done()
