"""Discord typing-indicator ownership: an earlier turn's loop teardown must not
deregister a later turn's loop.

``send_typing`` registers a per-channel refresh task in ``_typing_tasks`` and
``stop_typing`` cancels whatever it finds there. The loop also clears its own
registry entry in a ``finally``. Because cancellation is not instantaneous, a
cancelled loop can reach that ``finally`` AFTER the next turn has already armed
a fresh loop for the same channel — so an unconditional clear deregisters the
newer loop, which then keeps POSTing the typing indicator with no owner able to
stop it (a "typing…" bubble that never clears).

Behavioural tests against the real adapter methods, not snapshots.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, token="t")
    adapter._typing_tasks = {}

    # Count typing POSTs and yield control on each one so the test can
    # interleave turns deterministically.
    adapter.posts = 0

    async def _request(_route, *args, **kwargs):
        adapter.posts += 1
        await asyncio.sleep(0)

    client = MagicMock()
    client.http.request = _request
    adapter._client = client
    return adapter


@pytest.mark.asyncio
async def test_later_turns_loop_survives_an_earlier_turns_teardown():
    """Turn B arms while turn A's cancelled loop has not yet run its cleanup.
    Once A finishes unwinding, the registry must still point at B's loop —
    otherwise B is orphaned and nothing can ever stop it."""
    adapter = _make_adapter()

    await adapter.send_typing("chan")
    task_a = adapter._typing_tasks["chan"]
    await asyncio.sleep(0)  # let A enter its loop body

    # Turn A ends: cancel its loop and release the slot, but A has not yet
    # been scheduled to run its cleanup.
    task_a.cancel()
    adapter._typing_tasks.pop("chan", None)

    # Turn B arms in the same window.
    await adapter.send_typing("chan")
    task_b = adapter._typing_tasks["chan"]
    assert task_b is not task_a

    # Now let A finish unwinding.
    for _ in range(5):
        await asyncio.sleep(0)
    assert task_a.done()
    assert not task_b.done(), "turn B's loop should still be running"

    # The contract: B is still the registered owner for this channel.
    assert adapter._typing_tasks.get("chan") is task_b

    await adapter.stop_typing("chan")


@pytest.mark.asyncio
async def test_stop_typing_actually_stops_the_bubble_after_an_overlap():
    """The user-visible contract. After the same overlap, the owning turn's
    ``stop_typing`` must genuinely stop the indicator — the loop terminates and
    no further typing POSTs are issued."""
    adapter = _make_adapter()

    await adapter.send_typing("chan")
    task_a = adapter._typing_tasks["chan"]
    await asyncio.sleep(0)
    task_a.cancel()
    adapter._typing_tasks.pop("chan", None)

    await adapter.send_typing("chan")
    task_b = adapter._typing_tasks["chan"]
    for _ in range(5):
        await asyncio.sleep(0)

    # Turn B's owner stops typing, as the gateway does when the turn completes.
    await adapter.stop_typing("chan")
    assert task_b.done(), "the indicator loop must have terminated"

    posts_after_stop = adapter.posts
    for _ in range(10):
        await asyncio.sleep(0)
    assert adapter.posts == posts_after_stop, (
        "typing indicator kept POSTing after stop_typing — orphaned loop"
    )


@pytest.mark.asyncio
async def test_a_loop_that_ends_on_its_own_still_releases_the_channel():
    """Over-reach guard: ownership checking must not leak registry entries.
    A loop that is the registered owner when it exits must still clear its
    slot, or the duplicate-guard in ``send_typing`` would block every later
    turn on that channel from ever showing an indicator."""
    adapter = _make_adapter()

    await adapter.send_typing("chan")
    task = adapter._typing_tasks["chan"]
    await asyncio.sleep(0)

    task.cancel()
    for _ in range(5):
        await asyncio.sleep(0)

    assert task.done()
    assert "chan" not in adapter._typing_tasks

    # And a later turn can arm again on the same channel.
    await adapter.send_typing("chan")
    assert adapter._typing_tasks.get("chan") is not None
    await adapter.stop_typing("chan")


@pytest.mark.asyncio
async def test_stop_typing_clears_the_registry_for_the_normal_single_turn_path():
    """Control: the ordinary non-overlapping case is unchanged — arm, stop,
    slot released, loop terminated."""
    adapter = _make_adapter()

    await adapter.send_typing("chan")
    task = adapter._typing_tasks["chan"]
    await asyncio.sleep(0)

    await adapter.stop_typing("chan")

    assert task.done()
    assert "chan" not in adapter._typing_tasks
