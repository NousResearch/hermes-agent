"""A written sync cursor keeps the admitted IDs of every response that it does not cover."""

import asyncio
import pytest

from plugins.platforms.matrix.sync_transport import (
    DurableSyncStore,
    SyncCheckpoints,
)


def _store(path):
    return DurableSyncStore(path, "https://hs", "@bot:hs", "DEV", "token")


async def _persisted(path):
    store = _store(path)
    await store.load()
    return store._next_batch, store._accepted_events


async def _spin(times=20):
    for _ in range(times):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cancel_while_the_cursor_write_waits_for_the_lock_keeps_queued_admissions(tmp_path):
    store = _store(tmp_path)
    await store.load()
    await store.accept_intake("$later")
    first = asyncio.get_running_loop().create_future()
    checkpoints = SyncCheckpoints(store)
    await checkpoints.commit("s2", frozenset({"$first"}), (("$first", first),))
    await checkpoints.commit("s3", frozenset({"$later"}), ())

    # Another intake write has acquired the lock, for example a follow-up admitted by the
    # registration replay, which disconnect does not await.
    await store._write_lock.acquire()
    first.set_result(True)
    await _spin()
    cancelling = asyncio.create_task(checkpoints.cancel())
    await _spin()
    store._write_lock.release()
    await asyncio.wait_for(cancelling, timeout=2)

    assert await _persisted(tmp_path) == ("s2", {"$later"})

