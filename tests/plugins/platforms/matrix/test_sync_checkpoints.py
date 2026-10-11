"""A written sync cursor keeps the admitted IDs of every response that it does not cover."""

import asyncio
from types import SimpleNamespace

import pytest

from plugins.platforms.matrix.sync_transport import (
    DurableSyncStore,
    SyncCheckpoints,
    SyncDispatch,
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


@pytest.mark.asyncio
async def test_retried_dispatch_keeps_ids_admitted_by_the_failed_attempt(tmp_path):
    store = _store(tmp_path)
    await store.load()
    to_device_started, to_device_release = asyncio.Event(), asyncio.Event()

    async def slow_to_device():
        to_device_started.set()
        await to_device_release.wait()

    class Client:
        sync_store = store

        def handle_sync(self, response):
            if "rooms" not in response:
                dispatch.own_tasks([asyncio.create_task(slow_to_device())])

    dispatch = SyncDispatch(Client())

    async def admitted(_event):
        return True

    dispatch.intake_handlers = {admitted}
    checkpoints = SyncCheckpoints(store, dispatch.dispatching_intakes)
    first = asyncio.get_running_loop().create_future()
    await checkpoints.commit("s2", frozenset({"$first"}), (("$first", first),))

    # The first attempt at s3 admits $r, then a sibling fails, so take_intakes never runs.
    await dispatch._catch_errors(admitted, SimpleNamespace(event_id="$r"))
    retry = asyncio.create_task(dispatch.dispatch_sync({"to_device": {"events": []}}))
    try:
        await asyncio.wait_for(to_device_started.wait(), timeout=2)
        first.set_result(True)
        await asyncio.wait_for(checkpoints.settled(), timeout=2)
        persisted = await _persisted(tmp_path)
    finally:
        to_device_release.set()
        await asyncio.wait_for(retry, timeout=2)

    assert persisted == ("s2", {"$r"})
