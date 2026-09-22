"""The sticker-description cache write must not block the event loop.

``gateway.sticker_cache._save_cache`` ends in ``os.replace``, whose duration is
unbounded under filesystem pressure.  Its only production caller is Telegram's
``_handle_sticker`` -- an inbound-message coroutine -- so every sticker that
missed the cache paid that rename inline on the loop, stalling every other
adapter and every in-flight turn in the process.

These tests use no wall-clock thresholds.  The witness is ORDERING: the rename
is held open on a barrier released only by a background timer, and the sibling
task must have ticked BEFORE that release.  A companion gate-proof drives the
identical barrier through the synchronous form and asserts the loop DOES
starve, so the liveness assertion cannot pass vacuously.
"""
import asyncio
import json
import os
import threading

import pytest

from gateway import sticker_cache


@pytest.fixture()
def cache_file(tmp_path, monkeypatch):
    path = tmp_path / "sticker_cache.json"
    monkeypatch.setattr(sticker_cache, "CACHE_PATH", path, raising=True)
    return path


class _HeldReplace:
    """Replace ``os.replace`` with one that blocks until released."""

    def __init__(self, monkeypatch):
        self._gate = threading.Event()
        self._entered = threading.Event()
        self._real = os.replace
        monkeypatch.setattr(os, "replace", self._blocking, raising=True)

    def _blocking(self, src, dst, *a, **kw):
        self._entered.set()
        self._gate.wait(timeout=10.0)
        return self._real(src, dst, *a, **kw)

    def wait_until_entered(self, timeout=5.0):
        return self._entered.wait(timeout)

    def release(self):
        self._gate.set()


def test_the_cache_write_does_not_block_the_loop(cache_file, monkeypatch):
    """A stalled rename must not stop the loop from running other tasks."""

    async def scenario():
        held = _HeldReplace(monkeypatch)
        released = threading.Event()
        order: dict[str, bool] = {}

        async def sibling():
            await asyncio.sleep(0)
            # Captured AT TICK TIME: was the rename still being held?
            order["ticked_before_release"] = not released.is_set()

        write = asyncio.create_task(
            sticker_cache.cache_sticker_description_async(
                "uid_loop", "A cat waving", emoji="🐱", set_name="Cats"
            )
        )
        sibling_task = asyncio.create_task(sibling())

        def _release_later():
            released.set()
            held.release()

        releaser = threading.Timer(1.0, _release_later)
        releaser.start()
        try:
            await asyncio.wait_for(write, timeout=10.0)
        finally:
            releaser.cancel()

        await sibling_task
        assert held.wait_until_entered(), "the write never reached os.replace"
        assert "ticked_before_release" in order, "the sibling task never ran"
        assert order["ticked_before_release"], (
            "the loop did not advance the sibling task until the held rename "
            "was released -- the write is still blocking the event loop"
        )

        # Durability is preserved, not traded away for liveness.
        record = sticker_cache.get_cached_description("uid_loop")
        assert record["description"] == "A cat waving"
        assert record["emoji"] == "🐱"
        assert record["set_name"] == "Cats"
        assert json.loads(cache_file.read_text())["uid_loop"] == record

    asyncio.run(scenario())


def test_gate_proof_the_sync_form_does_block_the_loop(cache_file, monkeypatch):
    """The liveness test above is not vacuous.

    Call the SYNC form -- the pre-fix shape at the ``_handle_sticker`` call
    site -- from inside a coroutine and the very same barrier starves the loop.
    """

    async def scenario():
        held = _HeldReplace(monkeypatch)
        ticked = threading.Event()
        released = threading.Event()

        async def sibling():
            await asyncio.sleep(0)
            ticked.set()

        asyncio.create_task(sibling())

        def _release_later():
            released.set()
            held.release()

        releaser = threading.Timer(0.5, _release_later)
        releaser.start()
        try:
            sticker_cache.cache_sticker_description(
                "uid_inline", "A cat waving", emoji="🐱"
            )
        finally:
            releaser.cancel()

        assert released.is_set(), (
            "the inline write returned without the rename having been held; "
            "the barrier did not engage, so this proves nothing"
        )
        assert not ticked.is_set(), (
            "the sibling task ran while the inline rename was held. The "
            "inline call is supposed to starve the loop; if it no longer "
            "does, the liveness test above proves nothing."
        )

    asyncio.run(scenario())


def test_the_write_runs_off_the_loop_thread(cache_file, monkeypatch):
    """The rename must execute on a worker thread, not the loop thread."""
    seen: dict[str, int] = {}
    real_save = sticker_cache._save_cache

    def _record(cache):
        seen["thread"] = threading.get_ident()
        return real_save(cache)

    monkeypatch.setattr(sticker_cache, "_save_cache", _record, raising=True)

    async def scenario():
        seen["loop"] = threading.get_ident()
        await sticker_cache.cache_sticker_description_async("uid_thread", "A dog")

    asyncio.run(scenario())
    assert seen["thread"] != seen["loop"], (
        "the cache write ran on the event-loop thread; the off-loop dispatch "
        "is not in effect"
    )


def test_the_async_wrapper_preserves_the_sync_contract(cache_file):
    """Same stored record either way -- the sync form is still the contract."""
    sticker_cache.cache_sticker_description(
        "uid_sync", "A happy dog", emoji="🐕", set_name="Dogs"
    )
    sync_record = dict(sticker_cache.get_cached_description("uid_sync"))

    asyncio.run(
        sticker_cache.cache_sticker_description_async(
            "uid_async", "A happy dog", emoji="🐕", set_name="Dogs"
        )
    )
    async_record = dict(sticker_cache.get_cached_description("uid_async"))

    sync_record.pop("cached_at")
    async_record.pop("cached_at")
    assert sync_record == async_record


def test_concurrent_writes_do_not_lose_an_entry(cache_file, monkeypatch):
    """Off-loop dispatch removed the loop's accidental serialization.

    ``cache_sticker_description`` is a read-modify-write: ``_load_cache`` ->
    merge -> ``_save_cache``.  ``atomic_json_write`` makes each WRITE atomic;
    it does not make the TRIPLE atomic.  While the call was inline on the loop,
    the loop serialized every caller and the race could not be observed.
    Moving it to a worker thread introduces real concurrency, so the lock has
    to arrive in the SAME change -- otherwise two stickers described at once
    silently drop one of the two descriptions.

    Asserts on the DURABLE FILE, not on the API.
    """
    # 2.0s: with the lock in place the second caller cannot reach the barrier,
    # so the first one MUST time out here. That timeout is the green path's
    # cost, so keep it small.
    entered = threading.Barrier(2, timeout=2.0)
    real_load = sticker_cache._load_cache
    done = threading.Event()

    def _interleaving_load():
        data = real_load()
        # Force both callers to observe the SAME pre-state, which is what a
        # lost update requires. Only the first two callers rendezvous.
        if not done.is_set():
            try:
                entered.wait()
            except threading.BrokenBarrierError:
                pass
        return data

    monkeypatch.setattr(
        sticker_cache, "_load_cache", _interleaving_load, raising=True
    )

    async def scenario():
        try:
            await asyncio.gather(
                sticker_cache.cache_sticker_description_async("uid_a", "A"),
                sticker_cache.cache_sticker_description_async("uid_b", "B"),
            )
        finally:
            done.set()

    asyncio.run(scenario())

    durable = json.loads(cache_file.read_text())
    assert sorted(durable) == ["uid_a", "uid_b"], (
        "a concurrent read-modify-write lost an entry from the durable cache "
        f"file: keys = {sorted(durable)}. atomic_json_write makes each write "
        "atomic, not the load/merge/save triple."
    )
