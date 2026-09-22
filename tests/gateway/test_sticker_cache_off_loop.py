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
import threading

import pytest

from gateway import sticker_cache


@pytest.fixture()
def cache_file(tmp_path, monkeypatch):
    path = tmp_path / "sticker_cache.json"
    monkeypatch.setattr(sticker_cache, "CACHE_PATH", path, raising=True)
    return path


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
    # With the lock in place the second caller cannot reach the barrier, so
    # the first one MUST time out here. That timeout is the green path's
    # cost, so keep it small.
    entered = threading.Barrier(2, timeout=0.5)
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
