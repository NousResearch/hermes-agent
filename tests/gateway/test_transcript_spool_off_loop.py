"""The transcript cap-drop spool write must not block the event loop.

``SessionStore._append_to_transcript_serialized`` calls
``spool_dropped_transcript_message`` inline whenever the per-session pending
cap evicts a message.  That call ends in the ``mkstemp`` + ``fsync`` +
``os.replace`` tail of the atomic writer, whose duration is unbounded under
filesystem pressure -- and the append path is driven synchronously by inbound
message coroutines (Telegram ``_handle_text_message`` /
``_handle_media_message`` / ``_handle_location_message``, the runner's
``_handle_message*``).  A rename several plain-def frames below a
coroutine stalls every other task in the process.

Measured on the pre-fix shape with ``os.replace`` held for 0.30s, the on-loop
append took 0.318s.  These tests pin the fix without wall-clock thresholds:
they hold the rename open on a real barrier and assert the loop makes
progress anyway, and they pin the durability and ordering properties the
off-loop lane must not trade away.
"""
import asyncio
import json
import os
import threading

import pytest

from gateway import shutdown_flush


@pytest.fixture()
def spool_home(tmp_path, monkeypatch):
    """Point the pending spool at an isolated HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path, raising=True)
    monkeypatch.setattr(hermes_constants, "assert_named_profile_home_live", lambda *_a, **_k: None, raising=False)
    return tmp_path


@pytest.fixture(autouse=True)
def _drain_lane_between_tests():
    """No test may leave work queued for the next one."""
    yield
    shutdown_flush.fence_spool_lane(timeout=10.0)


def _spool_files(home):
    d = home / "pending_messages"
    return sorted(d.glob("pending-*.json")) if d.exists() else []


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


def test_spool_write_does_not_block_the_loop(spool_home, monkeypatch):
    """A stalled rename must not stop the loop from running other tasks.

    No stopwatch: the rename is held on a barrier for as long as the
    assertion needs, and the loop must still advance a sibling task.  On the
    pre-fix inline shape the loop cannot tick at all while the rename is
    held, so the wait below times out.
    """
    async def scenario():
        held = _HeldReplace(monkeypatch)
        ticked = asyncio.Event()

        async def sibling():
            # A task the loop can only run if it is not blocked.
            await asyncio.sleep(0)
            ticked.set()

        sibling_task = asyncio.create_task(sibling())
        result = shutdown_flush.spool_dropped_transcript_message(
            "sess-loop", {"role": "user", "content": "evicted"}
        )
        # The call returned while the rename is still held open.
        assert result is not None, "spooling must not be silently dropped"

        await asyncio.wait_for(ticked.wait(), timeout=5.0)
        assert held.wait_until_entered(), "the write never reached os.replace"

        held.release()
        await sibling_task
        # Durability is preserved: the payload lands once the write completes.
        await asyncio.to_thread(shutdown_flush.fence_spool_lane)
        files = _spool_files(spool_home)
        assert len(files) == 1
        payload = json.loads(files[0].read_text())
        assert payload["data"]["message"]["content"] == "evicted"
        assert payload["reason"] == shutdown_flush.TRANSCRIPT_CAP_DROP_REASON

    asyncio.run(scenario())


def test_gate_proof_inline_dispatch_does_block_the_loop(
    spool_home, monkeypatch
):
    """The test above is not vacuous.

    Force the loop-conditional dispatch to choose the inline branch -- the
    pre-fix shape -- and the very same barrier now starves the loop.
    """
    monkeypatch.setattr(
        shutdown_flush, "_loop_is_running", lambda: False, raising=True
    )

    async def scenario():
        held = _HeldReplace(monkeypatch)
        ticked = asyncio.Event()

        async def sibling():
            await asyncio.sleep(0)
            ticked.set()

        asyncio.create_task(sibling())
        releaser = threading.Timer(0.5, held.release)
        releaser.start()
        try:
            shutdown_flush.spool_dropped_transcript_message(
                "sess-inline", {"role": "user", "content": "evicted"}
            )
        finally:
            releaser.cancel()
            held.release()
        # The loop could not run the sibling while the write was inline.
        assert not ticked.is_set(), (
            "the inline write did not block the loop; the barrier is not "
            "actually holding the rename, so the sibling test proves nothing"
        )

    asyncio.run(scenario())


def test_no_running_loop_still_writes_inline(spool_home):
    """Off-loop callers (shutdown, workers) keep the synchronous contract."""
    result = shutdown_flush.spool_dropped_transcript_message(
        "sess-sync", {"role": "user", "content": "inline"}
    )
    assert result is not None
    assert result.exists(), "a non-loop caller must get a real written path"
    assert _spool_files(spool_home) == [result]


def test_drain_fences_queued_writes_so_none_are_resurrected(
    spool_home, monkeypatch
):
    """A queued write must land BEFORE the replay scans the directory.

    Otherwise the lane publishes a file after the drain that already returned,
    and the message reappears on a later drain as a duplicate of one the
    caller believes it replayed.
    """
    async def queue_it():
        held = _HeldReplace(monkeypatch)
        shutdown_flush.spool_dropped_transcript_message(
            "sess-drain", {"role": "user", "content": "queued"}
        )
        assert held.wait_until_entered()
        # Release on another thread so the drain below genuinely has to wait.
        threading.Timer(0.2, held.release).start()

    asyncio.run(queue_it())

    seen = []
    replayed, remaining = shutdown_flush.drain_transcript_spool(
        "sess-drain", lambda m: seen.append(m["content"])
    )
    assert seen == ["queued"], (
        "the drain scanned the spool directory before the queued lane write "
        "landed, so the message would be resurrected after replay"
    )
    assert replayed == 1
    assert remaining == 0
    assert _spool_files(spool_home) == []


def test_lane_preserves_drop_order(spool_home):
    """Replay order is drop order, even though writes happen off-thread."""
    async def scenario():
        for i in range(5):
            shutdown_flush.spool_dropped_transcript_message(
                "sess-order", {"role": "user", "content": f"c{i}"}
            )

    asyncio.run(scenario())

    seen = []
    replayed, remaining = shutdown_flush.drain_transcript_spool(
        "sess-order", lambda m: seen.append(m["content"])
    )
    assert replayed == 5
    assert remaining == 0
    assert seen == ["c0", "c1", "c2", "c3", "c4"]


def test_lane_write_failure_does_not_escape_to_the_caller(
    spool_home, monkeypatch
):
    """Spooling is best-effort; a lane failure must not break the caller."""
    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(shutdown_flush, "_write_payload", _boom, raising=True)

    async def scenario():
        shutdown_flush.spool_dropped_transcript_message(
            "sess-fail", {"role": "user", "content": "doomed"}
        )
        await asyncio.to_thread(shutdown_flush.fence_spool_lane)

    asyncio.run(scenario())
    assert _spool_files(spool_home) == []
