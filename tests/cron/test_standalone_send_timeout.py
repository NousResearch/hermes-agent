"""Standalone cron delivery must be bounded (#38780 / PR #38792).

``_standalone_send`` runs the platform send via ``asyncio.run``; before this fix a
send that never returned (dead transport, hung SDK) wedged cron finalization
indefinitely. The sender coroutine is now wrapped in ``asyncio.wait_for``:

- text sends get the 30s standalone budget; media sends use the configured
  media-send timeout (they legitimately exceed 30s);
- a timeout is reported as "delivery outcome unconfirmed" — the send was
  cancelled at the deadline, but it may still land server-side;
- on the pre-start ``RuntimeError`` path (asyncio.run refuses inside a running
  loop) the closed coroutine releases the wait_for wrapper with it, so the
  thread fallback cannot leave an unawaited wrapper behind;
- the thread fallback keeps the context-preserving copy_context() routing and
  gives the inner wait_for a 5s head start over the hard future deadline.
"""

import asyncio
import contextvars
import gc
import threading
import warnings
from types import SimpleNamespace
from typing import Any, cast

from cron import scheduler_delivery


def _target() -> Any:
    return cast(
        Any,
        SimpleNamespace(
            job={"id": "timeout-test"},
            platform="telegram",
            platform_name="telegram",
            chat_id="42",
            thread_id=None,
            pconfig=None,
            where="telegram:42",
        ),
    )


def test_hung_standalone_send_times_out_with_unconfirmed_outcome(monkeypatch):
    """A send that never returns is cancelled at the deadline and surfaces an
    explicit unconfirmed-outcome error instead of blocking forever."""
    cancelled = threading.Event()

    async def hang(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", hang)
    monkeypatch.setattr(scheduler_delivery, "_STANDALONE_SEND_TIMEOUT_SECONDS", 0.05)

    result, error = scheduler_delivery._standalone_send(_target(), "payload", [])

    assert result is None
    assert "timed out" in error and "unconfirmed" in error
    # The hung sender itself was cancelled, not just abandoned.
    assert cancelled.wait(timeout=3)


def test_running_loop_falls_back_to_thread_and_times_out(monkeypatch):
    """Inside a running loop the pre-start RuntimeError retry runs in a worker
    thread; the timeout must hold there too, with the profile context intact."""
    scope = contextvars.ContextVar("delivery_test_scope")
    seen = []
    cancelled = threading.Event()

    async def hang(*args, **kwargs):
        seen.append(scope.get())
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", hang)
    monkeypatch.setattr(scheduler_delivery, "_STANDALONE_SEND_TIMEOUT_SECONDS", 0.05)

    outcome = []

    def thread_main():
        scope.set("profile-under-test")

        async def call():
            return scheduler_delivery._standalone_send(_target(), "payload", [])

        outcome.append(asyncio.run(call()))

    worker = threading.Thread(target=thread_main, daemon=True)
    worker.start()
    try:
        worker.join(timeout=5)
        assert not worker.is_alive(), "fallback delivery exceeded its bounded deadline"
        result, error = outcome[0]
        assert result is None
        assert "timed out" in error
        assert cancelled.wait(timeout=3)
        # copy_context() routing: the fallback thread saw the caller's profile scope.
        assert seen == ["profile-under-test"]
    finally:
        cancelled.set()
        worker.join(timeout=5)


def test_prestart_rejection_releases_wrapper_and_fallback_delivers(monkeypatch):
    """The loop-refusal path must not leave an unawaited wait_for wrapper, and
    the thread fallback must still deliver the send exactly once."""
    calls = []

    async def ok_send(*args, **kwargs):
        calls.append(True)
        return {"success": True}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", ok_send)

    async def call():
        return scheduler_delivery._standalone_send(_target(), "payload", [])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, error = asyncio.run(call())
        gc.collect()

    assert error is None
    assert result == {"success": True}
    # First attempt never started its coroutine; only the fallback sent.
    assert calls == [True]
    assert not [w for w in caught if "was never awaited" in str(w.message)]


def test_media_send_uses_configured_media_timeout(monkeypatch):
    """Media deliveries resolve their budget from the media-send timeout
    config, not the 30s text budget."""
    media_timeout_calls = []

    async def ok_send(*args, **kwargs):
        return {"success": True}

    def media_timeout():
        media_timeout_calls.append(True)
        return 77

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", ok_send)
    monkeypatch.setattr("cron.scheduler_script._get_media_send_timeout", media_timeout)

    result, error = scheduler_delivery._standalone_send(
        _target(), "with media", [("clip.mp3", False)]
    )

    assert error is None
    assert result == {"success": True}
    assert media_timeout_calls == [True]
