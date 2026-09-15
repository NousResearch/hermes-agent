import asyncio
import contextvars
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest


@pytest.mark.parametrize("inside_loop", [False, True])
@pytest.mark.parametrize("has_media", [False, True])
def test_standalone_timeout_cancels_sender_and_preserves_context(
    monkeypatch, inside_loop, has_media
):
    from cron import scheduler_delivery as delivery
    from tools import send_message_tool

    release = threading.Event()
    cancelled = threading.Event()
    scope = contextvars.ContextVar("delivery_test_scope")
    seen = []
    outcome = []
    media_timeout_calls = []

    async def send(*args, **kwargs):
        seen.append(scope.get())
        try:
            while not release.is_set():
                await asyncio.sleep(0.01)
            return {"success": True}
        finally:
            cancelled.set()

    def get_media_timeout():
        media_timeout_calls.append(True)
        return 0.05

    monkeypatch.setattr(send_message_tool, "_send_to_platform", send)
    monkeypatch.setattr(delivery, "_STANDALONE_SEND_TIMEOUT_SECONDS", 0.05, raising=False)
    monkeypatch.setattr(delivery._script, "_get_media_send_timeout", get_media_timeout)
    target = cast(Any, SimpleNamespace(
        job={"id": "timeout-test"}, where="test:local", platform="test",
        pconfig=None, chat_id="local", thread_id=None,
    ))
    media_files = [("clip.mp3", False)] if has_media else []

    def run():
        scope.set("profile-under-test")

        async def nested():
            return delivery._standalone_send(target, "test payload", media_files)

        outcome.append(asyncio.run(nested()) if inside_loop else
                       delivery._standalone_send(target, "test payload", media_files))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        worker.join(timeout=3)
        assert not worker.is_alive(), "standalone delivery exceeded its bounded deadline"
        result, error = outcome[0]
        assert result is None
        assert "timed out" in error
        assert cancelled.wait(timeout=3)
        assert seen == ["profile-under-test"]
        assert media_timeout_calls == ([True] if has_media else [])
    finally:
        release.set()
        worker.join(timeout=3)
