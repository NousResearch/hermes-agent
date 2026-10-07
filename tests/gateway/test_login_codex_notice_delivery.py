"""Codex enrollment honors the shared notice policy and cancels timed-out delivery."""

import asyncio
import concurrent.futures
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from hermes_cli import auth_codex
from tests.gateway.test_login_codex import setup as _setup, _event, _finish

setup = _setup


def _device_code(monkeypatch):
    monkeypatch.setattr(auth_codex, "_codex_request_device_code", lambda *args: {
        "user_code": "TEST-CODE", "device_auth_id": "test-device", "interval": 5})
    poll = MagicMock(side_effect=RuntimeError("test-stop-after-delivery"))
    monkeypatch.setattr(auth_codex, "_codex_poll_authorization_code", poll)
    return poll


@pytest.mark.asyncio
@pytest.mark.parametrize("ignored", [False, True])
async def test_codex_verification_honors_notice_privacy_and_ignored_channels(setup, monkeypatch, ignored):
    runner, adapter, _, profile = setup
    del runner._deliver_platform_notice  # Exercise the real shared notice helper.
    adapter.config = SimpleNamespace(extra={
        "notice_delivery": "private", "ignored_channels": ["test-chat"] if ignored else []})
    adapter.send_private_notice = AsyncMock(return_value=SimpleNamespace(success=True))
    poll = _device_code(monkeypatch)
    assert "Codex sign-in started" in await runner._handle_login_command(_event(platform=Platform.SLACK))
    await _finish(runner)

    adapter.send.assert_not_awaited()
    if ignored:
        adapter.send_private_notice.assert_not_awaited()
        poll.assert_not_called()
    else:
        assert [call.args[2] for call in adapter.send_private_notice.await_args_list][:2] == [
            "https://auth.openai.com/codex/device", "TEST-CODE"]
        poll.assert_called_once()
    assert not (profile / "auth.json").exists()


@pytest.mark.asyncio
async def test_delivery_timeout_cancels_pending_verification_before_abort(setup, monkeypatch):
    runner, adapter, _, profile = setup
    del runner._deliver_platform_notice
    entered = threading.Event()
    stopped = asyncio.Event()

    async def stalled_send(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    adapter.send.side_effect = stalled_send
    # Failure copy uses the non-throwing renderer; isolate it from the stalled code send.
    runner._push_login = AsyncMock()
    poll = _device_code(monkeypatch)
    submit = asyncio.run_coroutine_threadsafe
    pending = []

    class DeliveryFuture:
        def __init__(self, future):
            self.future = future

        def result(self, timeout):
            assert timeout == 30
            assert entered.wait(5)
            raise concurrent.futures.TimeoutError()

        def cancel(self):
            return self.future.cancel()

    def submit_with_timeout(coro, loop):
        future = submit(coro, loop)
        pending.append(future)
        return DeliveryFuture(future)

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit_with_timeout)
    try:
        await runner._handle_login_command(_event())
        await _finish(runner)
        poll.assert_not_called()
        assert not (profile / "auth.json").exists()
        assert pending[0].cancelled()
        await asyncio.wait_for(stopped.wait(), timeout=5)
    finally:
        for future in pending:
            future.cancel()
