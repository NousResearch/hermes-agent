"""Slack teardown survives its own failures and its own cancellation (#147845).

The gateway disconnects adapters through a bounded wait that cancels ``disconnect()`` at its
deadline (``gateway/run_adapters._safe_adapter_disconnect`` -> ``_wait_or_detach``). Before the
teardown chain became best-effort, one workspace client whose ``close()`` raised — or one
``chat.stopStream`` the deadline cancelled — skipped every later step, including the scoped
platform-lock release, so the app token stayed owned by a process that was going away.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def _adapter(**state):
    """A connected-looking adapter: teardown runs the real chain against this state."""
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._running = True
    adapter._platform_lock_scope = "slack-app-token"
    adapter._platform_lock_identity = "xapp-fake"
    adapter._app_token = "xapp-fake"
    adapter._proxy_url = "http://proxy.example.com:3128"
    adapter._bot_user_id = "U_BOT"
    for key, value in state.items():
        setattr(adapter, key, value)
    return adapter


class _RaisingClient:
    """A workspace client whose close() raises, as the report's ``close-raise`` probe does."""

    def __init__(self, calls):
        self._calls = calls

    def close(self):
        self._calls.append("raise")
        raise RuntimeError("workspace client close failed")


class _RecordingClient:
    def __init__(self, calls):
        self._calls = calls

    def close(self):
        self._calls.append("closed")


class _HangingClient:
    """chat.stopStream never returns: the gateway's deadline cancels teardown right here."""

    def __init__(self, entered):
        self._entered = entered

    async def chat_stopStream(self, **kwargs):
        self._entered.set()
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_failing_workspace_client_close_still_releases_the_platform_lock(caplog):
    """close-raise: the step is logged, and the lock plus the state are still released."""
    calls = []
    app = MagicMock()
    app.client = _RaisingClient(calls)
    adapter = _adapter(_app=app)

    with patch("gateway.status_scoped_locks.release_scoped_lock") as release:
        await adapter.disconnect()

    release.assert_called_once_with("slack-app-token", "xapp-fake")
    assert adapter._platform_lock_identity is None
    assert adapter._app is None
    assert adapter._app_token is None
    assert adapter._proxy_url is None
    assert adapter._bot_user_id is None
    assert adapter._running is False
    assert "Closing a workspace client failed" in caplog.text


@pytest.mark.asyncio
async def test_one_failing_client_does_not_strand_the_other_workspace_clients():
    """The remaining Slack WebClients still get closed, so none leaks an aiohttp session."""
    calls = []
    app = MagicMock()
    app.client = _RaisingClient(calls)
    adapter = _adapter(_app=app, _team_clients={"T_FAKE": _RecordingClient(calls)})

    with patch("gateway.status_scoped_locks.release_scoped_lock"):
        await adapter.disconnect()

    assert calls == ["raise", "closed"]


@pytest.mark.asyncio
async def test_a_raising_step_does_not_skip_the_later_steps():
    """Step-level isolation: a step that raises is logged and the chain, lock included, goes on."""
    calls = []
    adapter = _adapter(
        _active_streams={("T_FAKE", "C_FAKE", "stream"): {"ts": "1.0", "sent": ""}},
        _team_clients={"T_FAKE": _RecordingClient(calls)},
    )

    async def _seal_explodes(key, stream, delta=None):
        raise RuntimeError("chat.stopStream exploded")

    adapter._seal_stream = _seal_explodes

    with patch("gateway.status_scoped_locks.release_scoped_lock") as release:
        await adapter.disconnect()

    release.assert_called_once_with("slack-app-token", "xapp-fake")
    assert adapter._active_streams == {}
    assert adapter._app is None
    assert calls == ["closed"]


@pytest.mark.asyncio
async def test_cancelled_teardown_finishes_the_rest_and_releases_the_lock(caplog):
    """seal-hang: a cancelled Slack API wait must not skip the later steps or the lock release."""
    entered = asyncio.Event()
    calls = []
    handler = MagicMock()
    handler.close_async = AsyncMock(return_value=None)
    adapter = _adapter(
        _handler=handler,
        _active_streams={("T_FAKE", "C_FAKE", "stream"): {"ts": "1.0", "sent": ""}},
        _team_clients={"T_FAKE": _RecordingClient(calls)},
    )
    adapter._get_client = lambda *args, **kwargs: _HangingClient(entered)

    with patch("gateway.status_scoped_locks.release_scoped_lock") as release:
        task = asyncio.ensure_future(adapter.disconnect())
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    release.assert_called_once_with("slack-app-token", "xapp-fake")
    assert adapter._platform_lock_identity is None
    assert adapter._app is None
    assert adapter._active_streams == {}
    # The steps queued behind the cancelled one still ran: socket mode, then the clients.
    handler.close_async.assert_awaited_once_with()
    assert calls == ["closed"]
    assert "Teardown step 'seal stream C_FAKE' was cancelled" in caplog.text
