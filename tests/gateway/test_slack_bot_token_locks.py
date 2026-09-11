"""Ported behavioral pins for Slack per-bot-token locks (multi-workspace).

connect() acquires one ``slack-bot-token`` scoped lock per workspace token
(sha256-hex identity, never raw token material) plus the app-token platform
lock; a failed workspace acquire releases the already-held locks and fails
closed; disconnect() releases everything.
"""

import asyncio
import contextlib
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig

import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


def _noop_decorator(_matcher):
    def decorator(fn):
        return fn

    return decorator


def _fake_create_task(coro):
    assert asyncio.iscoroutine(coro), f"expected coroutine, got {type(coro).__name__}"
    coro.close()
    loop = asyncio.get_event_loop()
    return loop.create_task(asyncio.Event().wait())


def _make_workspace_client(team_id, user_id, name):
    client = AsyncMock()
    client.auth_test = AsyncMock(
        return_value={"user_id": user_id, "user": name, "team_id": team_id, "team": name}
    )
    return client


class FakeSocketModeHandler:
    def __init__(self, app, app_token, proxy=None):
        self.app = app
        self.app_token = app_token
        self.client = MagicMock(proxy=None)

    async def start_async(self):
        return None

    async def close_async(self):
        return None


def _connect_stack(acquire):
    mock_app = MagicMock()
    mock_app.event = _noop_decorator
    mock_app.command = _noop_decorator
    mock_app.action = _noop_decorator
    mock_app.client = AsyncMock()
    calls = {"n": 0}

    def _new_client(*args, **kwargs):
        # 1st call: primary AsyncApp client; then one workspace client per token.
        calls["n"] += 1
        if calls["n"] == 1:
            return AsyncMock()
        if calls["n"] == 2:
            return _make_workspace_client("T_A", "U_A", "a")
        return _make_workspace_client("T_B", "U_B", "b")
    return [
        patch.object(_slack_mod, "AsyncApp", return_value=mock_app),
        patch.object(
            _slack_mod,
            "AsyncWebClient",
            side_effect=_new_client,
        ),
        patch.object(_slack_mod, "AsyncSocketModeHandler", FakeSocketModeHandler),
        patch.object(_slack_mod, "get_secret", return_value="xapp-test"),
        patch("gateway.status.acquire_scoped_lock", side_effect=acquire),
        patch("asyncio.create_task", side_effect=_fake_create_task),
    ]


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


class TestSlackBotTokenLocks:
    @pytest.mark.asyncio
    async def test_multi_workspace_acquires_one_lock_per_token(self):
        """Two workspace tokens → two slack-bot-token locks keyed by
        sha256 hex digests; raw tokens never stored on the adapter."""
        adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-A,xoxb-B"))
        adapter._socket_watchdog_interval_s = 9999
        acquired = []

        def _acquire(scope, identity, **kwargs):
            acquired.append((scope, identity))
            return (True, None)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("gateway.status.release_scoped_lock"))
            for p in _connect_stack(_acquire):
                stack.enter_context(p)
            assert await adapter.connect() is True

        bot_locks = [ident for scope, ident in acquired if scope == "slack-bot-token"]
        assert bot_locks == [_digest("xoxb-A"), _digest("xoxb-B")]
        assert adapter._bot_token_lock_identities == bot_locks
        assert "xoxb-A" not in repr(adapter._bot_token_lock_identities)
        assert "xoxb-B" not in repr(adapter._bot_token_lock_identities)

    @pytest.mark.asyncio
    async def test_second_token_conflict_releases_first_and_fails_closed(self):
        """A conflicting second workspace token releases the first lock and
        connect() returns False (fail closed, no half-held state)."""
        adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-A,xoxb-B"))
        adapter._socket_watchdog_interval_s = 9999

        def _acquire(scope, identity, **kwargs):
            if scope == "slack-bot-token" and identity == _digest("xoxb-B"):
                return (False, {"pid": 4242})
            return (True, None)

        with contextlib.ExitStack() as stack:
            mock_release = stack.enter_context(patch("gateway.status.release_scoped_lock"))
            for p in _connect_stack(_acquire):
                stack.enter_context(p)
            assert await adapter.connect() is False

        released = [c.args[1] for c in mock_release.call_args_list if c.args[0] == "slack-bot-token"]
        assert _digest("xoxb-A") in released
        assert adapter._bot_token_lock_identities == []

    @pytest.mark.asyncio
    async def test_disconnect_releases_all_bot_token_locks(self):
        adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-A,xoxb-B"))
        adapter._socket_watchdog_interval_s = 9999

        def _acquire(scope, identity, **kwargs):
            return (True, None)

        with contextlib.ExitStack() as stack:
            mock_release = stack.enter_context(patch("gateway.status.release_scoped_lock"))
            for p in _connect_stack(_acquire):
                stack.enter_context(p)
            assert await adapter.connect() is True
            assert adapter._bot_token_lock_identities == [_digest("xoxb-A"), _digest("xoxb-B")]
            await adapter.disconnect()

        released = [c.args[1] for c in mock_release.call_args_list if c.args[0] == "slack-bot-token"]
        assert sorted(released) == sorted([_digest("xoxb-A"), _digest("xoxb-B")])
        assert adapter._bot_token_lock_identities == []
