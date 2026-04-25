"""Regression guard for #15421 bug 1 — Slack channel session scoping.

Before this fix, every top-level Slack channel message got a unique
``thread_id`` (the message's own ``ts``) stamped onto its
``MessageSource``.  The gateway session store keys sessions by
``(platform, channel_id, thread_id)``, so each top-level message
spawned a **brand new session** and channel context never accumulated
across messages — even when the operator set ``reply_in_thread: false``
in ``config.yaml`` expecting channel-wide conversation.

The fix: when ``reply_in_thread: false`` is configured, top-level
channel messages now land on ``thread_id = None`` so the session store
groups them under a single channel-scoped session.  Genuine thread
replies (``event.thread_ts != ts``) still scope sessions per thread in
both modes — threading UX is unchanged when the operator actually
asks for it.

These tests drive the real ``SlackAdapter._handle_slack_message`` code
path with mocked aiohttp / user-resolution so the ``MessageEvent``
that reaches ``handle_message`` exposes exactly what the session store
will key on.  Asserting on the event keeps the seam tight against the
production function's behaviour rather than a re-implementation.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.slack import SlackAdapter


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point document cache to tmp_path so tests don't touch ~/.hermes."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )


def _channel_event(text: str, ts: str, thread_ts: str = None) -> dict:
    """Build a minimal ``message`` event for the Slack Events API
    resembling what ``handle_message_event`` would pass through."""
    event = {
        "channel": "C_CHAN",
        "channel_type": "channel",
        "user": "U_USER",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


class TestChannelSessionScopeDefault:
    """``reply_in_thread: true`` is the historical default.  Top-level
    channel messages still map ``thread_id = ts`` so each new message
    becomes its own threaded session — unchanged from the pre-#15421
    behaviour."""

    @pytest.mark.asyncio
    async def test_top_level_maps_to_ts_when_reply_in_thread_true(self, adapter):
        adapter.config.extra["reply_in_thread"] = True
        event = _channel_event(
            "<@U_BOT> hello",
            ts="1700000000.000001",
        )

        captured = []
        adapter.handle_message = AsyncMock(
            side_effect=lambda e: captured.append(e)
        )
        with patch.object(
            adapter, "_resolve_user_name",
            new=AsyncMock(return_value="testuser"),
        ):
            await adapter._handle_slack_message(event)

        assert len(captured) == 1, (
            "handler dropped the top-level channel mention — "
            "mention gating misfired"
        )
        source = captured[0].source
        assert source.thread_id == "1700000000.000001", (
            "legacy default (reply_in_thread=true) must keep stamping "
            "thread_id = ts so each top-level message gets its own "
            "threaded session — regression guard"
        )

    @pytest.mark.asyncio
    async def test_top_level_default_behaves_like_true(self, adapter):
        """Operators who never set ``reply_in_thread`` must see the
        historical behaviour (true).  Pin the default explicitly."""
        # Note: no adapter.config.extra["reply_in_thread"] set here.
        event = _channel_event(
            "<@U_BOT> hello",
            ts="1700000000.000002",
        )

        captured = []
        adapter.handle_message = AsyncMock(
            side_effect=lambda e: captured.append(e)
        )
        with patch.object(
            adapter, "_resolve_user_name",
            new=AsyncMock(return_value="testuser"),
        ):
            await adapter._handle_slack_message(event)

        assert len(captured) == 1
        assert captured[0].source.thread_id == "1700000000.000002"


class TestChannelSessionScopeShared:
    """``reply_in_thread: false`` is the #15421 fix: top-level channel
    messages get ``thread_id = None`` so all of them share one
    channel-scoped session.  Genuine thread replies still get their
    real ``thread_ts``."""

    @pytest.mark.asyncio
    async def test_top_level_maps_to_none_when_reply_in_thread_false(self, adapter):
        adapter.config.extra["reply_in_thread"] = False
        event = _channel_event(
            "<@U_BOT> hello",
            ts="1700000000.000003",
        )

        captured = []
        adapter.handle_message = AsyncMock(
            side_effect=lambda e: captured.append(e)
        )
        with patch.object(
            adapter, "_resolve_user_name",
            new=AsyncMock(return_value="testuser"),
        ):
            await adapter._handle_slack_message(event)

        assert len(captured) == 1
        source = captured[0].source
        assert source.thread_id is None, (
            "reply_in_thread=false must set thread_id=None for top-level "
            "channel messages so the session store groups them under a "
            "single channel-scoped session (#15421 bug 1)"
        )

    @pytest.mark.asyncio
    async def test_top_level_reply_to_id_stays_none_when_shared(self, adapter):
        """The outbound-side ``reply_to_message_id`` check already
        uses ``thread_ts != ts`` to decide whether to thread the
        response.  When ``thread_ts`` is None, the check evaluates
        ``None != ts`` → True → reply_to_message_id IS set.  That would
        thread the reply, which is the opposite of what
        reply_in_thread=false means for top-level messages.

        The fix ensures reply_to_message_id is None for top-level
        messages in shared-session mode so the bot posts a fresh
        channel message (not a threaded reply).
        """
        adapter.config.extra["reply_in_thread"] = False
        event = _channel_event(
            "<@U_BOT> hello",
            ts="1700000000.000004",
        )

        captured = []
        adapter.handle_message = AsyncMock(
            side_effect=lambda e: captured.append(e)
        )
        with patch.object(
            adapter, "_resolve_user_name",
            new=AsyncMock(return_value="testuser"),
        ):
            await adapter._handle_slack_message(event)

        assert captured[0].reply_to_message_id is None, (
            "top-level channel messages with reply_in_thread=false "
            "must not be threaded (reply_to_message_id=None)"
        )

    @pytest.mark.asyncio
    async def test_thread_reply_scopes_by_thread_even_when_shared(self, adapter):
        """Bug 1's fix targets ONLY top-level channel messages.  Genuine
        thread replies (``thread_ts != ts``) must still scope per-thread
        sessions so multi-person threaded conversations don't collide
        with unrelated channel chatter."""
        adapter.config.extra["reply_in_thread"] = False
        # Reply to an earlier thread root at ts=1700000000.000000
        event = _channel_event(
            "<@U_BOT> following up",
            ts="1700000000.000005",
            thread_ts="1700000000.000000",
        )

        captured = []
        adapter.handle_message = AsyncMock(
            side_effect=lambda e: captured.append(e)
        )
        with patch.object(
            adapter, "_resolve_user_name",
            new=AsyncMock(return_value="testuser"),
        ):
            await adapter._handle_slack_message(event)

        assert len(captured) == 1
        source = captured[0].source
        assert source.thread_id == "1700000000.000000", (
            "genuine thread replies must still scope by thread even "
            "when reply_in_thread=false — only TOP-LEVEL messages share "
            "the channel-wide session"
        )
        assert captured[0].reply_to_message_id == "1700000000.000000", (
            "reply should thread under the existing thread root"
        )


class TestThreadReplyAlwaysScopesByThread:
    """Cross-cutting invariant: genuine thread replies always scope by
    ``thread_ts`` regardless of ``reply_in_thread``.  If this ever
    regresses, every thread-scoped conversation leaks across threads."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reply_in_thread", [True, False])
    async def test_thread_reply_keyed_by_thread_ts(self, adapter, reply_in_thread):
        adapter.config.extra["reply_in_thread"] = reply_in_thread
        event = _channel_event(
            "<@U_BOT> thread reply",
            ts="1700000000.000010",
            thread_ts="1700000000.000009",
        )

        captured = []
        adapter.handle_message = AsyncMock(
            side_effect=lambda e: captured.append(e)
        )
        with patch.object(
            adapter, "_resolve_user_name",
            new=AsyncMock(return_value="testuser"),
        ):
            await adapter._handle_slack_message(event)

        assert len(captured) == 1, (
            f"thread reply dropped with reply_in_thread={reply_in_thread}"
        )
        assert captured[0].source.thread_id == "1700000000.000009"
