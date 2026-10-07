"""Mattermost ``gateway_platform_event`` fire-site: ``post_edited`` → ``message_edited`` (#126794).

Covers the Mattermost parity slice of the normalized-envelope pipeline:
* ``post_edited`` WS events normalize to the stable ``message_edited`` payload
  (chat_id, message_id, thread_id, bounded text, edited_at ISO 8601 UTC) and
  dispatch through the gateway-owned post-auth boundary
* the editor (post ``user_id``) is the authorized actor; bot-authored edits
  drop at the fire-site
* malformed events (missing ids / unparseable post JSON) drop, fail closed
* no installed gateway callback means no fire (no trusted auth boundary)
* the has_hook no-subscriber fast-path skips all normalization work
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

from plugins.platforms.mattermost.adapter import MattermostAdapter  # noqa: E402


def _adapter() -> MattermostAdapter:
    """Build a MattermostAdapter with the same mocked config as test_mattermost."""
    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"url": "https://mm.example.com"},
    )
    a = MattermostAdapter(config)
    a._bot_user_id = "bot-1"
    return a


def _edited_event(
    *,
    post_id="post-1",
    channel_id="chan-1",
    user_id="user-9",
    message="edited text",
    root_id=None,
    update_at=None,
    channel_type="O",
    sender_name="@alice",
    post_json=None,
):
    """A Mattermost ``post_edited`` WS event; the post rides as JSON under data.post."""
    post = post_json if post_json is not None else json.dumps({
        "id": post_id, "user_id": user_id, "channel_id": channel_id,
        "message": message, "root_id": root_id, "update_at": update_at,
    })
    return {"event": "post_edited", "data": {
        "post": post, "channel_type": channel_type, "sender_name": sender_name,
    }}


@pytest.fixture(autouse=True)
def _observer_available(monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)


def _capture(a):
    seen: list = []

    async def observe(event, source):
        seen.append((event, source))

    a.set_platform_event_handler(observe)
    return seen


class TestMessageEdited:
    def test_edit_normalized_and_fired(self):
        a = _adapter()
        seen = _capture(a)

        asyncio.run(a._handle_ws_event(_edited_event()))

        assert len(seen) == 1
        event, source = seen[0]
        assert event == {
            "platform": "mattermost",
            "event_type": "message_edited",
            "payload": {
                "chat_id": "chan-1",
                "message_id": "post-1",
                "thread_id": None,
                "text": "edited text",
                "edited_at": None,
            },
        }
        json.dumps(event)
        assert source.user_id == "user-9"
        assert source.user_name == "alice"
        assert source.chat_id == "chan-1"
        assert source.chat_type == "channel"

    def test_edit_in_thread_carries_thread_id(self):
        a = _adapter()
        seen = _capture(a)

        asyncio.run(a._handle_ws_event(_edited_event(root_id="root-1")))

        event, source = seen[0]
        assert event["payload"]["thread_id"] == "root-1"
        assert source.thread_id == "root-1"

    def test_top_level_edit_promotes_to_thread_in_thread_mode(self):
        a = _adapter()
        a._reply_mode = "thread"
        seen = _capture(a)

        asyncio.run(a._handle_ws_event(_edited_event()))

        event, source = seen[0]
        assert event["payload"]["thread_id"] == "post-1"
        assert source.thread_id == "post-1"

    def test_edit_respects_allowed_channels(self):
        a = _adapter()
        a.config.extra["allowed_channels"] = ["chan-allowed"]
        seen = _capture(a)

        asyncio.run(a._handle_ws_event(_edited_event(channel_id="chan-secret")))

        assert seen == []

    def test_dm_carries_dm_chat_type(self):
        a = _adapter()
        seen = _capture(a)

        asyncio.run(a._handle_ws_event(_edited_event(channel_type="D")))

        event, source = seen[0]
        assert source.chat_type == "dm"
        assert event["payload"]["chat_id"] == "chan-1"

    def test_bot_authored_edit_dropped(self):
        """The bot's own post edits must not fire."""
        a = _adapter()
        seen = _capture(a)

        asyncio.run(a._handle_ws_event(_edited_event(user_id="bot-1")))

        assert seen == []

    def test_edited_at_serialized_from_epoch_ms(self):
        a = _adapter()
        seen = _capture(a)
        ts_ms = int(datetime(2026, 8, 12, 10, 30, tzinfo=timezone.utc).timestamp() * 1000)

        asyncio.run(a._handle_ws_event(_edited_event(update_at=ts_ms)))

        assert seen[0][0]["payload"]["edited_at"] == "2026-08-12T10:30:00+00:00"

    def test_no_subscriber_skips_everything(self):
        a = _adapter()
        handler = AsyncMock()
        a.set_platform_event_handler(handler)
        a._message_edited_parts = MagicMock()

        import hermes_cli.lifecycle as lifecycle
        orig = lifecycle.has_hook
        lifecycle.has_hook = lambda _n: False
        try:
            asyncio.run(a._handle_ws_event(_edited_event()))
        finally:
            lifecycle.has_hook = orig

        a._message_edited_parts.assert_not_called()
        handler.assert_not_awaited()

    def test_no_gateway_callback_fails_closed(self):
        a = _adapter()  # set_platform_event_handler never called
        asyncio.run(a._handle_ws_event(_edited_event()))  # no raise

    def test_missing_ids_drop(self):
        a = _adapter()
        seen = _capture(a)

        asyncio.run(a._handle_ws_event(_edited_event(post_id="")))

        assert seen == []

    def test_unparseable_post_drops(self):
        a = _adapter()
        seen = _capture(a)

        asyncio.run(a._handle_ws_event(_edited_event(post_json="{not json")))

        assert seen == []

    def test_dispatch_error_is_swallowed(self):
        a = _adapter()

        async def boom(event, source):
            raise RuntimeError("plugin boom")

        a.set_platform_event_handler(boom)
        asyncio.run(a._handle_ws_event(_edited_event()))  # no raise


class TestWsRouting:
    async def _route(self, a, event):
        await a._handle_ws_event(event)

    def test_post_edited_routes_to_the_fire_site(self):
        a = _adapter()
        a._on_platform_post_edited = AsyncMock()
        a._platform_events_subscribed = lambda: False  # stop before the real pipeline

        asyncio.run(self._route(a, _edited_event()))

        a._on_platform_post_edited.assert_awaited_once()

    def test_new_post_does_not_route_to_the_edit_fire_site(self):
        a = _adapter()
        a._on_platform_post_edited = AsyncMock()

        asyncio.run(self._route(a, {"event": "posted", "data": {}}))

        a._on_platform_post_edited.assert_not_awaited()

    def test_other_events_still_ignored(self):
        a = _adapter()
        a._on_platform_post_edited = AsyncMock()

        asyncio.run(self._route(a, {"event": "status_change", "data": {}}))
        asyncio.run(self._route(a, {"data": {}}))

        a._on_platform_post_edited.assert_not_awaited()
