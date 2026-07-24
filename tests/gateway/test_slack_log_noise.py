"""C13 — Slack log noise & log privacy invariants.

Covers three bug classes consolidated in the slack/c13-lognoise cluster:

1. Catch-all event ack (#38847 / #64218, issue #6572): unhandled subscribed
   event types must be acked by a fallback listener registered AFTER every
   named handler — silencing slack_bolt "Unhandled request" WARNINGs and,
   more importantly, keeping Slack's Events API failure rate near 0% so
   Slack does not auto-disable the app's Event Subscriptions at scale.
   The catch-all must keep a DEBUG line (visibility into unknown event
   types) and must not log message content.

2. Log privacy (#58478, issue #58477, widened here): no message text /
   block content above DEBUG level anywhere in the adapter's inbound path,
   and even DEBUG lines must not embed raw message text (metadata only, or
   truncated lengths).

3. Clarify resolution logging: the chosen option text stays out of
   INFO-level logs (choice index only); full text is DEBUG-only and
   truncated.
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402

ADAPTER_LOGGER = "plugins.platforms.slack.adapter"


def _fake_create_task(coro, **kwargs):
    coro.close()
    task = MagicMock()
    task.done.return_value = True
    return task


def _connect_and_capture_handlers():
    """Run connect() with a mocked bolt app; return [(matcher, fn), ...]."""
    config = PlatformConfig(enabled=True, token="xoxb-fake")
    adapter = SlackAdapter(config)

    registered = []

    mock_app = MagicMock()

    def mock_event(event_type):
        def decorator(fn):
            registered.append((event_type, fn))
            return fn

        return decorator

    mock_app.event = mock_event
    mock_app.client = AsyncMock()

    mock_web_client = AsyncMock()
    mock_web_client.auth_test = AsyncMock(
        return_value={
            "user_id": "U_BOT",
            "user": "testbot",
            "team_id": "T_FAKE",
            "team": "FakeTeam",
        }
    )

    socket_mode_handler = MagicMock()
    socket_mode_handler.start_async = AsyncMock(return_value=None)

    with (
        patch.object(_slack_mod, "AsyncApp", return_value=mock_app),
        patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client),
        patch.object(
            _slack_mod, "AsyncSocketModeHandler", return_value=socket_mode_handler
        ),
        patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}),
        patch("gateway.status.acquire_scoped_lock", return_value=(True, None)),
        patch("asyncio.create_task", side_effect=_fake_create_task),
    ):
        asyncio.run(adapter.connect())

    return adapter, registered


class TestCatchAllEventAck:
    """#6572 / Event Subscriptions auto-disable — catch-all fallback ack."""

    def test_catchall_registered_last_and_named_handlers_first(self):
        _, registered = _connect_and_capture_handlers()
        matchers = [m for m, _fn in registered]

        # Named handlers exist and come first — bolt dispatches to the first
        # matching listener, so registration order IS the shadowing guarantee.
        assert "message" in matchers
        assert "app_mention" in matchers
        pattern_positions = [
            i for i, m in enumerate(matchers) if isinstance(m, re.Pattern)
        ]
        assert pattern_positions, "catch-all re.Pattern matcher must be registered"
        last_named = max(
            i for i, m in enumerate(matchers) if not isinstance(m, re.Pattern)
        )
        assert pattern_positions[-1] > last_named, (
            "catch-all must be registered AFTER every named event handler "
            f"(order: {matchers!r})"
        )

    @pytest.mark.asyncio
    async def test_catchall_acks_unsubscribed_event_quietly(self, caplog):
        """The catch-all fires for an unhandled type, logs only at DEBUG,
        and never logs message content."""
        _, registered = await asyncio.to_thread(_connect_and_capture_handlers)
        catchall_fn = next(
            fn for m, fn in reversed(registered) if isinstance(m, re.Pattern)
        )

        secret = "confidential launch codes in a reaction item"
        event = {
            "type": "reaction_added",
            "user": "U123",
            "item": {"type": "message", "text": secret},
        }
        body = {"event": event}
        bolt_logger = logging.getLogger("slack_bolt_listener_test")

        with caplog.at_level(logging.DEBUG, logger="slack_bolt_listener_test"):
            # Returning without raising IS the ack — bolt sends the Socket
            # Mode ack (HTTP 200) whenever a listener handles the event.
            await catchall_fn(event=event, body=body, logger=bolt_logger)

        listener_records = [
            r for r in caplog.records if r.name == "slack_bolt_listener_test"
        ]
        assert any(
            r.levelno == logging.DEBUG and "reaction_added" in r.getMessage()
            for r in listener_records
        ), "catch-all must keep a DEBUG line naming the unhandled event type"
        assert all(r.levelno <= logging.DEBUG for r in listener_records), (
            "catch-all must not log above DEBUG (that's the noise it exists "
            "to remove)"
        )
        assert secret not in caplog.text

    @pytest.mark.asyncio
    async def test_named_handlers_still_dispatch(self):
        """The catch-all must not swallow events the adapter DOES handle:
        the 'message' listener still routes into _handle_slack_message."""
        adapter, registered = await asyncio.to_thread(_connect_and_capture_handlers)
        message_fn = next(fn for m, fn in registered if m == "message")

        adapter._handle_slack_message = AsyncMock()
        event = {"type": "message", "text": "hi", "ts": "1.2", "channel": "C1"}
        await message_fn(event=event, say=AsyncMock(), body={"event": event})
        adapter._handle_slack_message.assert_awaited_once()


class TestInboundLogPrivacy:
    """#58477 (widened): no message text above DEBUG; DEBUG is metadata-only
    or truncated."""

    def _make_adapter(self):
        config = PlatformConfig(enabled=True, token="***")
        a = SlackAdapter(config)
        a._app = MagicMock()
        a._app.client = AsyncMock()
        a._bot_user_id = "U_BOT"
        a._running = True
        a.handle_message = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_message_pipeline_never_logs_text_above_debug(self, caplog):
        """Drive a real inbound channel message end-to-end and assert the
        message text appears in NO log record at any level, and block
        content appears in none above DEBUG."""
        secret_text = "SECRET-INBOUND-TEXT do not log me 12345"
        secret_block = "SECRET-BLOCK-QUOTE private incident data"
        adapter = self._make_adapter()
        adapter._dedup = MagicMock(is_duplicate=MagicMock(return_value=False))
        adapter._channel_team = {}

        event = {
            "type": "message",
            "channel": "C_PRIV",
            "channel_type": "channel",
            "ts": "1710000000.000200",
            "team": "T1",
            "user": "U_USER",
            "client_msg_id": "cmid-1",
            "text": f"<@U_BOT> {secret_text}",
            "blocks": [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_quote",
                            "elements": [
                                {
                                    "type": "rich_text_section",
                                    "elements": [
                                        {"type": "text", "text": secret_block}
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
            with patch.object(
                adapter, "_resolve_user_name", new=AsyncMock(return_value="u")
            ):
                await adapter._handle_slack_message(event)

        assert secret_text not in caplog.text
        assert secret_block not in caplog.text

    @pytest.mark.asyncio
    async def test_entry_diagnostic_log_is_metadata_only(self, caplog):
        """#30185's event-arrival DEBUG line surfaces routing metadata
        (type/subtype/user/bot_id/channel/ts) but never the message text."""
        secret = "bot payload that must stay out of logs"
        adapter = self._make_adapter()
        adapter._dedup = MagicMock(is_duplicate=MagicMock(return_value=True))

        event = {
            "type": "message",
            "subtype": "bot_message",
            "user": "U_OTHER_BOT",
            "bot_id": "B_OTHER",
            "bot_profile": {"name": "Peer Bot"},
            "ts": "12345.6789",
            "channel": "C_SHARED",
            "text": secret,
        }
        with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
            await adapter._handle_slack_message(event)

        entry_lines = [
            r.getMessage() for r in caplog.records if "event received" in r.getMessage()
        ]
        assert entry_lines, "entry diagnostic DEBUG line must fire"
        assert "bot_id=B_OTHER" in entry_lines[0]
        assert secret not in caplog.text


class TestClarifyLogPrivacy:
    """Clarify choice text is user content: index at INFO, text DEBUG-only."""

    @pytest.mark.asyncio
    async def test_clarify_resolution_logs_index_not_text_at_info(self, caplog):
        from tools import clarify_gateway as cm

        secret_choice = "migrate the production database tonight"
        with cm._lock:
            cm._entries.clear()
        cm.register("cid-priv", "sk-priv", "Pick", ["noop", secret_choice])

        config = PlatformConfig(enabled=True, token="xoxb-test-token")
        adapter = SlackAdapter(config)
        adapter._app = MagicMock()
        adapter._bot_user_id = "U_BOT"
        adapter._team_clients = {"T1": AsyncMock()}
        adapter._team_bot_user_ids = {"T1": "U_BOT"}
        adapter._channel_team = {"C1": "T1"}
        adapter._clarify_resolved["9.9"] = False
        adapter._team_clients["T1"].chat_update = AsyncMock()

        async def _handler(_event):
            return None

        adapter.set_message_handler(_handler)

        body = {
            "message": {
                "ts": "9.9",
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "❓ Pick"}}
                ],
            },
            "channel": {"id": "C1"},
            "user": {"name": "norbert", "id": "U_NORBERT"},
        }
        action = {"action_id": "hermes_clarify_choice_1", "value": "cid-priv|1"}

        with caplog.at_level(logging.DEBUG, logger=ADAPTER_LOGGER):
            with patch.object(
                adapter,
                "_is_interactive_user_authorized",
                return_value=True,
            ):
                await adapter._handle_clarify_action(AsyncMock(), body, action)

        info_and_above = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.INFO
        ]
        assert any("resolved clarify" in m for m in info_and_above)
        assert all(secret_choice not in m for m in info_and_above), (
            "clarify choice text must not appear at INFO or above"
        )
        # Full choice text remains available for debugging at DEBUG level.
        debug_msgs = [
            r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
        ]
        assert any(secret_choice in m for m in debug_msgs)
