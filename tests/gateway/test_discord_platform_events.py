"""Discord ``gateway_platform_event`` fire-sites (#64176 remaining scope).

Covers the Discord half of the normalized-envelope pipeline:
* ``message_edited`` / ``message_deleted`` / ``thread_created`` /
  ``thread_renamed`` normalize to stable plain-dict envelopes (no raw SDK
  objects) and dispatch through the gateway-owned post-auth boundary
* bot-authored events are dropped at the fire-site (streaming edits are noise)
* malformed events (missing ids / identities) drop, fail closed
* no installed gateway callback means no fire (no trusted auth boundary)
* the has_hook no-subscriber fast-path skips all normalization work
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# discord.py is an optional dep; mock it so the adapter imports
# (same shim as test_discord_attachment_download).
# ---------------------------------------------------------------------------
def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(
        View=object, button=lambda *a, **k: (lambda fn: fn), Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5,
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

# Import Thread from the mocked module, not discord directly, so isinstance
# checks in the adapter match the class our fixtures instantiate.
_DiscordThread = sys.modules["discord"].Thread

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


def _adapter() -> DiscordAdapter:
    """Build a DiscordAdapter without the heavy __init__ (fire-sites only
    need platform/config/gateway_runner and the handler slot)."""
    a = object.__new__(DiscordAdapter)
    a.platform = Platform.DISCORD
    a.config = SimpleNamespace(extra={})
    a.gateway_runner = None
    return a


def _channel(chan_id=555, thread=False):
    if thread:
        chan = _DiscordThread()
        chan.id = chan_id
        return chan
    return SimpleNamespace(id=chan_id)


def _message(
    *,
    message_id=456,
    chan=None,
    author_id=777,
    bot=False,
    content="hello world",
    edited_at=None,
):
    return SimpleNamespace(
        id=message_id,
        channel=chan if chan is not None else _channel(),
        author=SimpleNamespace(id=author_id, bot=bot, display_name="user"),
        content=content,
        edited_at=edited_at,
        guild=SimpleNamespace(id=999),
    )


def _thread_obj(*, thread_id=321, name="my thread", owner_id=777, parent_id=555):
    t = _DiscordThread()
    t.id = thread_id
    t.name = name
    t.owner_id = owner_id
    t.parent_id = parent_id
    t.guild = SimpleNamespace(id=999)
    return t


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
        after = _message(content="edited!")

        asyncio.run(a._on_platform_message_edit(_message(), after))

        assert len(seen) == 1
        event, source = seen[0]
        assert event == {
            "platform": "discord",
            "event_type": "message_edited",
            "payload": {
                "chat_id": "555",
                "message_id": "456",
                "thread_id": None,
                "text": "edited!",
                "edited_at": None,
            },
        }
        json.dumps(event)
        assert source.user_id == "777"
        assert source.chat_id == "555"

    def test_edit_in_thread_carries_thread_id(self):
        a = _adapter()
        seen = _capture(a)
        after = _message(chan=_channel(chan_id=888, thread=True))

        asyncio.run(a._on_platform_message_edit(None, after))

        event, source = seen[0]
        assert event["payload"]["thread_id"] == "888"
        assert event["payload"]["chat_id"] == "888"
        assert source.thread_id == "888"

    def test_bot_authored_edit_dropped(self):
        """The bot's own progressive streaming edits must not fire."""
        a = _adapter()
        seen = _capture(a)

        asyncio.run(a._on_platform_message_edit(None, _message(bot=True)))

        assert seen == []

    def test_edited_at_serialized(self):
        import datetime as _dt

        a = _adapter()
        seen = _capture(a)
        after = _message(
            edited_at=_dt.datetime(2026, 8, 12, 10, 30, tzinfo=_dt.timezone.utc),
        )

        asyncio.run(a._on_platform_message_edit(None, after))

        assert seen[0][0]["payload"]["edited_at"] == "2026-08-12T10:30:00+00:00"

    def test_no_subscriber_skips_everything(self):
        a = _adapter()
        handler = AsyncMock()
        a.set_platform_event_handler(handler)
        a._thread_id_and_chat_for_channel = MagicMock()

        import hermes_cli.lifecycle as lifecycle
        orig = lifecycle.has_hook
        lifecycle.has_hook = lambda _n: False
        try:
            asyncio.run(a._on_platform_message_edit(None, _message()))
        finally:
            lifecycle.has_hook = orig

        a._thread_id_and_chat_for_channel.assert_not_called()
        handler.assert_not_awaited()

    def test_no_gateway_callback_fails_closed(self):
        a = _adapter()  # set_platform_event_handler never called
        asyncio.run(a._on_platform_message_edit(None, _message()))  # no raise

    def test_missing_ids_drop(self):
        a = _adapter()
        seen = _capture(a)
        after = _message()
        after.channel = None

        asyncio.run(a._on_platform_message_edit(None, after))

        assert seen == []

    def test_dispatch_error_is_swallowed(self):
        a = _adapter()

        async def boom(event, source):
            raise RuntimeError("plugin boom")

        a.set_platform_event_handler(boom)
        asyncio.run(a._on_platform_message_edit(None, _message()))  # no raise


class TestMessageDeleted:
    def test_delete_normalized_and_fired(self):
        a = _adapter()
        seen = _capture(a)

        asyncio.run(a._on_platform_message_delete(_message()))

        event, source = seen[0]
        assert event == {
            "platform": "discord",
            "event_type": "message_deleted",
            "payload": {
                "chat_id": "555",
                "message_id": "456",
                "thread_id": None,
                "author_id": "777",
            },
        }
        assert source.user_id == "777"

    def test_bot_authored_delete_dropped(self):
        a = _adapter()
        seen = _capture(a)

        asyncio.run(a._on_platform_message_delete(_message(bot=True)))

        assert seen == []

    def test_missing_author_fails_closed(self):
        """No author identity means nothing to authorize against — drop."""
        a = _adapter()
        seen = _capture(a)
        msg = _message()
        msg.author = None

        asyncio.run(a._on_platform_message_delete(msg))

        assert seen == []


class TestThreadCreated:
    def test_thread_create_normalized_and_fired(self):
        a = _adapter()
        seen = _capture(a)

        asyncio.run(a._on_platform_thread_create(_thread_obj()))

        event, source = seen[0]
        assert event == {
            "platform": "discord",
            "event_type": "thread_created",
            "payload": {
                "thread_id": "321",
                "parent_chat_id": "555",
                "name": "my thread",
                "owner_id": "777",
            },
        }
        assert source.thread_id == "321"
        assert source.user_id == "777"

    def test_missing_owner_fails_closed(self):
        a = _adapter()
        seen = _capture(a)
        t = _thread_obj(owner_id=None)

        asyncio.run(a._on_platform_thread_create(t))

        assert seen == []


class TestThreadRenamed:
    def test_rename_normalized_and_fired(self):
        a = _adapter()
        seen = _capture(a)
        before = _thread_obj(name="old name")
        after = _thread_obj(name="new name")

        asyncio.run(a._on_platform_thread_update(before, after))

        event, _source = seen[0]
        assert event == {
            "platform": "discord",
            "event_type": "thread_renamed",
            "payload": {
                "thread_id": "321",
                "parent_chat_id": "555",
                "old_name": "old name",
                "new_name": "new name",
            },
        }

    def test_non_rename_update_dropped(self):
        """Archive/slowmode/tag updates share on_thread_update — only real
        renames fire."""
        a = _adapter()
        seen = _capture(a)
        before = _thread_obj(name="same")
        after = _thread_obj(name="same")

        asyncio.run(a._on_platform_thread_update(before, after))

        assert seen == []


class TestRunnerBoundaryIntegration:
    def test_unauthorized_discord_event_never_reaches_hooks(self):
        """Full path: adapter fire-site -> runner post-auth gate denies."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = lambda source: False
        invoked = MagicMock()
        a = _adapter()
        a.set_platform_event_handler(runner._handle_gateway_platform_event)

        import hermes_cli.lifecycle as lifecycle
        orig_invoke = lifecycle.invoke_hook
        lifecycle.invoke_hook = invoked
        try:
            asyncio.run(a._on_platform_message_edit(None, _message()))
        finally:
            lifecycle.invoke_hook = orig_invoke

        invoked.assert_not_called()

    def test_authorized_discord_event_reaches_hooks(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = lambda source: source.user_id == "777"
        invoked = MagicMock()
        a = _adapter()
        a.set_platform_event_handler(runner._handle_gateway_platform_event)

        import hermes_cli.lifecycle as lifecycle
        orig_invoke = lifecycle.invoke_hook
        lifecycle.invoke_hook = invoked
        try:
            asyncio.run(a._on_platform_message_edit(None, _message()))
        finally:
            lifecycle.invoke_hook = orig_invoke

        invoked.assert_called_once()
        args, kwargs = invoked.call_args
        assert args == ("gateway_platform_event",)
        assert kwargs["platform"] == "discord"
        assert kwargs["event_type"] == "message_edited"
