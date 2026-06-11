"""Regression tests for a batch of independently-verified Discord adapter bugs.

Each test pins one fix:

1. Forwarded-message text is dropped when the forwarder adds their own comment.
2. send() does not flag a surfaced 429 as retryable, corrupting the reply into
   a plain-text fallback.
3. The voice inactivity handler cancels itself mid-flight, dropping the
   "left channel" notice.
4. Media senders ignore metadata['thread_id'], splitting handoff replies across
   the parent channel and the thread.
5. Permanent auth/intent failures never escalate to a non-retryable fatal, so
   the gateway reconnects a bad credential forever.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType


def _ensure_discord_mock():
    """Install a minimal mock ``discord`` module only when the real package is
    absent — mirrors the bootstrap used by the other Discord test modules."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    if sys.modules.get("discord") is not None:
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(is_loaded=lambda: True)

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


# ── Fix 1: forwarded-message text is dropped when a comment is present ──────


class _DMChannelStub:
    def __init__(self, channel_id):
        self.id = channel_id


class _NoMatchType:
    """Sentinel so isinstance(channel, discord.Thread) is always False."""


@pytest.mark.asyncio
async def test_forwarded_text_combined_with_user_comment(monkeypatch):
    """A forward carries the forwarded body in message_snapshots; the user's own
    typed comment lives in message.content. When both are present the adapter
    must surface BOTH — the old `snapshot_text and not raw_content` guard dropped
    the forwarded body entirely whenever the forwarder also typed a comment.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999, name="Hermes"))
    adapter._text_batch_delay_seconds = 0  # dispatch synchronously, no batching

    # Route the DM through the real channel-type checks with cheap stubs.
    monkeypatch.setattr(discord_platform.discord, "DMChannel", _DMChannelStub)
    monkeypatch.setattr(discord_platform.discord, "Thread", _NoMatchType)

    snapshot = SimpleNamespace(content="The forwarded article body.", attachments=[])
    message = SimpleNamespace(
        id=1,
        content="check this out",
        message_snapshots=[snapshot],
        attachments=[],
        mentions=[],
        author=SimpleNamespace(id=42, name="alice", display_name="Alice", bot=False),
        channel=_DMChannelStub(555),
        reference=None,
        guild=None,
        created_at=None,
    )

    captured = {}

    async def fake_handle_message(event):
        captured["event"] = event

    monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

    await adapter._handle_message(message)

    text = captured["event"].text
    assert "check this out" in text, "forwarder's own comment must survive"
    assert "The forwarded article body." in text, "forwarded body must not be dropped"


@pytest.mark.asyncio
async def test_forwarded_text_alone_still_delivered(monkeypatch):
    """A bare forward (no comment) must still deliver the forwarded body."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999, name="Hermes"))
    adapter._text_batch_delay_seconds = 0  # dispatch synchronously, no batching

    monkeypatch.setattr(discord_platform.discord, "DMChannel", _DMChannelStub)
    monkeypatch.setattr(discord_platform.discord, "Thread", _NoMatchType)

    snapshot = SimpleNamespace(content="Only the forwarded body.", attachments=[])
    message = SimpleNamespace(
        id=2,
        content="",
        message_snapshots=[snapshot],
        attachments=[],
        mentions=[],
        author=SimpleNamespace(id=42, name="alice", display_name="Alice", bot=False),
        channel=_DMChannelStub(555),
        reference=None,
        guild=None,
        created_at=None,
    )

    captured = {}

    async def fake_handle_message(event):
        captured["event"] = event

    monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

    await adapter._handle_message(message)

    assert captured["event"].text == "Only the forwarded body."


# ── Fix 2: send() must flag a surfaced 429 as retryable ─────────────────────


class _RateLimited(Exception):
    """Name contains 'ratelimit' + carries retry_after, so the adapter's
    duck-typed _is_discord_rate_limit() classifies it as a 429."""

    def __init__(self, message, retry_after):
        super().__init__(message)
        self.retry_after = retry_after


@pytest.mark.asyncio
async def test_send_flags_rate_limit_as_retryable():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    async def fake_send(*, content, reference=None):
        raise _RateLimited("429 Too Many Requests", retry_after=5.0)

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(),
    )
    adapter._is_forum_parent = lambda channel: False

    result = await adapter.send("123", "hello")

    assert result.success is False
    # Without retryable=True the gateway misclassifies a 429 as a formatting
    # failure and ships the "Response formatting failed, plain text:" fallback.
    assert result.retryable is True


@pytest.mark.asyncio
async def test_send_non_rate_limit_error_stays_non_retryable():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    async def fake_send(*, content, reference=None):
        raise RuntimeError("some other failure")

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(),
    )
    adapter._is_forum_parent = lambda channel: False

    result = await adapter.send("123", "hello")

    assert result.success is False
    assert result.retryable is False


# ── Fix 3: voice inactivity handler must not cancel itself ──────────────────


@pytest.mark.asyncio
async def test_voice_timeout_handler_delivers_disconnect_notice(monkeypatch):
    """When the inactivity timer fires, _voice_timeout_handler awaits
    leave_voice_channel — which used to cancel the very task it runs in, raising
    CancelledError at the next await (the notice send) and dropping it. The
    self-cancel guard must keep the notice flowing.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.VOICE_TIMEOUT = 0  # fire immediately

    guild_id = 7
    text_channel_id = 888

    sent = []
    text_channel = SimpleNamespace(send=AsyncMock(side_effect=lambda msg: sent.append(msg)))
    adapter._client = SimpleNamespace(get_channel=lambda _id: text_channel)
    adapter._voice_text_channels[guild_id] = text_channel_id
    adapter._on_voice_disconnect = None

    # Register the handler as the running task, exactly as _reset_voice_timeout
    # would, so leave_voice_channel sees it as asyncio.current_task().
    task = asyncio.ensure_future(adapter._voice_timeout_handler(guild_id))
    adapter._voice_timeout_tasks[guild_id] = task
    await task

    assert sent == ["Left voice channel (inactivity timeout)."]


# ── Fix 4: media senders must honor metadata['thread_id'] ───────────────────


@pytest.mark.asyncio
async def test_resolve_send_channel_prefers_thread_id():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    requested = []

    def fake_get_channel(cid):
        requested.append(cid)
        return SimpleNamespace(id=cid)

    adapter._client = SimpleNamespace(get_channel=fake_get_channel, fetch_channel=AsyncMock())

    channel = await adapter._resolve_send_channel("100", metadata={"thread_id": "200"})

    assert channel.id == 200
    assert requested == [200], "thread_id must take precedence over chat_id"


@pytest.mark.asyncio
async def test_resolve_send_channel_falls_back_to_chat_id():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    requested = []

    def fake_get_channel(cid):
        requested.append(cid)
        return SimpleNamespace(id=cid)

    adapter._client = SimpleNamespace(get_channel=fake_get_channel, fetch_channel=AsyncMock())

    channel = await adapter._resolve_send_channel("100", metadata=None)

    assert channel.id == 100
    assert requested == [100]


@pytest.mark.asyncio
async def test_send_file_attachment_routes_to_thread(tmp_path):
    """The session-handoff path passes chat_id=parent and metadata.thread_id=
    handoff-thread. The file must land in the thread, not the parent channel.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    file_path = tmp_path / "report.txt"
    file_path.write_text("hello")

    posted_to = {}

    def fake_get_channel(cid):
        ch = SimpleNamespace(id=cid)

        async def _send(**kw):
            posted_to["channel_id"] = cid
            return SimpleNamespace(id=1)

        ch.send = _send
        return ch

    adapter._client = SimpleNamespace(get_channel=fake_get_channel, fetch_channel=AsyncMock())
    adapter._is_forum_parent = lambda channel: False

    result = await adapter._send_file_attachment(
        "100", str(file_path), caption="cap", metadata={"thread_id": "200"}
    )

    assert result.success is True
    assert posted_to["channel_id"] == 200, "file must be sent into the handoff thread"


# ── Fix 5: permanent auth/intent failure escalates to non-retryable fatal ───


class _LoginFailureBot:
    """Stand-in commands.Bot whose start() raises a permanent auth error,
    mirroring discord.py's LoginFailure (matched by class name)."""

    class LoginFailure(Exception):
        pass

    def __init__(self, **_):
        self.user = SimpleNamespace(id=999, name="Hermes")
        self._events = {}
        self.application_id = 999
        self.tree = SimpleNamespace(
            sync=AsyncMock(return_value=[]),
            fetch_commands=AsyncMock(return_value=[]),
            get_commands=lambda *a, **k: [],
            command=lambda *a, **k: (lambda fn: fn),
        )
        self.http = SimpleNamespace(
            upsert_global_command=AsyncMock(),
            edit_global_command=AsyncMock(),
            delete_global_command=AsyncMock(),
        )

    def event(self, fn):
        self._events[fn.__name__] = fn
        return fn

    async def start(self, token):
        raise _LoginFailureBot.LoginFailure("Improper token has been passed.")

    async def close(self):
        return None

    def is_closed(self):
        return False


@pytest.mark.asyncio
async def test_connect_escalates_permanent_auth_failure_as_non_retryable(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(message_content=False, dm_messages=False, guild_messages=False, members=False, voice_states=False)
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)
    monkeypatch.setattr(discord_platform.commands, "Bot", lambda **kwargs: _LoginFailureBot(**kwargs))
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())

    async def fake_wait_for(awaitable, timeout):
        # Let the bot task surface its exception, then reproduce the READY-wait
        # timeout connect() observes (on_ready never fired).
        await asyncio.sleep(0)
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(discord_platform.asyncio, "wait_for", fake_wait_for)

    ok = await adapter.connect()

    assert ok is False
    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_retryable is False
    assert adapter._fatal_error_code == "discord_LoginFailure"

    # disconnect() must harvest the bot task and clear it without re-raising.
    await adapter.disconnect()
    assert adapter._bot_task is None


@pytest.mark.asyncio
async def test_connect_plain_timeout_stays_retryable(monkeypatch):
    """A connect timeout with no permanent auth/intent error on the bot task
    must NOT be escalated to a fatal state — it stays retryable."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(message_content=False, dm_messages=False, guild_messages=False, members=False, voice_states=False)
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    class _SlowBot(_LoginFailureBot):
        async def start(self, token):
            await asyncio.sleep(60)  # connected but on_ready not yet fired

    monkeypatch.setattr(discord_platform.commands, "Bot", lambda **kwargs: _SlowBot(**kwargs))
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())

    async def fake_wait_for(awaitable, timeout):
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(discord_platform.asyncio, "wait_for", fake_wait_for)

    ok = await adapter.connect()

    assert ok is False
    assert adapter.has_fatal_error is False

    await adapter.disconnect()
