"""Tests for the Idea Box channel-level intake hook in the Discord adapter.

Covers the wiring added for task t_1b051b8b:

  - DISCORD_IDEABOX_CHANNELS env var is parsed correctly
  - _is_ideabox_intake_message matches direct channel IDs and parent IDs
    (so messages inside a forum post whose parent is in the intake set
    are caught too)
  - _dispatch_ideabox_intake returns False for non-intake messages
  - _dispatch_ideabox_intake sends the embed with the approval view
    for valid submissions
  - _dispatch_ideabox_intake sends a friendly error embed for invalid
    submissions (does NOT leak stack traces or internals)
  - _dispatch_ideabox_intake sends a duplicate embed for repeated URLs
  - The intake hook is invoked from _dispatch_discord_message
"""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Trigger the shared discord mock from tests/gateway/conftest.py before
# importing the production module.
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────


class _FakePlatform:
    value = "discord"


def _make_adapter():
    """Build a minimal DiscordAdapter instance for white-box testing.

    The adapter's __init__ takes a PlatformConfig; we pass a MagicMock so
    we can drive behaviour without spinning up a real client. We then
    manually wire the few attributes the intake hook reads.
    """
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._client = MagicMock()
    adapter._client.user = SimpleNamespace(id=999)
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    adapter._ideabox_view_from_content = lambda c: None  # type: ignore[attr-defined]
    # `self.name` reads `self.platform.value.title()` — provide a fake.
    adapter.platform = _FakePlatform()
    return adapter


def _text_message(channel_id="100", *, content="https://github.com/o/r", author_id="42", type_=0, parent=None, is_thread=False):
    """Build a mock Discord message living in the given text channel."""
    channel = SimpleNamespace(
        id=int(channel_id),
        type=0,  # TextChannel
        send=AsyncMock(),
    )
    if parent is not None:
        channel.parent = SimpleNamespace(id=int(parent))
    if is_thread:
        # mark as Thread
        channel.type = 11  # public thread
    author = SimpleNamespace(id=int(author_id), bot=False)
    msg = SimpleNamespace(
        id=1,
        channel=channel,
        content=content,
        author=author,
        type=type_,
        guild=SimpleNamespace(id=555),
    )
    return msg


def _forum_message(channel_id="200", parent_id="201", *, content="https://github.com/o/r2"):
    """Build a mock Discord message living inside a forum thread."""
    channel = SimpleNamespace(
        id=int(channel_id),
        type=11,  # public thread
        parent=SimpleNamespace(id=int(parent_id), type=15),  # forum parent
        send=AsyncMock(),
    )
    author = SimpleNamespace(id=42, bot=False)
    msg = SimpleNamespace(
        id=2,
        channel=channel,
        content=content,
        author=author,
        type=0,
        guild=SimpleNamespace(id=555),
    )
    return msg


# ── env-var parsing ───────────────────────────────────────────────────────


def test_ideabox_intake_channels_empty_by_default(monkeypatch):
    monkeypatch.delenv("DISCORD_IDEABOX_CHANNELS", raising=False)
    adapter = _make_adapter()
    assert adapter._ideabox_intake_channels() == set()


def test_ideabox_intake_channels_parses_csv(monkeypatch):
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "100, 200 ,300")
    adapter = _make_adapter()
    assert adapter._ideabox_intake_channels() == {"100", "200", "300"}


def test_ideabox_intake_channels_strips_whitespace_and_blanks(monkeypatch):
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "  ,, 100 , , 200, ,")
    adapter = _make_adapter()
    assert adapter._ideabox_intake_channels() == {"100", "200"}


# ── is_ideabox_intake_message ─────────────────────────────────────────────


def test_is_ideabox_intake_message_text_channel(monkeypatch):
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "100")
    adapter = _make_adapter()
    msg = _text_message(channel_id="100")
    assert adapter._is_ideabox_intake_message(msg) is True


def test_is_ideabox_intake_message_forum_parent(monkeypatch):
    """A message inside a forum thread whose parent is in the intake set
    is caught — that's the forum mode acceptance criterion."""
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "201")
    adapter = _make_adapter()
    msg = _forum_message(channel_id="777", parent_id="201")
    assert adapter._is_ideabox_intake_message(msg) is True


def test_is_ideabox_intake_message_other_channel(monkeypatch):
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "100")
    adapter = _make_adapter()
    msg = _text_message(channel_id="999")
    assert adapter._is_ideabox_intake_message(msg) is False


def test_is_ideabox_intake_message_unset_env(monkeypatch):
    monkeypatch.delenv("DISCORD_IDEABOX_CHANNELS", raising=False)
    adapter = _make_adapter()
    msg = _text_message(channel_id="100")
    assert adapter._is_ideabox_intake_message(msg) is False


# ── _dispatch_ideabox_intake (the full hook) ──────────────────────────────


def test_dispatch_ideabox_intake_skips_non_intake(monkeypatch):
    """The hook is a no-op for messages outside intake channels."""
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "100")
    adapter = _make_adapter()
    msg = _text_message(channel_id="999")

    async def run():
        return await adapter._dispatch_ideabox_intake(msg)

    consumed = asyncio.run(run())
    assert consumed is False
    msg.channel.send.assert_not_called()


def test_dispatch_ideabox_intake_text_channel_sends_embed(monkeypatch):
    """Valid GitHub URL in #idea-box text channel → embed + view sent."""
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "100")
    # Use a per-test DB to avoid cross-test pollution
    import tempfile
    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp(prefix="ideabox_hook_"))

    adapter = _make_adapter()
    msg = _text_message(channel_id="100", content="https://github.com/foo/bar")

    async def run():
        return await adapter._dispatch_ideabox_intake(msg)

    consumed = asyncio.run(run())
    assert consumed is True
    msg.channel.send.assert_called_once()
    kwargs = msg.channel.send.call_args.kwargs
    # Embed present
    assert "embed" in kwargs
    # View present (Approve/Amend/Reject buttons)
    assert "view" in kwargs
    # Embed contains triage_id in footer
    embed = kwargs["embed"]
    assert "t_" in (embed.get("footer", {}).get("text", ""))


def test_dispatch_ideabox_intake_forum_channel_sends_embed(monkeypatch):
    """Forum post whose parent is in the intake set → embed + view sent."""
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "201")
    import tempfile
    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp(prefix="ideabox_hook_"))

    adapter = _make_adapter()
    msg = _forum_message(channel_id="888", parent_id="201", content="https://github.com/x/y")

    async def run():
        return await adapter._dispatch_ideabox_intake(msg)

    consumed = asyncio.run(run())
    assert consumed is True
    msg.channel.send.assert_called_once()
    kwargs = msg.channel.send.call_args.kwargs
    assert "embed" in kwargs
    assert "view" in kwargs


def test_dispatch_ideabox_intake_invalid_sends_friendly_error(monkeypatch):
    """Malformed input → friendly error embed, NO stack trace leak."""
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "100")
    import tempfile
    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp(prefix="ideabox_hook_"))

    adapter = _make_adapter()
    msg = _text_message(channel_id="100", content="not a url or anything recognizable")

    async def run():
        return await adapter._dispatch_ideabox_intake(msg)

    consumed = asyncio.run(run())
    assert consumed is True
    msg.channel.send.assert_called_once()
    embed = msg.channel.send.call_args.kwargs["embed"]
    # No stack traces / internals leak
    description = embed.get("description", "")
    assert "Traceback" not in description
    assert "Traceback" not in str(embed)
    assert "❌" in embed.get("title", "")
    # Friendly guidance present
    assert "link" in description.lower() or "article" in description.lower()


def test_dispatch_ideabox_intake_duplicate_sends_dup_embed(monkeypatch):
    """Second submission of the same source → duplicate embed, no Kanban task."""
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "100")
    import tempfile
    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp(prefix="ideabox_hook_dup_"))

    adapter = _make_adapter()
    # Share one channel mock between both messages so we can inspect send history.
    shared_send = AsyncMock()
    def _make_msg(content):
        channel = SimpleNamespace(id=100, type=0, send=shared_send)
        author = SimpleNamespace(id=42, bot=False)
        return SimpleNamespace(
            id=1, channel=channel, content=content, author=author, type=0,
            guild=SimpleNamespace(id=555),
        )

    msg1 = _make_msg("https://github.com/dup/dup-1")
    msg2 = _make_msg("https://github.com/dup/dup-1")

    async def run():
        await adapter._dispatch_ideabox_intake(msg1)
        return await adapter._dispatch_ideabox_intake(msg2)

    consumed = asyncio.run(run())
    assert consumed is True
    # Two sends total — the second one is the duplicate embed
    assert shared_send.call_count == 2
    last_embed = shared_send.call_args_list[1].kwargs["embed"]
    assert "Duplicate" in last_embed.get("title", "")
    # The duplicate path must NOT include the approval view
    assert "view" not in shared_send.call_args_list[1].kwargs


def test_dispatch_ideabox_intake_exception_does_not_propagate(monkeypatch):
    """If the pipeline raises, the hook must still return True and send a friendly
    fallback so the message is NOT double-processed by the agent pipeline."""
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "100")

    adapter = _make_adapter()
    msg = _text_message(channel_id="100")

    async def boom(*a, **k):
        raise RuntimeError("simulated internal failure")

    with patch(
        "plugins.platforms.discord.ideabox.handler.handle_ideabox_submission",
        side_effect=boom,
    ):
        async def run():
            return await adapter._dispatch_ideabox_intake(msg)

        consumed = asyncio.run(run())
    # Even on error, we return True (consumed) so the agent pipeline skips it
    assert consumed is True
    # A friendly fallback message was sent
    msg.channel.send.assert_called_once()
    sent_text = msg.channel.send.call_args.args[0] if msg.channel.send.call_args.args else ""
    assert "⚠️" in sent_text
    # No internal exception class names leak
    assert "RuntimeError" not in sent_text
    assert "Traceback" not in sent_text


def test_dispatch_ideabox_intake_skips_bot_messages(monkeypatch):
    """Messages authored by the bot itself are not double-processed."""
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "100")
    adapter = _make_adapter()
    channel = SimpleNamespace(id=100, type=0, send=AsyncMock())
    msg = SimpleNamespace(
        id=3,
        channel=channel,
        content="https://github.com/o/r",
        author=adapter._client.user,  # the bot itself
        type=0,
        guild=None,
    )

    async def run():
        return await adapter._dispatch_ideabox_intake(msg)

    consumed = asyncio.run(run())
    assert consumed is False
    channel.send.assert_not_called()


# ── _dispatch_discord_message integration ──────────────────────────────────


def test_dispatch_discord_message_calls_ideabox_hook_first(monkeypatch):
    """Verify the new intake hook is invoked at the top of _dispatch_discord_message
    so intake-channel messages skip the normal Hermes agent pipeline."""
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "100")
    import tempfile
    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp(prefix="ideabox_hook_int_"))

    adapter = _make_adapter()
    # Set a ready event so we don't block
    import asyncio
    adapter._ready_event = asyncio.Event()
    adapter._ready_event.set()

    msg = _text_message(channel_id="100", content="https://github.com/integration/test")

    # Spy on the hook to confirm it's called
    hook_calls = []
    original_hook = adapter._dispatch_ideabox_intake

    async def spy_hook(m):
        hook_calls.append(m)
        return await original_hook(m)

    adapter._dispatch_ideabox_intake = spy_hook
    # Stub the rest of the pipeline so we never touch discord.py internals
    adapter._discord_message_admission = MagicMock(return_value=(False, False))
    adapter._handle_message = AsyncMock()

    async def run():
        return await adapter._dispatch_discord_message(msg)

    asyncio.run(run())
    assert len(hook_calls) == 1
    assert hook_calls[0] is msg
    # The intake consumed the message, so admission / _handle_message were NOT called
    adapter._discord_message_admission.assert_not_called()
    adapter._handle_message.assert_not_called()


def test_dispatch_discord_message_skips_to_pipeline_for_non_intake(monkeypatch):
    """Messages outside intake channels fall through to the normal pipeline."""
    monkeypatch.setenv("DISCORD_IDEABOX_CHANNELS", "100")
    import asyncio
    adapter = _make_adapter()
    adapter._ready_event = asyncio.Event()
    adapter._ready_event.set()

    msg = _text_message(channel_id="999", content="hello agent")
    # Hook returns False (not an intake message)
    adapter._dispatch_ideabox_intake = AsyncMock(return_value=False)
    adapter._discord_message_admission = MagicMock(return_value=(True, False))
    adapter._handle_message = AsyncMock(return_value=True)

    async def run():
        return await adapter._dispatch_discord_message(msg)

    result = asyncio.run(run())
    adapter._dispatch_ideabox_intake.assert_awaited_once_with(msg)
    adapter._handle_message.assert_awaited_once()
    assert result is True
