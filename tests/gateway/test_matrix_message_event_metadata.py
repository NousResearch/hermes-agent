"""Tests for Matrix MessageEvent metadata (sender, reply context).

The matrix adapter builds MessageEvent from inbound room events. Other adapters
(Signal, Slack, Telegram, Discord, Mattermost, IRC) populate the sender /
reply_to_* fields on MessageEvent so the gateway can:
  - prepend "[Name] message" to user prompt text in shared-multi-user sessions
  - render "[Replying to: ...]" with the replied-to author's name
  - render "[Replying to your previous message: ...]" when the reply target
    is the bot itself

Matrix historically dropped these fields, leaving the LLM unable to tell
who said what in a shared room. Agents saw only bare text strings, so a
self-emitted phantom interruption (e.g. "[This response was interrupted by
a user correction.]") looked identical to a real user message and triggered
endless reply loops.

These tests assert the invariant: every inbound Matrix text/media message
that the adapter dispatches via handle_message() must carry the sender's
MXID and display name on the MessageEvent (not buried in `source`), and
reply-targeted messages must carry the replied-to message's text and author.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import time

import pytest


def _make_adapter(require_mention=False, auto_thread=False, monkeypatch=None):
    """Create a MatrixAdapter with mocked config and bypassed display-name lookup."""
    # MATRIX_REQUIRE_MENTION and MATRIX_AUTO_THREAD are read once at __init__,
    # so they must be set in the environment before constructing the adapter.
    if monkeypatch is not None:
        monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "true" if require_mention else "false")
        monkeypatch.setenv("MATRIX_AUTO_THREAD", "true" if auto_thread else "false")
    else:
        import os
        os.environ["MATRIX_REQUIRE_MENTION"] = "true" if require_mention else "false"
        os.environ["MATRIX_AUTO_THREAD"] = "true" if auto_thread else "false"

    from plugins.platforms.matrix.adapter import MatrixAdapter

    from gateway.config import PlatformConfig

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    # Bypass mautrix state_store lookup — fall back to localpart.
    adapter._client = None
    # Stub the DM/identity lookup chain so we don't need a real mautrix
    # client. _is_allowed_matrix_room_event -> _is_dm_room ->
    # _resolve_room_identity, and _resolve_message_context ->
    # _resolve_room_identity (used to fetch chat_type).
    identity = SimpleNamespace(
        display_name="Test Room",
        room_topic=None,
        server_name="example.org",
        chat_type="dm",  # DM shortcut so we bypass MATRIX_ALLOWED_ROOMS
    )
    adapter._resolve_room_identity = AsyncMock(return_value=identity)
    return adapter


def _make_event(
    body,
    sender="@alice:example.org",
    event_id="$evt1",
    room_id="!room1:example.org",
    thread_id=None,
    in_reply_to_event_id=None,
):
    """Build a fake Matrix room message event with optional reply context."""
    content = {"body": body, "msgtype": "m.text"}

    relates_to = {}
    if thread_id:
        relates_to["rel_type"] = "m.thread"
        relates_to["event_id"] = thread_id
    if in_reply_to_event_id:
        relates_to["m.in_reply_to"] = {"event_id": in_reply_to_event_id}
    if relates_to:
        content["m.relates_to"] = relates_to

    return SimpleNamespace(
        sender=sender,
        event_id=event_id,
        room_id=room_id,
        # Use *recent* timestamp so we don't fall into the startup-grace
        # filter (which drops events older than `_startup_ts - 5s`). The
        # production adapter ignores real events that pre-date gateway
        # start; tests must use "now-ish" timestamps.
        timestamp=int(time.time() * 1000),
        content=content,
    )


# ---------------------------------------------------------------------------
# Sender metadata on MessageEvent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_message_carries_sender_mxid(monkeypatch):
    """The MXID of the message author must reach MessageEvent, not just source.

    Without this, downstream consumers cannot tell who said what.
    """
    adapter = _make_adapter(monkeypatch=monkeypatch)
    adapter._startup_ts = time.time() - 10

    event = _make_event("hello world", sender="@alice:example.org")
    await adapter._on_room_message(event)

    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.user_id == "@alice:example.org"
    # Display name fallback to localpart when no state_store is available.
    assert msg.user_name == "alice"


@pytest.mark.asyncio
async def test_text_message_carries_sender_for_different_user(monkeypatch):
    """MXID propagation must work for arbitrary senders, not just alice."""
    adapter = _make_adapter(monkeypatch=monkeypatch)
    adapter._startup_ts = time.time() - 10

    event = _make_event("hi from bob", sender="@bob:chat.example.org")
    await adapter._on_room_message(event)

    msg = adapter.handle_message.await_args.args[0]
    assert msg.user_id == "@bob:chat.example.org"
    assert msg.user_name == "bob"


# ---------------------------------------------------------------------------
# Reply context on MessageEvent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_carries_target_text_and_author(monkeypatch):
    """Reply messages must carry the replied-to message's text + author MXID/name.

    Other adapters (Signal, Slack, Telegram) already populate these fields.
    Matrix historically did not, so "[Replying to: ...]" rendered only the
    quoted text without any indicator of *who* the user was replying to.
    """
    adapter = _make_adapter(monkeypatch=monkeypatch)
    adapter._startup_ts = time.time() - 10

    # Reply body in Matrix is: "> <@user:server> quoted body\n\nactual reply"
    reply_body = "> <@carol:example.org> original question\n\nbecause reasons"
    event = _make_event(
        reply_body,
        sender="@dave:example.org",
        in_reply_to_event_id="$target1",
    )
    await adapter._on_room_message(event)

    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]

    # The reply target pointer must reach MessageEvent.
    assert msg.reply_to_message_id == "$target1"
    # The replied-to message's body must reach MessageEvent (stripped of
    # the "> " quote prefix). Used by gateway/run.py:16015 to render
    # [Replying to: "..."] in the LLM prompt.
    assert msg.reply_to_text is not None
    assert "original question" in msg.reply_to_text
    # The replied-to author's MXID must reach MessageEvent. Used to detect
    # "Replying to your previous message" vs "Replying to another user's message".
    assert msg.reply_to_author_id == "@carol:example.org"
    assert msg.reply_to_author_name == "carol"


@pytest.mark.asyncio
async def test_non_reply_message_has_no_reply_context(monkeypatch):
    """A non-reply message must not spuriously set reply_to_* fields."""
    adapter = _make_adapter(monkeypatch=monkeypatch)
    adapter._startup_ts = time.time() - 10

    event = _make_event("plain message, no reply")
    await adapter._on_room_message(event)

    msg = adapter.handle_message.await_args.args[0]
    assert msg.reply_to_message_id is None
    assert msg.reply_to_text is None
    assert msg.reply_to_author_id is None
    assert msg.reply_to_author_name is None


# ---------------------------------------------------------------------------
# Media message sender + reply context (sibling gap fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_message_carries_sender_and_reply_context(monkeypatch):
    """Media messages must carry sender + reply context, same as text messages.

    A user replying to a photo in Matrix produces a media event with the
    same inline ``> <@user:server> ...`` reply fallback. The media handler
    must parse it and populate reply_to_* fields, not just the text handler.
    """
    adapter = _make_adapter(monkeypatch=monkeypatch)
    adapter._startup_ts = time.time() - 10

    # Simulate a reply to a media message — the body carries the reply
    # fallback prefix. We don't need a real mxc:// URL since the adapter
    # validates URL format and rejects non-MXC; use a valid mxc:// URL.
    reply_body = "> <@erin:example.org> nice photo\n\n"
    event = SimpleNamespace(
        sender="@frank:example.org",
        event_id="$media_evt1",
        room_id="!room1:example.org",
        timestamp=int(time.time() * 1000),
        content={
            "body": reply_body,
            "msgtype": "m.image",
            "url": "mxc://example.org/abc123",
            "info": {"mimetype": "image/png", "size": 1024},
            "m.relates_to": {"m.in_reply_to": {"event_id": "$target_media1"}},
        },
    )
    await adapter._on_room_message(event)

    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]

    # Sender metadata must be present on media messages too.
    assert msg.user_id == "@frank:example.org"
    assert msg.user_name == "frank"
    # Reply context must be parsed from the fallback.
    assert msg.reply_to_message_id == "$target_media1"
    assert msg.reply_to_text is not None
    assert "nice photo" in msg.reply_to_text
    assert msg.reply_to_author_id == "@erin:example.org"
    assert msg.reply_to_author_name == "erin"
