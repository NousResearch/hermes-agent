"""Behavioral coverage for the Nextcloud Talk adapter.

Covers the review points from PR #11458:

* session commands (``!new``/``!reset``) forward to the gateway as command
  events (with ``user_id`` in the source) instead of being reset locally
* "Thinking..." acks are scoped per conversation (FIFO), never shared
  across chats or queued turns
* the adapter declares ``splits_long_messages`` and chunks in ``send()``
* ``validate_config()`` honors a custom ``app_password_env``
* the media temp dir is derived from ``tempfile.gettempdir()`` (portable)
"""

import asyncio
import os
import tempfile
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.nextcloud_talk.adapter import (
    MEDIA_TEMP_DIR,
    MAX_MESSAGE_LENGTH,
    NextcloudTalkAdapter,
    validate_config,
)


TOKEN = "roomtok1"


def _make_adapter(monkeypatch, **extra_overrides) -> NextcloudTalkAdapter:
    monkeypatch.setenv("NEXTCLOUD_TALK_APP_PASSWORD", "app-pw")
    extra = {
        "nextcloud_url": "https://nc.example.com",
        "username": "hermes",
        "conversations": [{"token": TOKEN}],
        **extra_overrides,
    }
    return NextcloudTalkAdapter(PlatformConfig(enabled=True, extra=extra))


def _talk_msg(text, *, msg_id=101, user="niko", token=TOKEN) -> dict:
    return {
        "id": msg_id,
        "token": token,
        "actorId": user,
        "actorDisplayName": user.title(),
        "message": text,
        "messageParameters": {},
        "systemMessage": "",
    }


class TestSessionCommandsForwardToGateway:
    @pytest.mark.asyncio
    async def test_new_and_reset_are_not_handled_locally(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert await adapter._handle_command("/new", TOKEN) is None
        assert await adapter._handle_command("/reset", TOKEN) is None
        # /help stays adapter-local (documents the "!" convention)
        assert "Nextcloud Talk" in await adapter._handle_command("/help", TOKEN)

    @pytest.mark.asyncio
    async def test_bang_new_forwards_command_event_with_user_id(self, monkeypatch):
        """"!new" is rewritten to "/new" and reaches handle_message as a
        command event whose source carries the sender's user_id — the
        gateway needs it to reset the correct per-user session key."""
        adapter = _make_adapter(monkeypatch)
        adapter._client = MagicMock()
        adapter._client.send_message = AsyncMock(return_value=(True, 555, None))
        seen = []

        async def capture(event):
            seen.append(event)

        adapter.handle_message = capture

        await adapter._on_poll_message(_talk_msg("!new"), TOKEN)

        assert len(seen) == 1
        event = seen[0]
        assert event.text == "/new"
        assert event.is_command()
        assert event.get_command() == "new"
        assert event.source.user_id == "niko"

    @pytest.mark.asyncio
    async def test_local_reset_shortcut_is_gone(self, monkeypatch):
        """No adapter-side reset path remains: forwarding must not touch a
        session store."""
        adapter = _make_adapter(monkeypatch)
        assert "/new" not in adapter._LOCAL_COMMANDS
        assert "/reset" not in adapter._LOCAL_COMMANDS


class TestPendingAckScoping:
    @pytest.mark.asyncio
    async def test_acks_are_scoped_per_conversation(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            conversations=[{"token": "chat_a"}, {"token": "chat_b"}],
        )
        adapter._pending_acks = {
            "chat_a": deque([11]),
            "chat_b": deque([22]),
        }
        client = MagicMock()
        client.edit_message = AsyncMock(return_value=(True, None))
        adapter._client = client

        result = await adapter.send("chat_b", "reply for B")

        assert result.success is True
        assert result.message_id == "22"
        client.edit_message.assert_awaited_once_with("chat_b", 22, "reply for B")
        # chat_a's ack must be untouched
        assert list(adapter._pending_acks["chat_a"]) == [11]
        assert "chat_b" not in adapter._pending_acks

    @pytest.mark.asyncio
    async def test_queued_turns_consume_acks_in_fifo_order(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._pending_acks = {TOKEN: deque([31, 32])}
        client = MagicMock()
        client.edit_message = AsyncMock(return_value=(True, None))
        adapter._client = client

        first = await adapter.send(TOKEN, "first reply")
        second = await adapter.send(TOKEN, "second reply")

        assert (first.message_id, second.message_id) == ("31", "32")
        assert client.edit_message.await_args_list[0].args == (TOKEN, 31, "first reply")
        assert client.edit_message.await_args_list[1].args == (TOKEN, 32, "second reply")

    @pytest.mark.asyncio
    async def test_failed_ack_edit_falls_back_to_send(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._pending_acks = {TOKEN: deque([41])}
        client = MagicMock()
        client.edit_message = AsyncMock(return_value=(False, "HTTP 404"))
        client.send_message = AsyncMock(return_value=(True, 900, None))
        adapter._client = client

        result = await adapter.send(TOKEN, "reply")

        assert result.success is True
        assert result.message_id == "900"
        client.send_message.assert_awaited()


class TestLongMessageSplitting:
    def test_adapter_declares_native_splitting(self, monkeypatch):
        # Without this flag gateway/delivery.py truncates long content
        # (e.g. cron output) before the adapter's own chunking runs.
        adapter = _make_adapter(monkeypatch)
        assert adapter.splits_long_messages is True

    @pytest.mark.asyncio
    async def test_send_chunks_long_content(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = MagicMock()
        client.send_message = AsyncMock(return_value=(True, 1, None))
        adapter._client = client

        long_content = "x" * (MAX_MESSAGE_LENGTH * 2 + 10)
        result = await adapter.send(TOKEN, long_content)

        assert result.success is True
        sent = [c.args[1] for c in client.send_message.await_args_list]
        assert len(sent) >= 2
        # truncate_message appends " (n/m)" pagination markers per chunk —
        # strip them before verifying the content survived intact.
        import re

        joined = "".join(re.sub(r"\s*\(\d+/\d+\)$", "", chunk) for chunk in sent)
        assert joined == long_content
        assert all(len(chunk) <= MAX_MESSAGE_LENGTH for chunk in sent)


class TestValidateConfigPasswordEnv:
    def test_custom_app_password_env_is_honored(self, monkeypatch):
        monkeypatch.delenv("NEXTCLOUD_TALK_APP_PASSWORD", raising=False)
        monkeypatch.setenv("MY_CUSTOM_NC_PW", "secret")
        config = SimpleNamespace(
            extra={
                "nextcloud_url": "https://nc.example.com",
                "username": "hermes",
                "conversations": [{"token": TOKEN}],
                "app_password_env": "MY_CUSTOM_NC_PW",
            }
        )
        assert validate_config(config) is True

    def test_missing_password_fails_validation(self, monkeypatch):
        monkeypatch.delenv("NEXTCLOUD_TALK_APP_PASSWORD", raising=False)
        config = SimpleNamespace(
            extra={
                "nextcloud_url": "https://nc.example.com",
                "username": "hermes",
                "conversations": [{"token": TOKEN}],
            }
        )
        assert validate_config(config) is False


class TestPortableTempDir:
    def test_media_dir_derives_from_tempfile(self):
        assert MEDIA_TEMP_DIR == os.path.join(tempfile.gettempdir(), "hermes-media")
