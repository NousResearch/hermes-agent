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


def _make_adapter(monkeypatch, *, authorize=True, **extra_overrides) -> NextcloudTalkAdapter:
    monkeypatch.setenv("NEXTCLOUD_TALK_APP_PASSWORD", "app-pw")
    extra = {
        "nextcloud_url": "https://nc.example.com",
        "username": "hermes",
        "conversations": [{"token": TOKEN}],
        **extra_overrides,
    }
    adapter = NextcloudTalkAdapter(PlatformConfig(enabled=True, extra=extra))
    # By default wire an allow-all authorization check so behavioral tests
    # model an *authorized* sender. Pre-effect authorization (PR #11458) fails
    # closed when no check is wired, so positive-path tests must opt in.
    # Pass ``authorize=None`` to leave the adapter with no wired check, or a
    # callable ``(user_id, chat_type, chat_id, **kw) -> bool`` for custom rules.
    if authorize is not None:
        if authorize is True or authorize is False:
            _verdict = authorize
            adapter.set_authorization_check(lambda *a, **k: _verdict)
        else:
            adapter.set_authorization_check(authorize)
    return adapter


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


class TestPreEffectAuthorization:
    """PR #11458 P1: caller authorization must run BEFORE any consequential
    effect. Conversation membership only selects the lane; the wired
    ``set_authorization_check`` callback is the caller authority. An
    unallowlisted participant in a configured conversation must cause zero
    local reply / status send / attachment download / STT egress / pending-ack
    mutation, and the check must fail closed when unwired.
    """

    def _instrumented_adapter(self, monkeypatch, **kw):
        """Adapter whose every side-effect path is a spy, so a test can assert
        *nothing* fired."""
        adapter = _make_adapter(monkeypatch, **kw)
        client = MagicMock()
        client.send_message = AsyncMock(return_value=(True, 555, None))
        client.download_file = AsyncMock(return_value=(True, "/tmp/x", None))
        adapter._client = client
        adapter._stt = SimpleNamespace(transcribe=AsyncMock(return_value="TRANSCRIBED"))
        adapter._download_attachment = AsyncMock(return_value="/tmp/x")
        adapter.send = AsyncMock()
        seen = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: seen.append(e))
        return adapter, client, seen

    @pytest.mark.asyncio
    async def test_unauthorized_actor_triggers_zero_effects(self, monkeypatch):
        # actorId "mallory" is a member of the configured conversation but
        # NOT authorized by the wired callback (only "niko" is allowed).
        adapter, client, seen = self._instrumented_adapter(
            monkeypatch,
            authorize=lambda uid, *a, **k: uid == "niko",
        )
        await adapter._on_poll_message(_talk_msg("hello", user="mallory"), TOKEN)

        # Zero local reply / status send
        adapter.send.assert_not_awaited()
        client.send_message.assert_not_awaited()
        # Zero attachment download / STT
        adapter._download_attachment.assert_not_awaited()
        adapter._stt.transcribe.assert_not_awaited()
        # Zero pending-ack / cache mutation
        assert adapter._pending_acks == {}
        assert TOKEN not in adapter._chat_name_cache
        # Zero gateway hand-off
        assert seen == []

    @pytest.mark.asyncio
    async def test_unauthorized_local_command_is_not_answered(self, monkeypatch):
        """Even an adapter-local ``!help`` must not be answered for an
        unauthorized actor (it is a consequential ``self.send`` reply)."""
        adapter, client, seen = self._instrumented_adapter(
            monkeypatch, authorize=False,
        )
        await adapter._on_poll_message(_talk_msg("!help", user="mallory"), TOKEN)
        adapter.send.assert_not_awaited()
        client.send_message.assert_not_awaited()
        assert seen == []

    @pytest.mark.asyncio
    async def test_unauthorized_audio_triggers_no_download_or_stt(self, monkeypatch):
        adapter, client, seen = self._instrumented_adapter(
            monkeypatch, authorize=False,
        )
        msg = _talk_msg("", user="mallory")
        msg["messageParameters"] = {
            "file": {"type": "file", "name": "memo.mp3", "path": "memo.mp3",
                     "mimetype": "audio/mpeg", "size": "1234"},
        }
        await adapter._on_poll_message(msg, TOKEN)
        adapter._download_attachment.assert_not_awaited()
        adapter._stt.transcribe.assert_not_awaited()
        assert seen == []

    @pytest.mark.asyncio
    async def test_unwired_authorization_fails_closed(self, monkeypatch):
        """No wired check (legacy/unknown) must NOT fall through to trust —
        it fails closed for consequential effects."""
        adapter, client, seen = self._instrumented_adapter(
            monkeypatch, authorize=None,
        )
        assert adapter._authorization_check is None
        await adapter._on_poll_message(_talk_msg("hello", user="niko"), TOKEN)
        adapter.send.assert_not_awaited()
        client.send_message.assert_not_awaited()
        assert seen == []

    @pytest.mark.asyncio
    async def test_authorized_actor_still_gets_normal_behavior(self, monkeypatch):
        """Positive side of the boundary: an authorized actor reaches the
        gateway with the normal MessageEvent."""
        adapter = _make_adapter(monkeypatch, authorize=lambda uid, *a, **k: uid == "niko")
        adapter._client = MagicMock()
        adapter._client.send_message = AsyncMock(return_value=(True, 555, None))
        seen = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: seen.append(e))

        await adapter._on_poll_message(_talk_msg("hello world", user="niko"), TOKEN)

        assert len(seen) == 1
        assert seen[0].text == "hello world"
        assert seen[0].source.user_id == "niko"

    @pytest.mark.asyncio
    async def test_allow_all_only_through_canonical_callback(self, monkeypatch):
        """"Allow-all" must be expressed via the same wired callback returning
        True — there is no ad-hoc adapter-side allowlist parser."""
        adapter = _make_adapter(monkeypatch, authorize=lambda *a, **k: True)
        adapter._client = MagicMock()
        adapter._client.send_message = AsyncMock(return_value=(True, 555, None))
        seen = []
        adapter.handle_message = AsyncMock(side_effect=lambda e: seen.append(e))

        # An arbitrary actor is admitted only because the callback says True.
        await adapter._on_poll_message(_talk_msg("hi", user="anyone"), TOKEN)
        assert len(seen) == 1
        # And the adapter carries no ad-hoc allowlist parser / env of its own.
        assert not hasattr(adapter, "_allowed_users")
        assert not hasattr(adapter, "_parse_allowed_users")

    @pytest.mark.asyncio
    async def test_authorization_uses_wired_callback_args_not_ambient(self, monkeypatch):
        """Multiplex/profile case: the wired callback receives the projected
        (actorId, chat_type, chat_id) — the transport profile's authority — so
        a profile-specific allowlist is what decides, not ambient state."""
        calls = []

        def check(user_id, chat_type=None, chat_id=None, **kw):
            calls.append((user_id, chat_type, chat_id))
            return user_id == "niko"

        adapter = _make_adapter(monkeypatch, authorize=check)
        adapter._client = MagicMock()
        adapter._client.send_message = AsyncMock(return_value=(True, 555, None))
        adapter.handle_message = AsyncMock()

        await adapter._on_poll_message(_talk_msg("hi", user="niko"), TOKEN)

        assert calls == [("niko", adapter._classify_chat(TOKEN), TOKEN)]
