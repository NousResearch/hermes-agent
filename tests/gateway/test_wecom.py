"""Tests for the WeCom platform adapter."""

import asyncio
import base64
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


class TestWeComRequirements:
    def test_returns_false_without_aiohttp(self, monkeypatch):
        monkeypatch.setattr("plugins.platforms.wecom.adapter.AIOHTTP_AVAILABLE", False)
        monkeypatch.setattr("plugins.platforms.wecom.adapter.HTTPX_AVAILABLE", True)
        from plugins.platforms.wecom.adapter import check_wecom_requirements

        assert check_wecom_requirements() is False


class TestWeComAdapterInit:
    def test_declares_non_editable_message_capability(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        assert WeComAdapter.SUPPORTS_MESSAGE_EDITING is False


class TestWeComAdapterAuthzScope:
    """dm_policy/allowlist reads must honor the profile secret scope under
    multiplexing (#93522): a secondary profile's own scope is authoritative
    and must not inherit the default profile's process-env authorization."""

    @pytest.fixture()
    def multiplex_on(self):
        from agent import secret_scope

        previous = secret_scope.is_multiplex_active()
        secret_scope.set_multiplex_active(True)
        try:
            yield
        finally:
            secret_scope.set_multiplex_active(previous)

    def test_scoped_construction_reads_authz_from_scope_not_environ(self, multiplex_on, monkeypatch):
        from agent import secret_scope
        from plugins.platforms.wecom.adapter import WeComAdapter

        monkeypatch.setenv("WECOM_DM_POLICY", "pairing")
        monkeypatch.setenv("WECOM_ALLOWED_USERS", "default-user")
        token = secret_scope.set_secret_scope(
            {"WECOM_DM_POLICY": "allowlist", "WECOM_ALLOWED_USERS": "scoped-user"}
        )
        try:
            adapter = WeComAdapter(PlatformConfig(enabled=True))
        finally:
            secret_scope.reset_secret_scope(token)
        assert adapter._dm_policy == "allowlist"
        assert adapter._allow_from == ["scoped-user"]

    def test_scoped_miss_does_not_admit_default_profiles_allowlist(self, multiplex_on, monkeypatch):
        from agent import secret_scope
        from plugins.platforms.wecom.adapter import WeComAdapter

        monkeypatch.setenv("WECOM_DM_POLICY", "allowlist")
        monkeypatch.setenv("WECOM_ALLOWED_USERS", "default-user")
        token = secret_scope.set_secret_scope({"SOMETHING_ELSE": "x"})
        try:
            adapter = WeComAdapter(PlatformConfig(enabled=True))
        finally:
            secret_scope.reset_secret_scope(token)
        assert adapter._dm_policy == "pairing"
        assert adapter._allow_from == []

    def test_scoped_construction_reads_bot_id_from_scope_not_environ(self, multiplex_on, monkeypatch):
        """bot_id must honor the same scope as its neighboring _secret read
        (both are read on adjacent lines in __init__) -- a secondary profile's
        own bot_id must never fall back to the default profile's os.environ
        value."""
        from agent import secret_scope
        from plugins.platforms.wecom.adapter import WeComAdapter

        monkeypatch.setenv("WECOM_BOT_ID", "default-profile-bot-id")
        monkeypatch.setenv("WECOM_SECRET", "default-profile-secret")
        token = secret_scope.set_secret_scope(
            {"WECOM_BOT_ID": "scoped-bot-id", "WECOM_SECRET": "scoped-secret"}
        )
        try:
            adapter = WeComAdapter(PlatformConfig(enabled=True))
        finally:
            secret_scope.reset_secret_scope(token)
        assert adapter._bot_id == "scoped-bot-id"
        assert adapter._secret == "scoped-secret"

    def test_scoped_miss_does_not_leak_default_profiles_bot_id(self, multiplex_on, monkeypatch):
        from agent import secret_scope
        from plugins.platforms.wecom.adapter import DEFAULT_WS_URL, WeComAdapter

        monkeypatch.setenv("WECOM_BOT_ID", "default-profile-bot-id")
        monkeypatch.setenv("WECOM_WEBSOCKET_URL", "wss://default-profile.example/ws")
        token = secret_scope.set_secret_scope({"SOMETHING_ELSE": "x"})
        try:
            adapter = WeComAdapter(PlatformConfig(enabled=True))
        finally:
            secret_scope.reset_secret_scope(token)
        assert adapter._bot_id == ""
        assert adapter._ws_url == DEFAULT_WS_URL


class TestWeComConnect:

    @pytest.mark.asyncio
    async def test_connect_records_handshake_failure_details(self, monkeypatch):
        import plugins.platforms.wecom.adapter as wecom_module
        from plugins.platforms.wecom.adapter import WeComAdapter

        class DummyClient:
            async def aclose(self):
                return None

        monkeypatch.setattr(wecom_module, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr(wecom_module, "HTTPX_AVAILABLE", True)
        monkeypatch.setattr(
            wecom_module,
            "httpx",
            SimpleNamespace(AsyncClient=lambda **kwargs: DummyClient()),
        )

        adapter = WeComAdapter(
            PlatformConfig(enabled=True, extra={"bot_id": "bot-1", "secret": "secret-1"})
        )
        adapter._open_connection = AsyncMock(side_effect=RuntimeError("invalid secret (errcode=40013)"))

        success = await adapter.connect()

        assert success is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "wecom_connect_error"
        assert "invalid secret" in (adapter.fatal_error_message or "")


class TestWeComQrScan:
    @patch("plugins.platforms.wecom.adapter.time")
    @patch("plugins.platforms.wecom.adapter.json.loads")
    @patch("plugins.platforms.wecom.adapter.logger")
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_qr_scan_timeout_uses_monotonic_clock(
        self,
        mock_request,
        mock_urlopen,
        _mock_logger,
        mock_json_loads,
        mock_time,
    ):
        from plugins.platforms.wecom.adapter import qr_scan_for_bot_info

        generate_resp = MagicMock()
        generate_resp.read.return_value = b'{"data":{"scode":"abc","auth_url":"https://example.com/qr"}}'
        generate_resp.__enter__.return_value = generate_resp
        generate_resp.__exit__.return_value = False

        poll_resp = MagicMock()
        poll_resp.read.return_value = b'{"data":{"status":"pending"}}'
        poll_resp.__enter__.return_value = poll_resp
        poll_resp.__exit__.return_value = False

        mock_urlopen.side_effect = [generate_resp, poll_resp]
        mock_json_loads.side_effect = [
            {"data": {"scode": "abc", "auth_url": "https://example.com/qr"}},
            {"data": {"status": "pending"}},
        ]
        mock_time.monotonic.side_effect = [1000, 1000.2, 1001.1]
        mock_time.time.side_effect = [1000, 900, 901, 902]
        mock_time.sleep = MagicMock()

        with patch("builtins.print"), patch.dict("sys.modules", {"qrcode": None}):
            result = qr_scan_for_bot_info(timeout_seconds=1)

        assert result is None
        assert mock_urlopen.call_count == 2


class TestWeComReplyMode:

    @pytest.mark.asyncio
    async def test_send_image_file_uses_passive_reply_media_when_reply_context_exists(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._reply_req_ids["msg-1"] = "req-1"
        adapter._prepare_outbound_media = AsyncMock(
            return_value={
                "data": b"image-bytes",
                "content_type": "image/png",
                "file_name": "demo.png",
                "detected_type": "image",
                "final_type": "image",
                "rejected": False,
                "reject_reason": None,
                "downgraded": False,
                "downgrade_note": None,
            }
        )
        adapter._upload_media_bytes = AsyncMock(return_value={"media_id": "media-1", "type": "image"})
        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "req-1"}, "errcode": 0}
        )

        result = await adapter.send_image_file("chat-123", "/tmp/demo.png", reply_to="msg-1")

        assert result.success is True
        adapter._send_reply_request.assert_awaited_once()
        args = adapter._send_reply_request.await_args.args
        assert args[0] == "req-1"
        assert args[1] == {"msgtype": "image", "image": {"media_id": "media-1"}}


class TestExtractText:

    def test_extracts_mixed_text(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        body = {
            "msgtype": "mixed",
            "mixed": {
                "msg_item": [
                    {"msgtype": "text", "text": {"content": "part1"}},
                    {"msgtype": "image", "image": {"url": "https://example.com/x.png"}},
                    {"msgtype": "text", "text": {"content": "part2"}},
                ]
            },
        }
        text, _reply_text = WeComAdapter._extract_text(body)
        assert text == "part1\npart2"


class TestCallbackDispatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cmd", ["aibot_msg_callback", "aibot_callback"])
    async def test_dispatch_accepts_new_and_legacy_callback_cmds(self, cmd):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._on_message = AsyncMock()

        await adapter._dispatch_payload({"cmd": cmd, "headers": {"req_id": "req-1"}, "body": {}})

        adapter._on_message.assert_awaited_once()


class TestPolicyHelpers:

    def test_dm_allowlist_honors_env_only_allowed_users(self, monkeypatch):
        """Env-only setup (WECOM_DM_POLICY + WECOM_ALLOWED_USERS, no config
        ``extra``) must populate the DM allowlist. Otherwise ``dm_policy:
        allowlist`` runs with an empty allowlist and drops every listed user
        at intake — the documented env vars become no-ops."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        monkeypatch.setenv("WECOM_DM_POLICY", "allowlist")
        monkeypatch.setenv("WECOM_ALLOWED_USERS", "user-1, user-2")

        adapter = WeComAdapter(PlatformConfig(enabled=True))

        assert adapter._dm_policy == "allowlist"
        assert adapter._allow_from == ["user-1", "user-2"]
        assert adapter._is_dm_allowed("user-1") is True
        assert adapter._is_dm_allowed("user-2") is True
        assert adapter._is_dm_allowed("stranger") is False


    def test_pairing_group_policy_blocks_without_explicit_group_allow_from(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(enabled=True, extra={"group_policy": "pairing"})
        )

        assert adapter._is_group_allowed("group-1", "user-1") is False


class TestMediaHelpers:
    def test_detect_wecom_media_type(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        assert WeComAdapter._detect_wecom_media_type("image/png") == "image"
        assert WeComAdapter._detect_wecom_media_type("video/mp4") == "video"
        assert WeComAdapter._detect_wecom_media_type("audio/amr") == "voice"
        assert WeComAdapter._detect_wecom_media_type("application/pdf") == "file"

    def test_voice_non_amr_downgrades_to_file(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        result = WeComAdapter._apply_file_size_limits(128, "voice", "audio/mpeg")

        assert result["final_type"] == "file"
        assert result["downgraded"] is True
        assert "AMR" in (result["downgrade_note"] or "")


class TestMediaUpload:


    @pytest.mark.asyncio
    async def test_download_remote_bytes_blocks_connect_time_rebind(self, monkeypatch):
        import httpcore
        from httpcore._backends.auto import AutoBackend
        from plugins.platforms.wecom.adapter import WeComAdapter
        from tools.url_safety import SSRFConnectionBlocked

        for proxy_var in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            monkeypatch.delenv(proxy_var, raising=False)

        answers = iter(("93.184.216.34", "169.254.169.254"))

        def fake_getaddrinfo(_host, port, *_args, **_kwargs):
            ip = next(answers)
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))
            ]

        connect_attempts = []

        async def fake_connect_tcp(
            _self,
            host,
            port,
            timeout=None,
            local_address=None,
            socket_options=None,
        ):
            connect_attempts.append((host, port))
            raise httpcore.ConnectError("stop before network")

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        monkeypatch.setattr(AutoBackend, "connect_tcp", fake_connect_tcp)

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        with pytest.raises(SSRFConnectionBlocked):
            await adapter._download_remote_bytes(
                "http://rebind.example/file.bin", max_bytes=1024
            )

        assert connect_attempts == []


class TestSend:


    @pytest.mark.asyncio
    async def test_send_voice_sends_caption_and_downgrade_note(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._prepare_outbound_media = AsyncMock(
            return_value={
                "data": b"voice-bytes",
                "content_type": "audio/mpeg",
                "file_name": "voice.mp3",
                "detected_type": "voice",
                "final_type": "file",
                "rejected": False,
                "reject_reason": None,
                "downgraded": True,
                "downgrade_note": "语音格式 audio/mpeg 不支持，企微仅支持 AMR 格式，已转为文件格式发送",
            }
        )
        adapter._upload_media_bytes = AsyncMock(return_value={"media_id": "media-1", "type": "file"})
        adapter._send_media_message = AsyncMock(return_value={"headers": {"req_id": "req-media"}, "errcode": 0})
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="msg-1"))

        result = await adapter.send_voice("chat-123", "/tmp/voice.mp3", caption="listen")

        assert result.success is True
        adapter._send_media_message.assert_awaited_once_with("chat-123", "file", "media-1")
        assert adapter.send.await_count == 2
        adapter.send.assert_any_await(chat_id="chat-123", content="listen", reply_to=None)
        adapter.send.assert_any_await(
            chat_id="chat-123",
            content="ℹ️ 语音格式 audio/mpeg 不支持，企微仅支持 AMR 格式，已转为文件格式发送",
            reply_to=None,
        )


    @pytest.mark.asyncio
    async def test_approval_confirmation_uses_proactive_send(self):
        """Regression: force_proactive_send=True must use APP_CMD_SEND to avoid
        consuming the req_id that the post-approval stream needs. Passive reply
        on the same req_id causes WeCom to render the stream seed as empty bubble."""
        from plugins.platforms.wecom.adapter import APP_CMD_SEND, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        # Simulate a cached req_id from the user's /approve message
        adapter._last_chat_req_ids["chat-123"] = "req-approve"
        adapter._send_request = AsyncMock(return_value={"headers": {"req_id": "req-approve"}, "errcode": 0})
        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "req-approve"}, "errcode": 0}
        )

        result = await adapter.send(
            "chat-123",
            "✅ Approved 1 command. Continuing...",
            metadata={"is_approval_prompt": True, "force_proactive_send": True},
        )

        assert result.success is True
        # Must use APP_CMD_SEND (proactive), NOT _send_reply_request (passive)
        adapter._send_request.assert_awaited_once_with(
            APP_CMD_SEND,
            {
                "chatid": "chat-123",
                "msgtype": "markdown",
                "markdown": {"content": "✅ Approved 1 command. Continuing..."},
            },
        )
        adapter._send_reply_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_approval_request_prompt_uses_passive_reply(self):
        """is_approval_prompt alone (without force_proactive_send) must still use
        passive reply. The initial approval *request* prompt needs passive reply
        because groups cannot use APP_CMD_SEND."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["group-chat"] = "req-user-msg"
        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "req-user-msg"}, "errcode": 0}
        )
        adapter._send_request = AsyncMock(return_value={"errcode": 0})

        result = await adapter.send(
            "group-chat",
            "⚠️ Dangerous command requires approval...",
            metadata={"is_approval_prompt": True},  # No force_proactive_send
        )

        assert result.success is True
        # Should use passive reply (preserving req_id for group delivery)
        adapter._send_reply_request.assert_awaited_once()
        adapter._send_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_with_active_stream_still_uses_passive_reply(self):
        """send() should NOT force proactive when a stream is active —
        that's too broad and breaks group delivery. Only explicit
        force_proactive_send metadata triggers proactive mode."""
        from plugins.platforms.wecom.adapter import WeComAdapter, StreamTurn

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-123"] = "req-latest"
        # Simulate an active stream turn for this chat
        turn = StreamTurn("chat-123", "req-latest")
        adapter._stream_turns["chat-123:turn-1"] = turn

        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "req-latest"}, "errcode": 0}
        )
        adapter._send_request = AsyncMock(return_value={"errcode": 0})

        result = await adapter.send("chat-123", "Some status message")

        assert result.success is True
        # Should still use passive reply — active stream doesn't force proactive
        adapter._send_reply_request.assert_awaited_once()
        adapter._send_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_force_proactive_falls_back_to_passive_for_groups(self):
        """Regression: force_proactive_send must NOT use APP_CMD_SEND in group chats.
        WeCom AI Bots cannot initiate APP_CMD_SEND in groups — only passive reply
        (APP_CMD_RESPONSE) bound to a req_id works."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["group-chat"] = "req-approve"
        # Mark this chat as a group
        adapter._group_chat_ids.add("group-chat")

        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "req-approve"}, "errcode": 0}
        )
        adapter._send_request = AsyncMock(return_value={"errcode": 0})

        result = await adapter.send(
            "group-chat",
            "✅ Approved 1 command. Continuing...",
            metadata={"is_approval_prompt": True, "force_proactive_send": True},
        )

        assert result.success is True
        # Group chats must fall back to passive reply even with force_proactive_send
        adapter._send_reply_request.assert_awaited_once()
        adapter._send_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_group_send_fails_early_without_req_id(self):
        """Group chats with no cached req_id must fail with a clear error
        instead of attempting APP_CMD_SEND (which WeCom will reject)."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        # No req_id cached for this group
        adapter._group_chat_ids.add("group-no-req")
        adapter._send_request = AsyncMock(return_value={"errcode": 0})

        result = await adapter.send("group-no-req", "hello group")

        assert result.success is False
        assert "req_id" in (result.error or "").lower()
        # Should NOT attempt APP_CMD_SEND
        adapter._send_request.assert_not_awaited()


class TestInboundMessages:
    @pytest.mark.asyncio
    async def test_on_message_builds_event(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={"group_policy": "allowlist", "group_allow_from": ["group-1"]},
            )
        )
        adapter._text_batch_delay_seconds = 0  # disable batching for tests
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=(["/tmp/test.png"], ["image/png"]))

        payload = {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": "req-1"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-1",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "hello"},
            },
        }

        await adapter._on_message(payload)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "hello"
        assert event.source.chat_id == "group-1"
        assert event.source.user_id == "user-1"
        assert event.media_urls == ["/tmp/test.png"]
        assert event.media_types == ["image/png"]


class TestWeComZombieSessionFix:
    """Tests for PR #11572 — device_id, markdown reply, group req_id fallback."""


    @pytest.mark.asyncio
    async def test_on_message_does_not_cache_blocked_sender_req_id(self):
        """Blocked chats shouldn't populate the proactive-send fallback cache."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={"group_policy": "allowlist", "group_allow_from": ["group-ok"]},
            )
        )
        adapter.handle_message = AsyncMock()
        adapter._extract_media = AsyncMock(return_value=([], []))

        payload = {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": "req-abc"},
            "body": {
                "msgid": "msg-1",
                "chatid": "group-blocked",
                "chattype": "group",
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": "hi"},
            },
        }

        await adapter._on_message(payload)
        adapter.handle_message.assert_not_awaited()
        assert "group-blocked" not in adapter._last_chat_req_ids

    def test_remember_chat_req_id_is_bounded(self):
        from plugins.platforms.wecom.adapter import DEDUP_MAX_SIZE, WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        for i in range(DEDUP_MAX_SIZE + 50):
            adapter._remember_chat_req_id(f"chat-{i}", f"req-{i}")
        assert len(adapter._last_chat_req_ids) <= DEDUP_MAX_SIZE
        # The most recently remembered chat must still be present.
        latest = f"chat-{DEDUP_MAX_SIZE + 49}"
        assert adapter._last_chat_req_ids[latest] == f"req-{DEDUP_MAX_SIZE + 49}"


    @pytest.mark.asyncio
    async def test_proactive_group_send_falls_back_to_cached_req_id(self):
        """Sending into a group without reply_to should use the last cached
        req_id via APP_CMD_RESPONSE — WeCom AI Bots cannot initiate APP_CMD_SEND
        in group chats (errcode 600039)."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["group-1"] = "inbound-req-42"
        adapter._send_reply_request = AsyncMock(
            return_value={"headers": {"req_id": "inbound-req-42"}, "errcode": 0}
        )
        adapter._send_request = AsyncMock(
            return_value={"headers": {"req_id": "new"}, "errcode": 0}
        )

        result = await adapter.send("group-1", "ping", reply_to=None)

        assert result.success is True
        # Must route through reply (APP_CMD_RESPONSE), not proactive send.
        adapter._send_reply_request.assert_awaited_once()
        adapter._send_request.assert_not_awaited()
        args = adapter._send_reply_request.await_args.args
        assert args[0] == "inbound-req-42"
        assert args[1]["msgtype"] == "markdown"
        assert args[1]["markdown"]["content"] == "ping"


class TestTextBatchFlushRace:
    """Regression tests for the cancel-delivery race in _flush_text_batch.

    When asyncio.sleep() fires and Task.cancel() is called before the task
    runs, CPython sets _must_cancel but cannot cancel the already-done sleep
    future.  CancelledError is then delivered at the *next* await
    (handle_message), after the task has already popped the event — the
    superseding task sees an empty batch and silently drops the message.
    The fix adds a synchronous task-registry check between the sleep and
    the pop so a superseded task returns before touching the event.
    """

    @pytest.mark.asyncio
    async def test_superseded_task_does_not_pop_or_process_event(self):
        """A flush task that has been superseded must leave the event in the
        batch dict for the new task to handle."""
        from gateway.platforms.base import MessageEvent, MessageType
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0

        key = "test-session"
        event = MessageEvent(text="hello", message_type=MessageType.TEXT)
        adapter._pending_text_batches[key] = event

        handle_calls = []

        async def fake_handle(evt):
            handle_calls.append(evt)

        adapter.handle_message = fake_handle

        # Create T1 and register it.
        t1 = asyncio.create_task(adapter._flush_text_batch(key))
        adapter._pending_text_batch_tasks[key] = t1

        # Simulate T2 superseding T1 before T1 wakes from sleep.
        t2 = asyncio.create_task(asyncio.sleep(0.2))
        adapter._pending_text_batch_tasks[key] = t2

        # Yield long enough for T1's sleep(0) to complete and T1 to run.
        await asyncio.sleep(0.05)

        t2.cancel()
        try:
            await t2
        except asyncio.CancelledError:
            pass

        # T1 must have returned without processing or removing the event.
        assert handle_calls == [], "superseded task must not call handle_message"
        assert adapter._pending_text_batches.get(key) is event, (
            "superseded task must not pop the event"
        )

    @pytest.mark.asyncio
    async def test_active_task_processes_event_normally(self):
        """When the task is not superseded it must still process the event."""
        from gateway.platforms.base import MessageEvent, MessageType
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._text_batch_delay_seconds = 0

        key = "test-session"
        event = MessageEvent(text="world", message_type=MessageType.TEXT)
        adapter._pending_text_batches[key] = event

        handle_calls = []

        async def fake_handle(evt):
            handle_calls.append(evt)

        adapter.handle_message = fake_handle

        t1 = asyncio.create_task(adapter._flush_text_batch(key))
        adapter._pending_text_batch_tasks[key] = t1

        # No superseding task — T1 should process normally.
        await asyncio.sleep(0.05)

        assert handle_calls == [event], "active task must call handle_message"
        assert adapter._pending_text_batches.get(key) is None, (
            "active task must pop the event after processing"
        )


class TestAttachmentTextMerge:
    """WeCom sends "image + text" as two separate inbound callbacks (an
    attachment-only frame, then a text frame ~hundreds of ms later).

    Dispatching the attachment immediately spawns an agent run that the
    trailing text then "interrupts" (junk "⚡ Interrupting" + "✅" acks).
    The adapter buffers an attachment-only message on the existing text-batch
    machinery for a short merge window so the following text merges into ONE
    dispatched event. These tests exercise the real _on_message path.
    """

    @staticmethod
    def _make_adapter(merge_delay: float = 0.15):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "dm_policy": "open",
                    "attachment_text_merge_delay_seconds": merge_delay,
                },
            )
        )
        # DM open policy needs the opt-in env flag; force intake open.
        adapter._is_dm_intake_allowed = lambda sender_id: True
        # Keep the text-split batch window tiny so tests are fast.
        adapter._text_batch_delay_seconds = 0.05
        adapter.handle_message = AsyncMock()
        return adapter

    @staticmethod
    def _image_payload(msgid: str, media):
        return {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": f"req-{msgid}"},
            "body": {
                "msgid": msgid,
                "from": {"userid": "user-1"},
                "msgtype": "image",
                "image": {"url": "https://example.com/x.png"},
                "_media": media,
            },
        }

    @staticmethod
    def _text_payload(msgid: str, content: str):
        return {
            "cmd": "aibot_msg_callback",
            "headers": {"req_id": f"req-{msgid}"},
            "body": {
                "msgid": msgid,
                "from": {"userid": "user-1"},
                "msgtype": "text",
                "text": {"content": content},
            },
        }

    @pytest.mark.asyncio
    async def test_image_then_text_merge_into_one_event(self):
        """image-then-text within the window → ONE dispatched event carrying
        both the media and the text, and NO immediate dispatch of the image
        (so the busy-handler interrupt path is never triggered)."""
        adapter = self._make_adapter(merge_delay=0.2)

        async def fake_extract_media(body):
            if body.get("msgtype") == "image":
                return (["/tmp/x.png"], ["image/png"])
            return ([], [])

        adapter._extract_media = fake_extract_media

        await adapter._on_message(self._image_payload("img-1", None))
        # Image must be held, not dispatched.
        adapter.handle_message.assert_not_called()

        # Text arrives within the merge window.
        await asyncio.sleep(0.05)
        await adapter._on_message(self._text_payload("txt-1", "what is this?"))
        adapter.handle_message.assert_not_called()

        # After the window elapses, exactly one merged event dispatches.
        await asyncio.sleep(0.3)
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        from gateway.platforms.base import MessageType

        assert event.text == "what is this?"
        assert event.media_urls == ["/tmp/x.png"]
        assert event.media_types == ["image/png"]
        assert event.message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_image_only_dispatched_after_window(self):
        """image-only with no following text → still dispatched on its own
        after the merge window (must not be dropped)."""
        adapter = self._make_adapter(merge_delay=0.15)
        adapter._extract_media = AsyncMock(return_value=(["/tmp/x.png"], ["image/png"]))

        await adapter._on_message(self._image_payload("img-1", None))
        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.3)
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        from gateway.platforms.base import MessageType

        assert event.media_urls == ["/tmp/x.png"]
        assert event.message_type == MessageType.PHOTO

    @pytest.mark.asyncio
    async def test_multiple_attachments_then_text_all_merged(self):
        """Two attachment-only frames then text → all media merged into one
        dispatched event with the text."""
        adapter = self._make_adapter(merge_delay=0.2)

        counter = {"n": 0}

        async def fake_extract_media(body):
            if body.get("msgtype") == "image":
                counter["n"] += 1
                n = counter["n"]
                return ([f"/tmp/x{n}.png"], ["image/png"])
            return ([], [])

        adapter._extract_media = fake_extract_media

        await adapter._on_message(self._image_payload("img-1", None))
        await asyncio.sleep(0.03)
        await adapter._on_message(self._image_payload("img-2", None))
        await asyncio.sleep(0.03)
        await adapter._on_message(self._text_payload("txt-1", "describe both"))
        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.35)
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "describe both"
        assert event.media_urls == ["/tmp/x1.png", "/tmp/x2.png"]
        assert event.media_types == ["image/png", "image/png"]

    @pytest.mark.asyncio
    async def test_pure_text_unaffected(self):
        """Regression: pure text still flows through the text-batch path and
        dispatches as a single text event."""
        adapter = self._make_adapter()
        adapter._extract_media = AsyncMock(return_value=([], []))

        await adapter._on_message(self._text_payload("txt-1", "just text"))
        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.2)
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        from gateway.platforms.base import MessageType

        assert event.text == "just text"
        assert event.media_urls == []
        assert event.message_type == MessageType.TEXT



# === NATIVE STREAMING (msgtype: stream) ===


class TestWeComNativeStreamingCapability:
    """SUPPORTS_NATIVE_STREAMING + supports_native_streaming() probe."""

    def test_class_attribute_declares_native_streaming(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        assert WeComAdapter.SUPPORTS_NATIVE_STREAMING is True

    def test_supports_native_streaming_returns_true_for_dm(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        assert adapter.supports_native_streaming(chat_type="dm") is True

    def test_supports_native_streaming_returns_true_for_group(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        assert adapter.supports_native_streaming(chat_type="group") is True

    def test_max_stream_content_length_is_20480(self):
        from plugins.platforms.wecom.adapter import (
            MAX_STREAM_CONTENT_LENGTH, WeComAdapter,
        )

        assert MAX_STREAM_CONTENT_LENGTH == 20480
        assert WeComAdapter.MAX_STREAM_CONTENT_LENGTH == 20480

    def test_stream_expired_errcode_constant(self):
        from plugins.platforms.wecom.streaming import STREAM_EXPIRED_ERRCODE

        assert STREAM_EXPIRED_ERRCODE == 846608

# === STREAM TESTS PLACEHOLDER ===


class TestResolveStreamReqId:
    """`_resolve_stream_req_id` precedence: reply_to → cached chat → None."""

    def test_prefers_explicit_reply_to(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._reply_req_ids["msg-123"] = "explicit-req"
        adapter._last_chat_req_ids["chat-1"] = "cached-req"

        assert adapter._resolve_stream_req_id("chat-1", "msg-123") == "explicit-req"

    def test_falls_back_to_cached_chat_req_id(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "cached-req"

        assert adapter._resolve_stream_req_id("chat-1", reply_to=None) == "cached-req"

    def test_returns_none_when_no_anchor(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        assert adapter._resolve_stream_req_id("unknown-chat", None) is None

    def test_quoted_reply_to_falls_through_to_chat_cache(self):
        """``quote:msg-id`` (quote-context marker) is not a real reply anchor."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "cached-req"

        assert adapter._resolve_stream_req_id("chat-1", "quote:m-1") == "cached-req"


# === LIFECYCLE TESTS PLACEHOLDER ===


class TestSendStreamFrame:
    """`send_stream_frame` lifecycle: init → cumulative updates → finalize."""

    @staticmethod
    def _mock_send_json_with_immediate_ack(adapter):
        """Mock _send_reply_queued to bypass ack tracking entirely.

        For tests that verify frame content/ordering, we don't need actual
        ack tracking — just record what was sent and always succeed.
        """
        sent_frames = []

        async def mock_send_reply_queued(reply_req_id, body, *, is_final=False, skip_if_pending=False):
            sent_frames.append({
                "req_id": reply_req_id,
                "body": body,
                "is_final": is_final,
            })
            return {"errcode": 0, "errmsg": "ok"}

        adapter._send_reply_queued = AsyncMock(side_effect=mock_send_reply_queued)
        adapter._sent_frames = sent_frames

    @pytest.mark.asyncio
    async def test_first_call_seeds_thinking_frame_then_returns_true(self):
        """First frame for a chat sends <think></think> seed, then the
        content frame.

        Fire-and-forget: intermediate frames are pushed immediately (pure
        identity-dedup), so any non-empty payload produces a content frame
        right after the seed — no min_chars / sentence-boundary gating.
        """
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._ws = MagicMock(closed=False)
        # Mock _send_reply_queued to bypass ack tracking
        self._mock_send_json_with_immediate_ack(adapter)

        payload = "hello world"
        ok = await adapter.send_stream_frame(payload, chat_id="chat-1")

        assert ok is True
        # seed + content = 2 frames
        assert len(adapter._sent_frames) == 2
        seed_frame = adapter._sent_frames[0]
        assert seed_frame["body"]["msgtype"] == "stream"
        assert seed_frame["body"]["stream"]["content"] == "<think></think>"
        assert seed_frame["body"]["stream"]["finish"] is False

        content_frame = adapter._sent_frames[1]
        assert content_frame["body"]["stream"]["content"] == payload
        assert content_frame["body"]["stream"]["finish"] is False

    @pytest.mark.asyncio
    async def test_first_and_second_call_share_stream_id(self):
        """Successive frames use the same stream_id.

        Fire-and-forget pushes each distinct cumulative payload immediately,
        so this exercises stream_id continuity across frames, not chunker
        thresholds.
        """
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._ws = MagicMock(closed=False)
        # Immediate ack so all frames are sent (no pending-skip)
        self._mock_send_json_with_immediate_ack(adapter)

        first = "alpha"
        second = "alpha beta"  # cumulative growth — differs from `first`
        await adapter.send_stream_frame(first, chat_id="chat-1")
        await adapter.send_stream_frame(second, chat_id="chat-1")

        # seed + first + second = 3 frames
        assert len(adapter._sent_frames) == 3
        ids = [frame["body"]["stream"]["id"] for frame in adapter._sent_frames]
        assert ids[0] == ids[1] == ids[2]
        assert ids[0].startswith("stream_")

    @pytest.mark.asyncio
    async def test_intermediate_frame_skipped_when_pending_ack(self):
        """Intermediate frames are skipped if a prior frame's ack is pending.

        This is the new ack-tracking semantics: if the seed frame's ack hasn't
        returned yet, the next intermediate frame is skipped (returns success
        but doesn't actually send). This prevents errcode 6000 version conflict.
        """
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_json = AsyncMock()  # No auto-ack — pending stays pending
        adapter._ws = MagicMock(closed=False)

        await adapter.send_stream_frame("alpha", chat_id="chat-1")
        # Seed frame sent, pending_ack is set. Immediately send another:
        ok = await adapter.send_stream_frame("alpha beta", chat_id="chat-1")

        assert ok is True  # returns True (skip is silent success)
        # Only seed frame sent; second was skipped due to pending ack.
        assert adapter._send_json.await_count == 1

        # accumulated_text still updated in StreamTurn despite skip.
        turn = list(adapter._stream_turns.values())[0]
        assert turn.accumulated_text == "alpha beta"

    @pytest.mark.asyncio
    async def test_intermediate_frame_cap_drops_excess(self):
        """After MAX_INTERMEDIATE_FRAMES, further intermediate frames are dropped."""
        from plugins.platforms.wecom.adapter import WeComAdapter
        from plugins.platforms.wecom.streaming import MAX_INTERMEDIATE_FRAMES

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._ws = MagicMock(closed=False)
        # Auto-ack so seed + first frame go through
        self._mock_send_json_with_immediate_ack(adapter)

        # First call creates turn + seed + content frame.
        turn_id = "cap-test"
        await adapter.send_stream_frame("first", chat_id="chat-1", turn_id=turn_id)
        turn_key = f"chat-1:{turn_id}"
        turn = adapter._stream_turns[turn_key]

        # Artificially set counter to the cap.
        turn._intermediate_frames_sent = MAX_INTERMEDIATE_FRAMES
        turn._last_frame_sent_at = 0  # clear time throttle

        # Record count BEFORE the overflow frame to assert it was truly skipped.
        before_overflow = len(adapter._sent_frames)

        # Next intermediate frame should be dropped.
        ok = await adapter.send_stream_frame("overflow", chat_id="chat-1", turn_id=turn_id)
        assert ok is True
        assert turn.accumulated_text == "overflow"
        # No additional frame sent — overflow was dropped.
        assert len(adapter._sent_frames) == before_overflow

        # Finalize still goes through unconditionally.
        ok = await adapter.send_stream_frame("final", chat_id="chat-1", finalize=True, turn_id=turn_id)
        assert ok is True

    @pytest.mark.asyncio
    async def test_finalize_sends_finish_true_and_resets_state(self):
        """Finalize frame waits for pending ack, sends finish=true, cleans up turn."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._ws = MagicMock(closed=False)
        # Auto-ack so seed + content + finalize all go through
        self._mock_send_json_with_immediate_ack(adapter)

        # With turn_id, creates independent turn
        turn_id = "test-turn-1"
        await adapter.send_stream_frame("partial", chat_id="chat-1", turn_id=turn_id)
        turn_key = "chat-1:test-turn-1"
        assert turn_key in adapter._stream_turns
        turn = adapter._stream_turns[turn_key]
        assert turn.stream_id is not None

        ok = await adapter.send_stream_frame(
            "partial final", chat_id="chat-1", finalize=True, turn_id=turn_id,
        )

        assert ok is True
        # After finalize, turn should be cleaned up
        assert turn_key not in adapter._stream_turns
        # Finalize goes through _send_reply_queued (mocked).
        # Find the finalize frame (is_final=True)
        finalize_frames = [
            f for f in adapter._sent_frames
            if f["body"].get("stream", {}).get("finish") is True
        ]
        assert len(finalize_frames) == 1
        assert finalize_frames[0]["body"]["stream"]["content"] == "partial final"

# === FAILURE TESTS PLACEHOLDER ===


class TestSendStreamFrameFailures:
    """Behavior when no req_id, 846608 expiry, or generic transport errors."""

    @pytest.mark.asyncio
    async def test_returns_false_when_no_req_id_available(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        # No reply_to, nothing in _last_chat_req_ids.
        adapter._send_reply_request = AsyncMock()

        ok = await adapter.send_stream_frame("hi", chat_id="unknown-chat")

        assert ok is False
        adapter._send_reply_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_false_when_chat_id_missing_on_first_call(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._send_reply_request = AsyncMock()

        ok = await adapter.send_stream_frame("hi", chat_id=None)

        assert ok is False
        adapter._send_reply_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_846608_marks_chat_expired_and_returns_false(self):
        """846608 on finalize frame marks the chat expired and returns False."""
        from plugins.platforms.wecom.adapter import WeComAdapter
        from plugins.platforms.wecom.streaming import STREAM_EXPIRED_ERRCODE

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._ws = MagicMock(closed=False)

        # Mock _send_reply_queued: intermediate succeeds, final returns 846608
        async def mock_queued(reply_req_id, body, *, is_final=False, skip_if_pending=False):
            if is_final:
                return {"errcode": STREAM_EXPIRED_ERRCODE, "errmsg": "stream expired"}
            return {"errcode": 0, "errmsg": "ok"}

        adapter._send_reply_queued = AsyncMock(side_effect=mock_queued)

        # First call (seed + content) succeeds
        turn_id = "test-turn-2"
        await adapter.send_stream_frame("hello", chat_id="chat-1", turn_id=turn_id)
        # Now try to finalize — ack returns 846608.
        ok = await adapter.send_stream_frame("hello final", chat_id="chat-1", finalize=True, turn_id=turn_id)

        assert ok is False
        assert "chat-1" in adapter._stream_expired_chats
        # This specific turn should be cleaned up
        turn_key = "chat-1:test-turn-2"
        assert turn_key not in adapter._stream_turns

    @pytest.mark.asyncio
    async def test_subsequent_call_to_expired_chat_short_circuits(self):
        """Once a chat is in ``_stream_expired_chats``, send_stream_frame
        bails immediately for new turns without touching the WS."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._stream_expired_chats.add("chat-1")
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_reply_request = AsyncMock()

        # Without turn_id: short-circuits immediately
        ok = await adapter.send_stream_frame("hi", chat_id="chat-1")
        assert ok is False
        adapter._send_reply_request.assert_not_awaited()

        # With a new turn_id: also short-circuits (can't create new turn)
        ok = await adapter.send_stream_frame("hi", chat_id="chat-1", turn_id="new-turn")
        assert ok is False
        adapter._send_reply_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inbound_message_clears_expired_marker(self):
        """A fresh inbound req_id must resurrect the stream channel."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._stream_expired_chats.add("chat-1")

        adapter._remember_chat_req_id("chat-1", "fresh-req-id")

        assert "chat-1" not in adapter._stream_expired_chats

    @pytest.mark.asyncio
    async def test_generic_transport_error_on_intermediate_is_fire_and_forget(self):
        """A generic transport error on an INTERMEDIATE frame is fire-and-forget.

        The seed frame here fails with a generic RuntimeError.  An intermediate
        frame failing is transient and self-healing — a later cumulative frame
        (or the finalize frame) re-carries the full text — so the turn must stay
        live (keep-alive keeps refreshing it) and the call returns True.  It
        must NOT retire the turn or trip the consumer's send() fallback, which
        would re-deliver content the stream will overwrite (duplicate bubble).

        Contrast with the finalize-frame failure paths, which still return False
        and retire so the consumer can fall back (see the double_send /
        stream_dup_fix suites).
        """
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_reply_request = AsyncMock(
            side_effect=RuntimeError("ws disconnected"),
        )

        turn_id = "test-turn-3"
        ok = await adapter.send_stream_frame("hi", chat_id="chat-1", turn_id=turn_id)

        assert ok is True
        # Intermediate failure keeps the turn alive and leaves the chat usable.
        turn_key = "chat-1:test-turn-3"
        assert turn_key in adapter._stream_turns
        assert "chat-1" not in adapter._stream_expired_chats

# === SEND_TYPING TESTS PLACEHOLDER ===


class TestSendTypingTriggersThinking:
    """``send_typing`` is a no-op — typing is handled by stream consumer."""

    @pytest.mark.asyncio
    async def test_send_typing_is_noop(self):
        """send_typing must not open any stream — the consumer seed frame does."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_json = AsyncMock()
        adapter._ws = MagicMock(closed=False)

        await adapter.send_typing("chat-1")

        adapter._send_json.assert_not_awaited()
        # No stream turns created
        assert len(adapter._stream_turns) == 0

    @pytest.mark.asyncio
    async def test_send_typing_does_not_raise(self):
        """send_typing must never raise regardless of state."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        await adapter.send_typing("chat-1")
        await adapter.send_typing("")
        await adapter.send_typing(None)  # type: ignore


class TestStreamContentTruncation:
    """Bytes (not codepoints) are truncated to MAX_STREAM_CONTENT_LENGTH."""

    def test_ascii_below_limit_passes_through(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        out = WeComAdapter._truncate_stream_content("hello", 1000)
        assert out == "hello"

    def test_ascii_above_limit_is_byte_capped(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        big = "x" * 30000
        out = WeComAdapter._truncate_stream_content(big, 20480)
        assert len(out.encode("utf-8")) <= 20480

    def test_multibyte_truncation_does_not_split_codepoints(self):
        """A 3-byte CJK char must not be sliced mid-byte and emit garbage."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        # Each "你" is 3 UTF-8 bytes.  Limit at 5 bytes — must keep one
        # full char and drop the half-cut second char rather than emit ï¿½.
        out = WeComAdapter._truncate_stream_content("你你", 5)
        assert out == "你"
        # Crucially, must be valid UTF-8 (no replacement chars from
        # mid-byte slices).
        assert "�" not in out


@pytest.mark.skip(reason="Obsolete: send() no longer closes streams in new per-turn architecture")
class TestSendClosesActiveStream:
    """OBSOLETE: These tests verify old behavior where send() closed active streams.

    In the new per-turn architecture (post Round 3 fixes), send() and streaming
    are completely independent. Streams are managed by their creators
    (GatewayStreamConsumer) via send_stream_frame(finalize=True, turn_id=...).

    See test_wecom_per_turn.py for tests of the new per-turn model.
    """

    @pytest.mark.asyncio
    async def test_send_finalizes_active_stream_opened_by_consumer(self):
        """When the stream consumer opened a stream and then send() delivers
        the response (e.g. fallback path), send() must close the stream."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._send_json = AsyncMock()
        adapter._ws = MagicMock(closed=False)
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

        # Manually set active stream state (as the consumer would).
        adapter._active_stream_id = "stream_test"
        adapter._active_stream_req_id = "req-1"
        adapter._active_stream_chat_id = "chat-1"

        result = await adapter.send("chat-1", "Hello world!")

        assert result.success is True
        assert adapter._active_stream_id is None
        finalize_calls = [
            call for call in adapter._send_reply_request.await_args_list
            if call.args[1].get("stream", {}).get("finish") is True
        ]
        assert len(finalize_calls) == 1
        assert finalize_calls[0].args[1]["stream"]["content"] == "Hello world!"

    @pytest.mark.asyncio
    async def test_send_ignores_stream_for_different_chat(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-2"] = "req-2"
        adapter._send_reply_request = AsyncMock(return_value={"errcode": 0})

        adapter._active_stream_id = "stream_test"
        adapter._active_stream_req_id = "req-1"
        adapter._active_stream_chat_id = "chat-1"

        result = await adapter.send("chat-2", "Hi")

        assert result.success is True
        assert adapter._active_stream_id is not None  # untouched

    @pytest.mark.asyncio
    async def test_send_falls_through_when_stream_expired(self):
        from plugins.platforms.wecom.adapter import WeComAdapter
        from plugins.platforms.wecom.streaming import STREAM_EXPIRED_ERRCODE

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"

        async def fake(req_id, body, **kwargs):
            if body.get("stream", {}).get("finish"):
                return {"errcode": STREAM_EXPIRED_ERRCODE, "errmsg": "expired"}
            return {"errcode": 0, "headers": {"req_id": req_id}}

        adapter._send_reply_request = AsyncMock(side_effect=fake)
        adapter._active_stream_id = "stream_test"
        adapter._active_stream_req_id = "req-1"
        adapter._active_stream_chat_id = "chat-1"

        result = await adapter.send("chat-1", "Final answer")

        assert result.success is True
        assert adapter._active_stream_id is None
        assert "chat-1" in adapter._stream_expired_chats



class TestFireAndForgetFrameFlow:
    """Integration: send_stream_frame pushes each distinct cumulative payload
    immediately (pure identity-dedup), with no sentence/min-chars buffering."""

    def _mock_send_json_with_immediate_ack(self, adapter):
        sent_frames = []

        async def mock_send(reply_req_id, body, **kwargs):
            is_final = kwargs.get("is_final", False)
            sent_frames.append({
                "req_id": reply_req_id,
                "body": body,
                "is_final": is_final,
            })
            return {"errcode": 0, "errmsg": "ok"}

        adapter._send_reply_queued = AsyncMock(side_effect=mock_send)
        adapter._sent_frames = sent_frames

    @pytest.mark.asyncio
    async def test_short_text_sent_immediately(self):
        """Fire-and-forget: even a short body ships right after the seed —
        there is no min_chars buffering anymore."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._ws = MagicMock(closed=False)
        self._mock_send_json_with_immediate_ack(adapter)

        ok = await adapter.send_stream_frame("Hello.", chat_id="chat-1")
        assert ok is True
        # seed + content = 2 frames (the 6-char body is NOT buffered).
        assert len(adapter._sent_frames) == 2
        assert adapter._sent_frames[0]["body"]["stream"]["content"] == "<think></think>"
        assert adapter._sent_frames[1]["body"]["stream"]["content"] == "Hello."
        assert adapter._sent_frames[1]["body"]["stream"]["finish"] is False

    @pytest.mark.asyncio
    async def test_finalize_sends_accumulated_tail(self):
        """Finalize emits the accumulated text with finish=true.

        With no chunker, finalize uses the caller's cumulative text directly.
        """
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._ws = MagicMock(closed=False)
        self._mock_send_json_with_immediate_ack(adapter)

        # 1: intermediate frame (seed + content).
        await adapter.send_stream_frame("Short.", chat_id="chat-1")
        # 2: finalize with the same text. Content equals last_sent_content, so
        # the adapter appends a zero-width space to force a distinct final frame.
        ok = await adapter.send_stream_frame(
            "Short.", chat_id="chat-1", finalize=True,
        )
        assert ok is True
        # seed + content + finalize = 3 frames.
        assert len(adapter._sent_frames) == 3
        final_frame = adapter._sent_frames[-1]
        assert final_frame["body"]["stream"]["finish"] is True
        # Content survives the finalize (zero-width space appended when it
        # matched the previous frame verbatim).
        assert final_frame["body"]["stream"]["content"].startswith("Short.")

    @pytest.mark.asyncio
    async def test_duplicate_intermediate_content_is_deduped(self):
        """Identical cumulative content skips the send (pure identity-dedup),
        but still returns success."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._last_chat_req_ids["chat-1"] = "req-1"
        adapter._ws = MagicMock(closed=False)
        self._mock_send_json_with_immediate_ack(adapter)

        await adapter.send_stream_frame("same text", chat_id="chat-1")
        ok = await adapter.send_stream_frame("same text", chat_id="chat-1")
        assert ok is True
        # seed + first content only; the identical repeat was deduped.
        assert len(adapter._sent_frames) == 2
        assert adapter._sent_frames[-1]["body"]["stream"]["content"] == "same text"


class TestFinalFrameAckTimeoutSemantics:
    """Regression: final-frame ack timeout must not raise / trigger fallback.

    See docs/rca-wecom-stream-final-ack-timeout-duplicate.md — when WeCom's
    ack returns past the 5s window but the frame *was* delivered, raising
    causes the upper layer to fall back to a normal markdown send and the
    user sees the same content twice.  The fix: treat ack timeout as
    success-with-uncertainty and let the caller mark the turn delivered.
    """

    @pytest.mark.asyncio
    async def test_final_frame_ack_timeout_returns_success(self):
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._ws = MagicMock(closed=False)
        adapter._REPLY_ACK_TIMEOUT = 0.05  # snappy for the test
        # _send_json succeeds but no ack ever arrives.
        adapter._send_json = AsyncMock()

        response = await adapter._send_reply_queued(
            "req-1",
            {"msgtype": "stream", "stream": {"id": "stream_x", "content": "final", "finish": True}},
            is_final=True,
        )

        # Aligned-with-official semantics: success-shaped response with the
        # ack_pending flag set so callers can log / observe but no exception.
        assert response.get("errcode") == 0
        assert response.get("ack_pending") is True
        assert "ack_timeout" in response.get("errmsg", "")

    @pytest.mark.asyncio
    async def test_final_frame_send_failure_still_raises(self):
        """Genuine send failures (network/serialization) must still propagate.

        The ack-timeout relaxation only covers the case where the bytes went
        out but the ack didn't return. If ``_send_json`` itself raises, the
        upstream caller still needs to see the error.
        """
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        adapter._ws = MagicMock(closed=False)
        adapter._send_json = AsyncMock(side_effect=RuntimeError("ws closed"))

        with pytest.raises(RuntimeError, match="ws closed"):
            await adapter._send_reply_queued(
                "req-1",
                {"msgtype": "stream", "stream": {"id": "stream_x", "content": "x", "finish": True}},
                is_final=True,
            )
