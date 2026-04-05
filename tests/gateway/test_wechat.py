import asyncio
import time
from pathlib import Path

import pytest
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig


class TestWeChatPlatformEnum:
    def test_wechat_enum_exists(self):
        assert Platform.WECHAT.value == "wechat"


class TestWeChatConfigLoading:
    def test_apply_env_overrides_wechat(self, monkeypatch):
        monkeypatch.setenv("WECHAT_ENABLED", "true")
        monkeypatch.setenv("WECHAT_API_BASE_URL", "https://wx.example.com")
        monkeypatch.setenv("WECHAT_CDN_BASE_URL", "https://cdn.example.com/c2c")
        monkeypatch.setenv("WECHAT_ACCOUNT_ID", "bot-account-1")
        monkeypatch.setenv("WECHAT_HOME_CHANNEL", "user@im.wechat")
        monkeypatch.setenv("WECHAT_HOME_CHANNEL_NAME", "WeChat Home")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.WECHAT in config.platforms
        wc = config.platforms[Platform.WECHAT]
        assert wc.enabled is True
        assert wc.extra["base_url"] == "https://wx.example.com"
        assert wc.extra["cdn_base_url"] == "https://cdn.example.com/c2c"
        assert wc.extra["account_id"] == "bot-account-1"
        assert wc.home_channel is not None
        assert wc.home_channel.chat_id == "user@im.wechat"
        assert wc.home_channel.name == "WeChat Home"

    def test_connected_platforms_includes_wechat(self):
        from gateway.config import GatewayConfig

        config = GatewayConfig()
        config.platforms[Platform.WECHAT] = PlatformConfig(enabled=True)

        assert Platform.WECHAT in config.get_connected_platforms()


class TestWeChatStateStore:
    def test_state_store_round_trip(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat_state import WeChatAccount, WeChatStateStore

        store = WeChatStateStore()
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="user-1",
            enabled=True,
        )
        store.save_account(account)
        store.save_sync_cursor("acct-1", "cursor-123")
        store.set_context_token("acct-1", "peer@im.wechat", "ctx-123")

        loaded = store.load_account("acct-1")
        assert loaded is not None
        assert loaded.account_id == "acct-1"
        assert loaded.token == "secret-token"
        assert store.list_account_ids() == ["acct-1"]
        assert store.load_sync_cursor("acct-1") == "cursor-123"
        assert store.get_context_token("acct-1", "peer@im.wechat") == "ctx-123"


class TestWeChatAdapterSend:
    @pytest.mark.asyncio
    async def test_send_uses_saved_context_token(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._transport.send_text = AsyncMock(return_value={"message_id": "msg-1"})
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._state.set_context_token("acct-1", "peer@im.wechat", "ctx-abc")
        adapter._running = True

        result = await adapter.send("peer@im.wechat", "hello world")

        assert result.success is True
        assert result.message_id == "msg-1"
        adapter._transport.send_text.assert_awaited_once_with(
            account=adapter._state.load_account("acct-1"),
            to_user_id="peer@im.wechat",
            text="hello world",
            context_token="ctx-abc",
        )

    @pytest.mark.asyncio
    async def test_send_image_uses_local_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        image_path = tmp_path / "sample.png"
        image_path.write_bytes(b"png")

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._transport.send_media_file = AsyncMock(return_value={"message_id": "img-1"})
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._state.set_context_token("acct-1", "peer@im.wechat", "ctx-abc")
        adapter._running = True

        result = await adapter.send_image("peer@im.wechat", str(image_path), caption="look")

        assert result.success is True
        adapter._transport.send_media_file.assert_awaited_once_with(
            account=adapter._state.load_account("acct-1"),
            to_user_id="peer@im.wechat",
            file_path=str(image_path),
            text="look",
            context_token="ctx-abc",
        )

    @pytest.mark.asyncio
    async def test_send_image_downloads_remote_url_before_upload(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._transport.send_media_file = AsyncMock(return_value={"message_id": "img-remote-1"})
        adapter._transport._raw_http_get = AsyncMock(return_value=b"remote-png")
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._state.set_context_token("acct-1", "peer@im.wechat", "ctx-abc")
        adapter._running = True

        result = await adapter.send_image("peer@im.wechat", "https://example.com/assets/photo.png", caption="look")

        assert result.success is True
        adapter._transport._raw_http_get.assert_awaited_once_with(url="https://example.com/assets/photo.png")
        kwargs = adapter._transport.send_media_file.await_args.kwargs
        assert kwargs["account"] == adapter._state.load_account("acct-1")
        assert kwargs["to_user_id"] == "peer@im.wechat"
        assert kwargs["text"] == "look"
        assert kwargs["context_token"] == "ctx-abc"
        sent_path = Path(kwargs["file_path"])
        assert sent_path.exists()
        assert sent_path.suffix == ".png"
        assert sent_path.read_bytes() == b"remote-png"

    @pytest.mark.asyncio
    async def test_send_document_uses_context_token_to_pick_account(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        doc_path = tmp_path / "sample.pdf"
        doc_path.write_bytes(b"pdf")

        adapter = WeChatAdapter(PlatformConfig(enabled=True))
        adapter._transport.send_media_file = AsyncMock(return_value={"message_id": "doc-ctx-1"})
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="token-1",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner-1@im.wechat",
                enabled=True,
            )
        )
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-2",
                token="token-2",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner-2@im.wechat",
                enabled=True,
            )
        )
        adapter._state.set_context_token("acct-2", "peer@im.wechat", "ctx-2")
        adapter._running = True

        result = await adapter.send_document("peer@im.wechat", str(doc_path), caption="doc")

        assert result.success is True
        adapter._transport.send_media_file.assert_awaited_once_with(
            account=adapter._state.load_account("acct-2"),
            to_user_id="peer@im.wechat",
            file_path=str(doc_path),
            text="doc",
            context_token="ctx-2",
        )

    @pytest.mark.asyncio
    async def test_send_document_uses_transport_media_route(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        doc_path = tmp_path / "sample.pdf"
        doc_path.write_bytes(b"pdf")

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._transport.send_media_file = AsyncMock(return_value={"message_id": "doc-1"})
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._state.set_context_token("acct-1", "peer@im.wechat", "ctx-abc")
        adapter._running = True

        result = await adapter.send_document("peer@im.wechat", str(doc_path), caption="doc")

        assert result.success is True
        adapter._transport.send_media_file.assert_awaited_once_with(
            account=adapter._state.load_account("acct-1"),
            to_user_id="peer@im.wechat",
            file_path=str(doc_path),
            text="doc",
            context_token="ctx-abc",
        )

    @pytest.mark.asyncio
    async def test_send_image_file_routes_to_media_transport(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        image_path = tmp_path / "sample.png"
        image_path.write_bytes(b"png")

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._transport.send_media_file = AsyncMock(return_value={"message_id": "img-file-1"})
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._state.set_context_token("acct-1", "peer@im.wechat", "ctx-abc")
        adapter._running = True

        result = await adapter.send_image_file("peer@im.wechat", str(image_path), caption="look")

        assert result.success is True
        adapter._transport.send_media_file.assert_awaited_once_with(
            account=adapter._state.load_account("acct-1"),
            to_user_id="peer@im.wechat",
            file_path=str(image_path),
            text="look",
            context_token="ctx-abc",
        )

    @pytest.mark.asyncio
    async def test_send_voice_routes_to_media_transport(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        audio_path = tmp_path / "sample.amr"
        audio_path.write_bytes(b"voice")

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._transport.send_media_file = AsyncMock(return_value={"message_id": "voice-1"})
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._state.set_context_token("acct-1", "peer@im.wechat", "ctx-abc")
        adapter._running = True

        result = await adapter.send_voice("peer@im.wechat", str(audio_path), caption="voice")

        assert result.success is True
        adapter._transport.send_media_file.assert_awaited_once_with(
            account=adapter._state.load_account("acct-1"),
            to_user_id="peer@im.wechat",
            file_path=str(audio_path),
            text="voice",
            context_token="ctx-abc",
        )

    @pytest.mark.asyncio
    async def test_send_video_routes_to_media_transport(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        video_path = tmp_path / "sample.mp4"
        video_path.write_bytes(b"video")

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._transport.send_media_file = AsyncMock(return_value={"message_id": "video-1"})
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._state.set_context_token("acct-1", "peer@im.wechat", "ctx-abc")
        adapter._running = True

        result = await adapter.send_video("peer@im.wechat", str(video_path), caption="clip")

        assert result.success is True
        adapter._transport.send_media_file.assert_awaited_once_with(
            account=adapter._state.load_account("acct-1"),
            to_user_id="peer@im.wechat",
            file_path=str(video_path),
            text="clip",
            context_token="ctx-abc",
        )

    @pytest.mark.asyncio
    async def test_send_typing_uses_cached_ticket(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://wx.example.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._typing_cache[("acct-1", "peer@im.wechat")] = ("ticket-1", time.time() + 60)
        adapter._transport.send_typing = AsyncMock(return_value={"ret": 0})
        adapter._running = True

        await adapter.send_typing("peer@im.wechat")

        adapter._transport.send_typing.assert_awaited_once_with(
            account=adapter._state.load_account("acct-1"),
            ilink_user_id="peer@im.wechat",
            typing_ticket="ticket-1",
            status=1,
        )

    @pytest.mark.asyncio
    async def test_stop_typing_uses_cached_ticket(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://wx.example.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._typing_cache[("acct-1", "peer@im.wechat")] = ("ticket-1", time.time() + 60)
        adapter._transport.send_typing = AsyncMock(return_value={"ret": 0})
        adapter._running = True

        await adapter.stop_typing("peer@im.wechat")

        adapter._transport.send_typing.assert_awaited_once_with(
            account=adapter._state.load_account("acct-1"),
            ilink_user_id="peer@im.wechat",
            typing_ticket="ticket-1",
            status=2,
        )


class TestWeChatLogin:
    @pytest.mark.asyncio
    async def test_start_login_returns_qr_payload(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True))
        adapter._transport.start_login = AsyncMock(
            return_value={
                "session_key": "sess-1",
                "qrcode_url": "https://qr.example.com",
                "message": "scan me",
            }
        )

        result = await adapter.start_login()

        assert result["session_key"] == "sess-1"
        assert result["qrcode_url"] == "https://qr.example.com"
        adapter._transport.start_login.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_finish_login_persists_account(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True))
        adapter._transport.wait_login = AsyncMock(
            return_value={
                "connected": True,
                "account_id": "acct-9",
                "bot_token": "bot-token-9",
                "base_url": "https://wx.example.com",
                "user_id": "owner@im.wechat",
                "message": "ok",
            }
        )

        result = await adapter.wait_login("sess-9")

        assert result["connected"] is True
        loaded = adapter._state.load_account("acct-9")
        assert loaded is not None
        assert loaded.token == "bot-token-9"
        assert loaded.base_url == "https://wx.example.com"
        assert loaded.user_id == "owner@im.wechat"

    @pytest.mark.asyncio
    async def test_wait_login_starts_poll_task_when_running(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True))
        adapter._transport.wait_login = AsyncMock(
            return_value={
                "connected": True,
                "account_id": "acct-2",
                "bot_token": "secret-token-2",
                "base_url": "https://wx.example.com",
                "user_id": "owner2@im.wechat",
            }
        )
        adapter._poll_account_loop = AsyncMock()
        adapter._running = True

        await adapter.wait_login("sess-2")

        assert "acct-2" in adapter._poll_tasks

    @pytest.mark.asyncio
    async def test_login_then_send_image_file_forms_minimal_usable_flow(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter

        image_path = tmp_path / "sample.png"
        image_path.write_bytes(b"png")

        adapter = WeChatAdapter(PlatformConfig(enabled=True))
        adapter._transport.wait_login = AsyncMock(
            return_value={
                "connected": True,
                "account_id": "acct-3",
                "bot_token": "secret-token-3",
                "base_url": "https://wx.example.com",
                "user_id": "owner3@im.wechat",
            }
        )
        adapter._transport.send_media_file = AsyncMock(return_value={"message_id": "img-3"})
        adapter._poll_account_loop = AsyncMock()
        adapter._running = True

        await adapter.wait_login("sess-3")
        adapter._state.set_context_token("acct-3", "peer@im.wechat", "ctx-3")
        result = await adapter.send_image_file(
            "peer@im.wechat",
            str(image_path),
            caption="look",
            metadata={"account_id": "acct-3"},
        )

        assert result.success is True
        adapter._transport.send_media_file.assert_awaited_once()


class TestWeChatPolling:
    @pytest.mark.asyncio
    async def test_poll_once_converts_text_message_and_updates_state(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount
        from gateway.platforms.base import MessageType

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter.handle_message = AsyncMock()
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._transport.get_updates = AsyncMock(
            return_value={
                "msgs": [
                    {
                        "message_id": 101,
                        "from_user_id": "peer@im.wechat",
                        "create_time_ms": 1710000000000,
                        "context_token": "ctx-new",
                        "item_list": [
                            {"type": 1, "text_item": {"text": "hello from wechat"}}
                        ],
                    }
                ],
                "get_updates_buf": "cursor-next",
            }
        )

        await adapter._poll_account_once("acct-1")

        assert adapter._state.load_sync_cursor("acct-1") == "cursor-next"
        assert adapter._state.get_context_token("acct-1", "peer@im.wechat") == "ctx-new"
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "hello from wechat"
        assert event.message_type == MessageType.TEXT
        assert event.source.platform == Platform.WECHAT
        assert event.source.chat_id == "peer@im.wechat"
        assert event.source.user_id == "peer@im.wechat"
        assert event.source.user_id_alt == "acct-1"
        assert event.message_id == "101"


    @pytest.mark.asyncio
    async def test_poll_once_forwards_and_persists_longpoll_timeout_hint(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter.handle_message = AsyncMock()
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="***",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._transport.get_updates = AsyncMock(side_effect=[
            {"msgs": [], "get_updates_buf": "cursor-1", "longpolling_timeout_ms": 15000},
            {"msgs": [], "get_updates_buf": "cursor-2"},
        ])

        await adapter._poll_account_once("acct-1")
        await adapter._poll_account_once("acct-1")

        assert adapter._poll_longpoll_timeout_ms["acct-1"] == 15000
        first_kwargs = adapter._transport.get_updates.await_args_list[0].kwargs
        second_kwargs = adapter._transport.get_updates.await_args_list[1].kwargs
        assert first_kwargs["longpolling_timeout_ms"] is None
        assert second_kwargs["longpolling_timeout_ms"] == 15000

    @pytest.mark.asyncio
    async def test_connect_starts_poll_task_for_saved_account(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        adapter = WeChatAdapter(PlatformConfig(enabled=True))
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._transport.get_updates = AsyncMock(side_effect=asyncio.CancelledError())

        ok = await adapter.connect()
        assert ok is True
        assert "acct-1" in adapter._poll_tasks
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_closes_transport_session(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True))
        adapter._transport.close = AsyncMock()
        adapter._typing_cache[("acct-1", "peer@im.wechat")] = ("ticket", time.time() + 60)
        adapter._paused_until["acct-1"] = time.time() + 60
        adapter._mark_connected()

        await adapter.disconnect()

        adapter._transport.close.assert_awaited_once()
        assert adapter._typing_cache == {}
        assert adapter._paused_until == {}
        assert adapter._running is False

    @pytest.mark.asyncio
    async def test_poll_loop_pauses_after_repeated_failures(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount

        adapter = WeChatAdapter(PlatformConfig(enabled=True))
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._transport.get_updates = AsyncMock(side_effect=[RuntimeError("boom-1"), RuntimeError("boom-2"), asyncio.CancelledError()])
        adapter._sleep = AsyncMock(return_value=None)
        adapter._running = True

        with pytest.raises(asyncio.CancelledError):
            await adapter._poll_account_loop("acct-1")

        assert adapter._sleep.await_count >= 2

    @pytest.mark.asyncio
    async def test_poll_loop_pauses_on_session_expired(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount
        from gateway.platforms.wechat_transport import WeChatSessionExpiredError

        adapter = WeChatAdapter(PlatformConfig(enabled=True))
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._transport.get_updates = AsyncMock(side_effect=[WeChatSessionExpiredError("expired")])

        async def _fake_sleep(_seconds):
            adapter._running = False

        adapter._sleep = AsyncMock(side_effect=_fake_sleep)
        adapter._running = True

        await adapter._poll_account_loop("acct-1")

        assert adapter.is_account_paused("acct-1") is True
        adapter._sleep.assert_awaited()

    @pytest.mark.asyncio
    async def test_poll_once_downloads_inbound_image_to_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount
        from gateway.platforms.base import MessageType

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter.handle_message = AsyncMock()
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._transport.get_updates = AsyncMock(
            return_value={
                "msgs": [
                    {
                        "message_id": 102,
                        "from_user_id": "peer@im.wechat",
                        "create_time_ms": 1710000000000,
                        "context_token": "ctx-img",
                        "item_list": [
                            {"type": 2, "image_item": {"url": "https://cdn.example.com/sample.png"}}
                        ],
                    }
                ],
                "get_updates_buf": "cursor-img",
            }
        )
        adapter._download_media_to_cache = AsyncMock(return_value=str(tmp_path / "cached.png"))

        await adapter._poll_account_once("acct-1")

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.PHOTO
        assert event.media_urls == [str(tmp_path / "cached.png")]

    @pytest.mark.asyncio
    async def test_poll_once_downloads_inbound_file_to_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount
        from gateway.platforms.base import MessageType

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter.handle_message = AsyncMock()
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._transport.get_updates = AsyncMock(
            return_value={
                "msgs": [
                    {
                        "message_id": 103,
                        "from_user_id": "peer@im.wechat",
                        "create_time_ms": 1710000000000,
                        "context_token": "ctx-file",
                        "item_list": [
                            {"type": 4, "file_item": {"url": "https://cdn.example.com/sample.pdf", "file_name": "sample.pdf"}}
                        ],
                    }
                ],
                "get_updates_buf": "cursor-file",
            }
        )
        adapter._download_media_to_cache = AsyncMock(return_value=str(tmp_path / "cached.pdf"))

        await adapter._poll_account_once("acct-1")

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.DOCUMENT
        assert event.media_urls == [str(tmp_path / "cached.pdf")]

    @pytest.mark.asyncio
    async def test_poll_once_downloads_inbound_video_to_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount
        from gateway.platforms.base import MessageType

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter.handle_message = AsyncMock()
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._transport.get_updates = AsyncMock(
            return_value={
                "msgs": [
                    {
                        "message_id": 104,
                        "from_user_id": "peer@im.wechat",
                        "create_time_ms": 1710000000000,
                        "context_token": "ctx-video",
                        "item_list": [
                            {"type": 5, "video_item": {"url": "https://cdn.example.com/sample.mp4"}}
                        ],
                    }
                ],
                "get_updates_buf": "cursor-video",
            }
        )
        adapter._download_media_to_cache = AsyncMock(return_value=str(tmp_path / "cached.mp4"))

        await adapter._poll_account_once("acct-1")

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.VIDEO
        assert event.media_urls == [str(tmp_path / "cached.mp4")]

    @pytest.mark.asyncio
    async def test_poll_once_uses_voice_text_when_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount
        from gateway.platforms.base import MessageType

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter.handle_message = AsyncMock()
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._transport.get_updates = AsyncMock(
            return_value={
                "msgs": [
                    {
                        "message_id": 105,
                        "from_user_id": "peer@im.wechat",
                        "create_time_ms": 1710000000000,
                        "context_token": "ctx-voice",
                        "item_list": [
                            {"type": 3, "voice_item": {"text": "transcribed voice"}}
                        ],
                    }
                ],
                "get_updates_buf": "cursor-voice",
            }
        )

        await adapter._poll_account_once("acct-1")

        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.VOICE
        assert event.text == "transcribed voice"

    @pytest.mark.asyncio
    async def test_poll_once_downloads_voice_media_when_no_text(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.wechat_state import WeChatAccount
        from gateway.platforms.base import MessageType

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter.handle_message = AsyncMock()
        adapter._state.save_account(
            WeChatAccount(
                account_id="acct-1",
                token="secret-token",
                base_url="https://ilinkai.weixin.qq.com",
                user_id="owner@im.wechat",
                enabled=True,
            )
        )
        adapter._transport.get_updates = AsyncMock(
            return_value={
                "msgs": [
                    {
                        "message_id": 106,
                        "from_user_id": "peer@im.wechat",
                        "create_time_ms": 1710000000000,
                        "context_token": "ctx-voice2",
                        "item_list": [
                            {"type": 3, "voice_item": {"url": "https://cdn.example.com/sample.amr"}}
                        ],
                    }
                ],
                "get_updates_buf": "cursor-voice2",
            }
        )
        adapter._download_media_to_cache = AsyncMock(return_value=str(tmp_path / "cached.amr"))

        await adapter._poll_account_once("acct-1")

        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.VOICE
        assert event.media_urls == [str(tmp_path / "cached.amr")]

    @pytest.mark.asyncio
    async def test_build_message_event_prefers_text_over_nontext_items(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter
        from gateway.platforms.base import MessageType

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._download_media_to_cache = AsyncMock(return_value=str(tmp_path / "cached.bin"))

        event = await adapter._build_message_event(
            "acct-1",
            {
                "message_id": 107,
                "from_user_id": "peer@im.wechat",
                "create_time_ms": 1710000000000,
                "item_list": [
                    {"type": 4, "file_item": {"file_name": "sample.pdf", "url": "https://cdn.example.com/sample.pdf"}},
                    {"type": 1, "text_item": {"text": "body text"}},
                ],
            },
        )

        assert event.text == "body text"
        assert event.message_type == MessageType.DOCUMENT

    def test_resolve_item_download_url_prefers_media_struct(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        url = adapter._resolve_item_download_url(
            {
                "media": {"full_url": "https://cdn.example.com/full", "encrypt_query_param": "enc-q", "aes_key": "xyz"},
                "url": "https://cdn.example.com/fallback",
            }
        )

        assert url == "https://cdn.example.com/full"

    @pytest.mark.asyncio
    async def test_download_media_to_cache_prefers_media_full_url(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._transport._raw_http_get = AsyncMock(return_value=b"abc")

        cached = await adapter._download_media_to_cache(
            {
                "type": 4,
                "file_item": {
                    "file_name": "sample.pdf",
                    "media": {"full_url": "https://cdn.example.com/full", "encrypt_query_param": "enc-q", "aes_key": "xyz"},
                    "url": "https://cdn.example.com/fallback",
                },
            }
        )

        assert cached
        adapter._transport._raw_http_get.assert_awaited_once_with(url="https://cdn.example.com/full")

    @pytest.mark.asyncio
    async def test_download_image_to_cache_uses_decrypted_transport_bytes(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.wechat import WeChatAdapter

        adapter = WeChatAdapter(PlatformConfig(enabled=True, extra={"account_id": "acct-1"}))
        adapter._transport.fetch_media_bytes = AsyncMock(return_value=b"\x89PNG\r\n\x1a\nplain-image")

        cached = await adapter._download_media_to_cache(
            {
                "type": 2,
                "image_item": {
                    "media": {"full_url": "https://cdn.example.com/full.png", "aes_key": "xyz"},
                    "url": "https://cdn.example.com/fallback.jpg",
                },
            }
        )

        assert cached
        cached_path = Path(cached)
        assert cached_path.suffix == ".png"
        assert cached_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        adapter._transport.fetch_media_bytes.assert_awaited_once_with(
            {"full_url": "https://cdn.example.com/full.png", "aes_key": "xyz"}
        )


class TestWeChatTransport:
    def test_markdown_to_plain_text_matches_official_core_rules(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        text = "# Title\n[link](https://example.com)\n![img](https://x/y.png)\n```py\nprint(1)\n```"

        result = transport.markdown_to_plain_text(text)

        assert "link" in result
        assert "https://example.com" not in result
        assert "![img]" not in result
        assert "print(1)" in result

    def test_build_upload_request_shape(self, tmp_path):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport, _aes_ecb_padded_size

        p = tmp_path / "sample.png"
        p.write_bytes(b"hello")
        transport = OfficialWeChatTransport()

        request = transport._build_upload_request(file_path=str(p), to_user_id="peer", media_type=1)

        assert len(request["filekey"]) == 32
        assert request["rawsize"] == 5
        assert request["filesize"] == _aes_ecb_padded_size(5)
        assert len(request["rawfilemd5"]) == 32
        assert len(request["_aes_key_raw"]) == 16
        assert len(request["aeskey"]) == 32
        assert request["_plaintext"] == b"hello"
        assert request["no_need_thumb"] is True

    def test_extract_download_url_prefers_full_url_then_query(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()

        media = {"full_url": "https://cdn.example.com/full", "encrypt_query_param": "enc-q"}
        assert transport._extract_download_url(media) == "https://cdn.example.com/full"

        media2 = {"encrypt_query_param": "enc-q"}
        assert transport._extract_download_url(media2) == "https://novac2c.cdn.weixin.qq.com/c2c/download?encrypted_query_param=enc-q"

    def test_extract_download_url_uses_configured_cdn_base_for_encrypted_query(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport(cdn_base_url="https://cdn.example.com/custom")

        url = transport._extract_download_url({"encrypt_query_param": "enc-q"})

        assert url == "https://cdn.example.com/custom/download?encrypted_query_param=enc-q"

    @pytest.mark.asyncio
    async def test_fetch_media_bytes_uses_extracted_download_url(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        transport._raw_http_get = AsyncMock(return_value=b"payload")

        data = await transport.fetch_media_bytes({"full_url": "https://cdn.example.com/full"})

        assert data == b"payload"
        transport._raw_http_get.assert_awaited_once_with(url="https://cdn.example.com/full")

    @pytest.mark.asyncio
    async def test_fetch_media_bytes_passes_through_decrypt_hook(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        transport._raw_http_get = AsyncMock(return_value=b"encrypted")
        transport._maybe_decrypt_media = AsyncMock(return_value=b"plain")

        data = await transport.fetch_media_bytes({"full_url": "https://cdn.example.com/full", "aes_key": "abc"})

        assert data == b"plain"
        transport._maybe_decrypt_media.assert_awaited_once_with(b"encrypted", {"full_url": "https://cdn.example.com/full", "aes_key": "abc"})

    def test_build_headers_match_official_shape(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        headers = transport._build_headers(token="secret-token")

        assert headers["Content-Type"] == "application/json"
        assert headers["AuthorizationType"] == "ilink_bot_token"
        assert headers["Authorization"] == "Bearer secret-token"
        assert "X-WECHAT-UIN" in headers
        assert "iLink-App-Id" in headers
        assert "iLink-App-ClientVersion" in headers

    @pytest.mark.asyncio
    async def test_start_login_calls_qrcode_endpoint(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        transport._api_get = AsyncMock(
            return_value={
                "qrcode": "qr-token",
                "qrcode_img_content": "https://qr.example.com/img",
            }
        )

        result = await transport.start_login(account_id="acct-1")

        assert result["session_key"] == "acct-1"
        assert result["qrcode_url"] == "https://qr.example.com/img"
        transport._api_get.assert_awaited_once_with(
            "ilink/bot/get_bot_qrcode",
            params={"bot_type": "3"},
        )

    @pytest.mark.asyncio
    async def test_wait_login_calls_status_endpoint(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        transport._active_logins["sess-1"] = {"qrcode": "qr-token"}
        transport._api_get = AsyncMock(
            return_value={
                "status": "confirmed",
                "bot_token": "bot-token",
                "ilink_bot_id": "acct-9",
                "baseurl": "https://wx.example.com",
                "ilink_user_id": "owner@im.wechat",
            }
        )

        result = await transport.wait_login("sess-1", timeout_ms=1000)

        assert result["connected"] is True
        assert result["account_id"] == "acct-9"
        assert result["bot_token"] == "bot-token"
        transport._api_get.assert_awaited_once_with(
            "ilink/bot/get_qrcode_status",
            params={"qrcode": "qr-token"},
            base_url=transport.base_url,
        )

    @pytest.mark.asyncio
    async def test_wait_login_returns_wait_status_without_clearing_session(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        transport._active_logins["sess-1"] = {"qrcode": "qr-token"}
        transport._api_get = AsyncMock(return_value={"status": "wait"})

        result = await transport.wait_login("sess-1", timeout_ms=1000)

        assert result["connected"] is False
        assert result["message"] == "wait"
        assert "sess-1" in transport._active_logins

    @pytest.mark.asyncio
    async def test_wait_login_refreshes_qr_on_expired(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        transport._active_logins["sess-1"] = {"qrcode": "qr-old", "qrcode_url": "old-url"}
        transport._api_get = AsyncMock(side_effect=[
            {"status": "expired"},
            {"qrcode": "qr-new", "qrcode_img_content": "https://qr.example.com/new"},
        ])

        result = await transport.wait_login("sess-1", timeout_ms=1000)

        assert result["connected"] is False
        assert result["message"] == "expired"
        assert transport._active_logins["sess-1"]["qrcode"] == "qr-new"
        assert transport._active_logins["sess-1"]["qrcode_url"] == "https://qr.example.com/new"

    @pytest.mark.asyncio
    async def test_wait_login_switches_base_url_on_redirect(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        transport._active_logins["sess-1"] = {"qrcode": "qr-token"}
        transport._api_get = AsyncMock(return_value={"status": "scaned_but_redirect", "redirect_host": "redirect.wx.example.com"})

        result = await transport.wait_login("sess-1", timeout_ms=1000)

        assert result["connected"] is False
        assert result["message"] == "scaned_but_redirect"
        assert transport._active_logins["sess-1"]["base_url"] == "https://redirect.wx.example.com"

    @pytest.mark.asyncio
    async def test_start_login_raises_on_protocol_error(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        transport._api_get = AsyncMock(return_value={"ret": -1, "errmsg": "bad request"})

        with pytest.raises(RuntimeError, match="errcode=-1"):
            await transport.start_login()

    @pytest.mark.asyncio
    async def test_get_updates_posts_cursor_and_token(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"msgs": [], "get_updates_buf": "cursor-next"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        result = await transport.get_updates(account=account, cursor="cursor-prev")

        assert result["get_updates_buf"] == "cursor-next"
        transport._api_post.assert_awaited_once_with(
            "ilink/bot/getupdates",
            json_body={"get_updates_buf": "cursor-prev", "base_info": {"channel_version": transport._channel_version}},
            token="secret-token",
            base_url="https://wx.example.com",
        )

    @pytest.mark.asyncio
    async def test_get_updates_returns_empty_on_timeout(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(side_effect=TimeoutError())
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        result = await transport.get_updates(account=account, cursor="cursor-prev")

        assert result["msgs"] == []
        assert result["get_updates_buf"] == "cursor-prev"

    @pytest.mark.asyncio
    async def test_get_updates_raises_on_protocol_error(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"ret": -2, "errmsg": "poll failed"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        with pytest.raises(RuntimeError, match="poll failed"):
            await transport.get_updates(account=account, cursor="cursor-prev")

    @pytest.mark.asyncio
    async def test_get_updates_posts_longpoll_timeout_hint_when_provided(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"msgs": [], "get_updates_buf": "cursor-next"})
        account = WeChatAccount(
            account_id="acct-1",
            token="***",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        await transport.get_updates(account=account, cursor="cursor-prev", longpolling_timeout_ms=15000)

        transport._api_post.assert_awaited_once_with(
            "ilink/bot/getupdates",
            json_body={
                "get_updates_buf": "cursor-prev",
                "base_info": {"channel_version": transport._channel_version},
                "longpolling_timeout_ms": 15000,
            },
            token="***",
            base_url="https://wx.example.com",
        )

    @pytest.mark.asyncio
    async def test_get_config_uses_context_token(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"ret": 0, "typing_ticket": "ticket-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        result = await transport.get_config(account=account, ilink_user_id="peer@im.wechat", context_token="ctx-1")

        assert result["typing_ticket"] == "ticket-1"
        transport._api_post.assert_awaited_once_with(
            "ilink/bot/getconfig",
            json_body={"ilink_user_id": "peer@im.wechat", "context_token": "ctx-1", "base_info": {"channel_version": transport._channel_version}},
            token="secret-token",
            base_url="https://wx.example.com",
        )

    @pytest.mark.asyncio
    async def test_get_config_raises_on_protocol_error(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"errcode": 40001, "errmsg": "bad config"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        with pytest.raises(RuntimeError, match="bad config"):
            await transport.get_config(account=account, ilink_user_id="peer@im.wechat", context_token="ctx-1")

    @pytest.mark.asyncio
    async def test_send_typing_posts_status(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"ret": 0})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        await transport.send_typing(account=account, ilink_user_id="peer@im.wechat", typing_ticket="ticket-1", status=1)

        transport._api_post.assert_awaited_once_with(
            "ilink/bot/sendtyping",
            json_body={"ilink_user_id": "peer@im.wechat", "typing_ticket": "ticket-1", "status": 1, "base_info": {"channel_version": transport._channel_version}},
            token="secret-token",
            base_url="https://wx.example.com",
        )

    @pytest.mark.asyncio
    async def test_send_text_raises_on_protocol_error(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"ret": -5, "errmsg": "send failed"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        with pytest.raises(RuntimeError, match="send failed"):
            await transport.send_text(account=account, to_user_id="peer@im.wechat", text="hello", context_token="ctx-1")

    @pytest.mark.asyncio
    async def test_get_upload_url_includes_base_info(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"upload_param": "up-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        upload_request = {
            "filekey": "file-key",
            "media_type": 1,
            "to_user_id": "peer@im.wechat",
            "rawsize": 100,
            "rawfilemd5": "abc123",
            "filesize": 112,
            "no_need_thumb": True,
            "aeskey": "deadbeef" * 4,
            "_aes_key_raw": b"\x00" * 16,
            "_plaintext": b"x" * 100,
        }

        result = await transport.get_upload_url(account=account, upload_request=upload_request)

        assert result["upload_param"] == "up-1"
        call_kwargs = transport._api_post.await_args.kwargs
        body = call_kwargs["json_body"]
        assert body["filekey"] == "file-key"
        assert body["media_type"] == 1
        assert body["to_user_id"] == "peer@im.wechat"
        assert body["base_info"] == {"channel_version": transport._channel_version}
        assert "_aes_key_raw" not in body
        assert "_plaintext" not in body

    def test_build_upload_request_for_file_has_md5_and_size(self, tmp_path):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport, _aes_ecb_padded_size

        file_path = tmp_path / "sample.pdf"
        file_path.write_bytes(b"abcdef")

        transport = OfficialWeChatTransport()
        payload = transport._build_upload_request(file_path=str(file_path), to_user_id="peer@im.wechat", media_type=3)

        assert len(payload["filekey"]) == 32
        assert payload["to_user_id"] == "peer@im.wechat"
        assert payload["media_type"] == 3
        assert payload["rawsize"] == 6
        assert payload["filesize"] == _aes_ecb_padded_size(6)
        assert len(payload["rawfilemd5"]) == 32
        assert len(payload["_aes_key_raw"]) == 16
        assert payload["_plaintext"] == b"abcdef"

    @pytest.mark.asyncio
    async def test_upload_image_uses_get_upload_url_and_cdn_post(self, tmp_path):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        image_path = tmp_path / "sample.png"
        image_path.write_bytes(b"png-bytes")

        transport = OfficialWeChatTransport()
        transport.get_upload_url = AsyncMock(return_value={"upload_param": "token-1"})
        transport._cdn_upload = AsyncMock(return_value={"ok": True, "download_param": "dl-enc-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        uploaded = await transport._upload_image(account=account, to_user_id="peer@im.wechat", file_path=str(image_path))

        assert len(uploaded["filekey"]) == 32
        assert uploaded["downloadEncryptedQueryParam"] == "dl-enc-1"
        assert uploaded["fileSize"] == 9
        transport.get_upload_url.assert_awaited_once()
        transport._cdn_upload.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cdn_upload_encrypts_and_posts_with_retry(self, tmp_path):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport, _encrypt_aes_ecb
        import secrets

        aes_key = secrets.token_bytes(16)
        plaintext = b"png-bytes"

        transport = OfficialWeChatTransport()
        transport._raw_http_post = AsyncMock(return_value={"ok": True, "download_param": "dl-enc-1", "url": "https://cdn/upload", "size": 16})

        result = await transport._cdn_upload(
            upload_url="https://cdn.example.com/upload?param=abc",
            plaintext=plaintext,
            aes_key=aes_key,
            label="test",
        )

        assert result["download_param"] == "dl-enc-1"
        transport._raw_http_post.assert_awaited_once()
        kwargs = transport._raw_http_post.await_args.kwargs
        assert kwargs["headers"]["Content-Type"] == "application/octet-stream"
        assert kwargs["data"] == _encrypt_aes_ecb(plaintext, aes_key)
        assert kwargs["data"] != plaintext

    @pytest.mark.asyncio
    async def test_raw_http_get_reads_remote_file_bytes(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        transport._raw_http_get = AsyncMock(return_value=b"pdf-bytes")

        data = await transport._raw_http_get(url="https://cdn.example.com/sample.pdf")

        assert data == b"pdf-bytes"

    @pytest.mark.asyncio
    async def test_upload_file_uses_get_upload_url_and_cdn_post(self, tmp_path):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        doc_path = tmp_path / "sample.pdf"
        doc_path.write_bytes(b"pdf-bytes")

        transport = OfficialWeChatTransport()
        transport.get_upload_url = AsyncMock(return_value={"upload_param": "token-1"})
        transport._cdn_upload = AsyncMock(return_value={"ok": True, "download_param": "dl-enc-2"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        uploaded = await transport._upload_file(account=account, to_user_id="peer@im.wechat", file_path=str(doc_path))

        assert len(uploaded["filekey"]) == 32
        assert uploaded["downloadEncryptedQueryParam"] == "dl-enc-2"
        transport.get_upload_url.assert_awaited_once()
        transport._cdn_upload.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_media_file_routes_image_upload(self, tmp_path):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        image_path = tmp_path / "sample.png"
        image_path.write_bytes(b"png")

        transport = OfficialWeChatTransport()
        transport._upload_media = AsyncMock(return_value={"filekey": "img-key", "downloadEncryptedQueryParam": "enc-q", "aeskey": "abc" * 10, "fileSize": 3, "fileSizeCiphertext": 16})
        transport._send_image_message = AsyncMock(return_value={"message_id": "img-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        result = await transport.send_media_file(
            account=account,
            to_user_id="peer@im.wechat",
            file_path=str(image_path),
            text="look",
            context_token="ctx-1",
        )

        assert result["message_id"] == "img-1"
        transport._upload_media.assert_awaited_once()
        transport._send_image_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_image_message_posts_expected_payload(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"message_id": "img-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        await transport._send_image_message(
            account=account,
            to_user_id="peer@im.wechat",
            uploaded={"filekey": "img-key", "downloadEncryptedQueryParam": "enc-q", "aeskey": "abc", "fileSizeCiphertext": 123},
            text="look",
            context_token="ctx-1",
        )

        assert transport._api_post.await_count == 2
        first_payload = transport._api_post.await_args_list[0].kwargs["json_body"]
        second_payload = transport._api_post.await_args_list[1].kwargs["json_body"]
        assert first_payload["msg"]["to_user_id"] == "peer@im.wechat"
        assert first_payload["msg"]["context_token"] == "ctx-1"
        assert first_payload["msg"]["message_type"] == 2
        assert first_payload["msg"]["message_state"] == 2
        assert first_payload["msg"]["item_list"][0]["type"] == 1
        assert second_payload["msg"]["item_list"][0]["type"] == 2
        assert second_payload["msg"]["client_id"]
        assert second_payload["msg"]["item_list"][0]["image_item"]["media"]["encrypt_query_param"] == "enc-q"
        assert second_payload["msg"]["item_list"][0]["image_item"]["mid_size"] == 123

    @pytest.mark.asyncio
    async def test_send_media_file_routes_video_upload(self, tmp_path):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        video_path = tmp_path / "sample.mp4"
        video_path.write_bytes(b"video")

        transport = OfficialWeChatTransport()
        transport._upload_media = AsyncMock(return_value={"filekey": "vid-key", "downloadEncryptedQueryParam": "enc-v", "aeskey": "abc" * 10, "fileSize": 5, "fileSizeCiphertext": 16})
        transport._send_video_message = AsyncMock(return_value={"message_id": "vid-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        result = await transport.send_media_file(
            account=account,
            to_user_id="peer@im.wechat",
            file_path=str(video_path),
            text="clip",
            context_token="ctx-1",
        )

        assert result["message_id"] == "vid-1"
        transport._upload_media.assert_awaited_once()
        transport._send_video_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_media_file_routes_document_upload(self, tmp_path):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        doc_path = tmp_path / "sample.pdf"
        doc_path.write_bytes(b"pdf")

        transport = OfficialWeChatTransport()
        transport._upload_media = AsyncMock(return_value={"filekey": "doc-key", "downloadEncryptedQueryParam": "enc-f", "aeskey": "abc" * 10, "fileSize": 3, "fileSizeCiphertext": 16})
        transport._send_file_message = AsyncMock(return_value={"message_id": "doc-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        result = await transport.send_media_file(
            account=account,
            to_user_id="peer@im.wechat",
            file_path=str(doc_path),
            text="doc",
            context_token="ctx-1",
        )

        assert result["message_id"] == "doc-1"
        transport._upload_media.assert_awaited_once()
        transport._send_file_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_file_message_posts_expected_payload(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"message_id": "file-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        await transport._send_file_message(
            account=account,
            to_user_id="peer@im.wechat",
            uploaded={"filekey": "doc-key", "downloadEncryptedQueryParam": "enc-f", "aeskey": "abc", "fileSize": 456},
            file_path="/tmp/sample.pdf",
            text="doc",
            context_token="ctx-1",
        )

        assert transport._api_post.await_count == 2
        payload = transport._api_post.await_args_list[1].kwargs["json_body"]
        assert payload["msg"]["to_user_id"] == "peer@im.wechat"
        assert payload["msg"]["context_token"] == "ctx-1"
        assert payload["msg"]["item_list"][0]["type"] == 4
        assert payload["msg"]["message_type"] == 2
        assert payload["msg"]["message_state"] == 2
        assert payload["msg"]["item_list"][0]["file_item"]["media"]["encrypt_query_param"] == "enc-f"
        assert payload["msg"]["item_list"][0]["file_item"]["len"] == "456"

    @pytest.mark.asyncio
    async def test_send_video_message_posts_expected_payload(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"message_id": "video-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        await transport._send_video_message(
            account=account,
            to_user_id="peer@im.wechat",
            uploaded={"filekey": "vid-key", "downloadEncryptedQueryParam": "enc-v", "aeskey": "abc", "fileSizeCiphertext": 789},
            text="clip",
            context_token="ctx-1",
        )

        assert transport._api_post.await_count == 2
        payload = transport._api_post.await_args_list[1].kwargs["json_body"]
        assert payload["msg"]["item_list"][0]["type"] == 5
        assert payload["msg"]["message_type"] == 2
        assert payload["msg"]["message_state"] == 2
        assert payload["msg"]["item_list"][0]["video_item"]["media"]["encrypt_query_param"] == "enc-v"
        assert payload["msg"]["item_list"][0]["video_item"]["video_size"] == 789

    @pytest.mark.asyncio
    async def test_send_media_file_routes_voice_upload(self, tmp_path):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        voice_path = tmp_path / "sample.amr"
        voice_path.write_bytes(b"voice")

        transport = OfficialWeChatTransport()
        transport._upload_media = AsyncMock(return_value={"filekey": "voice-key", "downloadEncryptedQueryParam": "enc-a", "aeskey": "abc" * 10, "fileSize": 5, "fileSizeCiphertext": 16})
        transport._send_voice_message = AsyncMock(return_value={"message_id": "voice-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        result = await transport.send_media_file(
            account=account,
            to_user_id="peer@im.wechat",
            file_path=str(voice_path),
            text="voice",
            context_token="ctx-1",
        )

        assert result["message_id"] == "voice-1"
        transport._upload_media.assert_awaited_once()
        transport._send_voice_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_voice_message_posts_expected_payload(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"message_id": "voice-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        await transport._send_voice_message(
            account=account,
            to_user_id="peer@im.wechat",
            uploaded={"filekey": "voice-key", "downloadEncryptedQueryParam": "enc-a", "aeskey": "abc"},
            text="voice",
            context_token="ctx-1",
        )

        assert transport._api_post.await_count == 2
        payload = transport._api_post.await_args_list[1].kwargs["json_body"]
        assert payload["msg"]["item_list"][0]["type"] == 3
        assert payload["msg"]["item_list"][0]["voice_item"]["media"]["encrypt_query_param"] == "enc-a"

    @pytest.mark.asyncio
    async def test_send_text_posts_official_msg_envelope(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport
        from gateway.platforms.wechat_state import WeChatAccount

        transport = OfficialWeChatTransport()
        transport._api_post = AsyncMock(return_value={"message_id": "txt-1"})
        account = WeChatAccount(
            account_id="acct-1",
            token="secret-token",
            base_url="https://wx.example.com",
            user_id="owner@im.wechat",
            enabled=True,
        )

        await transport.send_text(account=account, to_user_id="peer@im.wechat", text="hello", context_token="ctx-1")

        payload = transport._api_post.await_args.kwargs["json_body"]
        assert payload["msg"]["to_user_id"] == "peer@im.wechat"
        assert payload["msg"]["context_token"] == "ctx-1"
        assert payload["msg"]["message_type"] == 2
        assert payload["msg"]["message_state"] == 2
        assert payload["msg"]["client_id"]
        assert payload["msg"]["item_list"][0]["type"] == 1

    @pytest.mark.asyncio
    async def test_aes_ecb_encrypt_decrypt_roundtrip(self):
        from gateway.platforms.wechat_transport import _encrypt_aes_ecb, _decrypt_aes_ecb
        import secrets

        key = secrets.token_bytes(16)
        plaintext = b"Hello WeChat CDN upload test data!"

        ciphertext = _encrypt_aes_ecb(plaintext, key)
        assert ciphertext != plaintext
        assert len(ciphertext) % 16 == 0

        decrypted = _decrypt_aes_ecb(ciphertext, key)
        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_maybe_decrypt_media_with_valid_key(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport, _encrypt_aes_ecb
        import base64, secrets

        key = secrets.token_bytes(16)
        plaintext = b"secret image data"
        ciphertext = _encrypt_aes_ecb(plaintext, key)
        aes_key_b64 = base64.b64encode(key).decode()

        transport = OfficialWeChatTransport()
        result = await transport._maybe_decrypt_media(ciphertext, {"aes_key": aes_key_b64})

        assert result == plaintext

    @pytest.mark.asyncio
    async def test_maybe_decrypt_media_returns_raw_without_key(self):
        from gateway.platforms.wechat_transport import OfficialWeChatTransport

        transport = OfficialWeChatTransport()
        raw = b"raw bytes"

        result = await transport._maybe_decrypt_media(raw, {})
        assert result == raw

        result2 = await transport._maybe_decrypt_media(raw, {"aes_key": ""})
        assert result2 == raw

    def test_aes_ecb_padded_size(self):
        from gateway.platforms.wechat_transport import _aes_ecb_padded_size
        import math

        # mirrors official: Math.ceil((size + 1) / 16) * 16
        assert _aes_ecb_padded_size(0) == 16
        assert _aes_ecb_padded_size(1) == 16
        assert _aes_ecb_padded_size(15) == 16
        assert _aes_ecb_padded_size(16) == 32
        assert _aes_ecb_padded_size(31) == 32
        assert _aes_ecb_padded_size(32) == 48
