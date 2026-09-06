"""Image transport failures retain useful context without exposing signed URLs."""
import logging
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult

SECRET = "opaqueImageCredential123"
URL = f"https://example.com/image.png?X-Amz-Signature={SECRET}&width=1024"
ERROR = f"download rejected: {URL}"


@pytest.fixture(autouse=True)
def forced_boundary(monkeypatch, caplog):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
    caplog.set_level(logging.DEBUG)


@pytest.mark.parametrize("query_key", ["token", "to%6ben", "%74oken", "X-Amz-Signature", "X%2dAmz%2dSignature"])
def test_transport_error_redacts_encoded_and_signed_queries(query_key):
    from agent.redact import redact_sensitive_text
    message = f"GET https://example.com/i?{query_key}={SECRET}&width=1024 failed"
    result = redact_sensitive_text(message, force=True, redact_url_credentials=True)
    assert SECRET not in result
    assert "width=1024" in result


def test_transport_error_redaction_is_idempotent_for_signed_url():
    from agent.redact import redact_sensitive_text
    first = redact_sensitive_text(ERROR, force=True, redact_url_credentials=True)
    assert redact_sensitive_text(first, force=True, redact_url_credentials=True) == first
    assert SECRET not in first


class _Adapter(BasePlatformAdapter):
    name = "diagnostic-test"
    platform = "telegram"
    supports_native_remote_images = True
    async def connect(self, **kwargs): return True
    async def disconnect(self): pass
    async def send(self, *args, **kwargs): return SendResult(success=True)
    async def get_chat_info(self, chat_id): return {}


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_error", [False, True])
async def test_base_image_failures_redact_traceback_and_preserve_transport(raise_error, caplog):
    adapter = object.__new__(_Adapter)
    adapter.send_image = AsyncMock(return_value=SendResult(success=False, error=ERROR))
    if raise_error:
        adapter.send_image.side_effect = RuntimeError(ERROR)
    await adapter.send_multiple_images("chat", [(URL, "preview")])
    adapter.send_image.assert_awaited_once_with(chat_id="chat", image_url=URL, caption="preview", metadata=None)
    assert "download rejected" in caplog.text
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform,method,dependency", [
    ("feishu", "send_image", "_download_remote_image"),
    ("feishu", "send_animation", "_download_remote_document"),
    ("matrix", "send_image", "_download_external_media_with_cap"),
    ("simplex", "send_image", "cache_image_from_url"),
    ("signal", "send_image", "cache_image_from_url"),
    ("signal", "send_multiple_images", "cache_image_from_url"),
])
async def test_download_diagnostics(platform, method, dependency, monkeypatch, caplog):
    module = import_module(f"gateway.platforms.{platform}" if platform == "signal" else f"plugins.platforms.{platform}.adapter")
    cls = getattr(module, {"feishu": "FeishuAdapter", "matrix": "MatrixAdapter", "simplex": "SimplexAdapter", "signal": "SignalAdapter"}[platform])
    adapter = object.__new__(cls)
    adapter.platform = SimpleNamespace(value=platform)
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    adapter._stop_typing_indicator = AsyncMock()
    if method == "send_animation":
        adapter.send_image = AsyncMock(return_value=SendResult(success=True))
    downloader = AsyncMock(side_effect=RuntimeError(ERROR))
    monkeypatch.setattr(module if dependency == "cache_image_from_url" else adapter, dependency, downloader)
    result = await getattr(adapter, method)("chat", [(URL, "")] if method == "send_multiple_images" else URL)
    assert downloader.await_args.args[0] == URL
    if result and not result.success:
        assert SECRET not in result.error
    assert "download" in caplog.text.lower()
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["prepare", "upload", "timeout"])
async def test_wecom_media_diagnostics(failure, caplog):
    from plugins.platforms.wecom.adapter import WeComAdapter
    adapter = object.__new__(WeComAdapter)
    adapter.platform = SimpleNamespace(value="wecom")
    adapter._prepare_outbound_media = AsyncMock(return_value={"rejected": False, "data": b"image", "final_type": "image", "file_name": "image.png"})
    adapter._upload_media_bytes = AsyncMock(side_effect=TimeoutError() if failure == "timeout" else RuntimeError(ERROR))
    adapter._cached_reply_req_id = MagicMock(return_value=None)
    adapter._find_active_turn_for_chat = MagicMock(return_value=False)
    adapter._stream_expired_chats = set()
    if failure == "prepare":
        adapter._prepare_outbound_media.side_effect = RuntimeError(ERROR)
    result = await adapter._send_media_source("chat", URL)
    adapter._prepare_outbound_media.assert_awaited_once_with(URL, file_name=None)
    assert not result.success
    assert SECRET not in result.error
    assert SECRET not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [URL, f"https://[invalid/x.png?token={SECRET}"])
async def test_weixin_ssrf_rejection_has_query_free_diagnostic(url, monkeypatch, caplog):
    from gateway.platforms.weixin import WeixinAdapter
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: False)
    download = AsyncMock()
    monkeypatch.setattr("gateway.platforms.weixin._download_bytes", download)
    adapter = object.__new__(WeixinAdapter)
    with pytest.raises(ValueError) as caught:
        await adapter._download_remote_media(url)
    assert "SSRF protection" in str(caught.value)
    assert SECRET not in str(caught.value)
    assert "blocked unsafe remote-media URL" in caplog.text
    assert SECRET not in caplog.text
    download.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["decline", "exception"])
async def test_relay_media_diagnostics_keep_payload_raw(failure, caplog):
    from gateway.relay.adapter import RelayAdapter
    adapter = object.__new__(RelayAdapter)
    adapter.descriptor = SimpleNamespace(supports_op=lambda op: True)
    adapter._platform_by_chat = {}
    adapter._transport = SimpleNamespace(send_outbound=AsyncMock(return_value={"success": False, "error": ERROR}))
    if failure == "exception":
        adapter._transport.send_outbound.side_effect = RuntimeError(ERROR)
    action = {"op": "send_media", "source_url": URL}
    assert await adapter._gated_op("chat", action) is None
    adapter._transport.send_outbound.assert_awaited_once_with(action, platform=None)
    assert action["source_url"] == URL
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["qq", "yuanbao", "teams"])
async def test_native_upload_errors(platform, monkeypatch, caplog):
    if platform == "qq":
        from gateway.platforms.qqbot.adapter import QQAdapter
        adapter = object.__new__(QQAdapter)
        adapter.platform = SimpleNamespace(value="qq")
        adapter._app_id = "test"
        adapter._ensure_connected = AsyncMock(return_value=True)
        adapter._guess_chat_type = MagicMock(return_value="c2c")
        adapter._upload_media = AsyncMock(side_effect=RuntimeError(ERROR))
        result = await adapter._send_media("chat", URL, 1, "image")
        assert adapter._upload_media.await_args.kwargs["url"] == URL
    elif platform == "yuanbao":
        from gateway.platforms.yuanbao import ImageUrlHandler
        downloader = AsyncMock(side_effect=RuntimeError(ERROR))
        monkeypatch.setattr("gateway.platforms.yuanbao.media_download_url", downloader)
        adapter = SimpleNamespace(name="yuanbao", MEDIA_MAX_SIZE_MB=10,
            _connection=SimpleNamespace(ws=True), _outbound=SimpleNamespace(slow_notifier=MagicMock()))
        result = await ImageUrlHandler().handle(adapter, "chat", image_url=URL)
        assert downloader.await_args.args[0] == URL
    else:
        import sys
        from plugins.platforms.teams.adapter import TeamsAdapter
        attachment = MagicMock()
        monkeypatch.setitem(sys.modules, "microsoft_teams.api", SimpleNamespace(
            Attachment=attachment, MessageActivityInput=MagicMock()))
        adapter = object.__new__(TeamsAdapter)
        adapter._app = True
        adapter._send_via_conv_ref = AsyncMock(side_effect=RuntimeError(ERROR))
        result = await adapter.send_image("chat", URL)
        assert attachment.call_args.kwargs["content_url"] == URL
    assert not result.success
    assert "download rejected" in result.error
    assert SECRET not in result.error
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
async def test_telegram_preserves_url_and_upload_retry_without_raw_tracebacks(monkeypatch, caplog):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = SimpleNamespace(value="telegram")
    adapter._bot = MagicMock()
    adapter._send_media = AsyncMock(side_effect=RuntimeError(ERROR))
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=SimpleNamespace(content=b"image", raise_for_status=lambda: None))
    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client", lambda **kwargs: client)
    await adapter.send_animation("chat", URL)
    assert adapter._send_media.await_count == 3
    assert adapter._send_media.await_args_list[0].kwargs["animation"] == URL
    assert adapter._send_media.await_args_list[1].kwargs["photo"] == URL
    assert adapter._send_media.await_args_list[2].kwargs["photo"] == b"image"
    client.get.assert_awaited_once_with(URL)
    assert "File upload send_photo also failed" in caplog.text
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send_image", "send_animation", "send_multiple_images"])
async def test_discord_download_errors(method, monkeypatch, caplog):
    from plugins.platforms.discord import adapter as module
    adapter = object.__new__(module.DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")
    adapter._client = True
    adapter._resolve_channel = AsyncMock(return_value=MagicMock())
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    if method == "send_animation":
        adapter.send_image = AsyncMock(return_value=SendResult(success=True))
    downloader = AsyncMock(side_effect=RuntimeError(ERROR))
    monkeypatch.setattr(module, "is_safe_url", lambda url: True)
    monkeypatch.setattr(module, "_read_url_image_with_redirect_guard", downloader)
    await getattr(adapter, method)("chat", [(URL, "")] if method == "send_multiple_images" else URL)
    assert downloader.await_args.args[1] == URL
    assert "download" in caplog.text.lower()
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform,method", [
    ("slack", "send_image"), ("slack", "send_multiple_images"),
    ("mattermost", "send_image"), ("mattermost", "send_multiple_images"),
])
async def test_http_download_errors(platform, method, monkeypatch, caplog):
    import aiohttp
    module = import_module(f"plugins.platforms.{platform}.adapter")
    cls = module.SlackAdapter if platform == "slack" else module.MattermostAdapter
    adapter = object.__new__(cls)
    adapter.platform = SimpleNamespace(value=platform)
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    adapter._app = True
    adapter._suppressed_ignored = MagicMock(return_value=False)
    adapter._dm_target = AsyncMock(return_value="chat")
    adapter._resolve_thread_ts = MagicMock(return_value=None)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(side_effect=RuntimeError(ERROR)) if platform == "slack" else MagicMock(side_effect=aiohttp.ClientError(ERROR))
    adapter._session = client
    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client", lambda **kwargs: client)
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
    await getattr(adapter, method)("chat", [(URL, "")] if method == "send_multiple_images" else URL)
    assert client.get.call_count >= 1
    assert all(call.args[0] == URL for call in client.get.call_args_list)
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_transport_error_redactor_failure_is_closed(monkeypatch):
    from gateway.platforms.base import redact_transport_error_text
    def broken(*args, **kwargs):
        raise RuntimeError(ERROR)
    monkeypatch.setattr("agent.redact.redact_sensitive_text", broken)
    assert redact_transport_error_text(RuntimeError(ERROR)) == "<transport error redacted>"


@pytest.mark.parametrize("url,expected", [
    (f"https://[invalid/x.png?token={SECRET}", "<invalid URL>"),
    (f"/image.png?token={SECRET}#private", "/image.png"),
    (f"//user:{SECRET}@example.com/image.png?token={SECRET}", "//example.com/.../image.png"),
    ("/tmp/ordinary image.png", "/tmp/ordinary image.png"),
])
def test_safe_url_for_log_handles_invalid_and_relative_urls(url, expected):
    from gateway.platforms.base import safe_url_for_log
    assert safe_url_for_log(url) == expected
    assert len(safe_url_for_log(url, max_len=5)) <= 5
