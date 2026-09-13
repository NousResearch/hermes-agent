"""Custom image text sinks share the base signed-URL policy."""
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin,class_name", [
    ("buzz", "BuzzAdapter"), ("dingtalk", "DingTalkAdapter"),
    ("email", "EmailAdapter"), ("google_chat", "GoogleChatAdapter"),
    ("mattermost", "MattermostAdapter"), ("slack", "SlackAdapter"),
    ("qqbot", "QQAdapter"), ("wecom", "WeComMediaMixin"),
])
async def test_custom_plaintext_image_fallback_masks_signed_url(plugin, class_name, monkeypatch):
    path = ("gateway.platforms.qqbot.adapter" if plugin == "qqbot" else
            "plugins.platforms.wecom.media" if plugin == "wecom" else
            f"plugins.platforms.{plugin}.adapter")
    cls = getattr(import_module(path), class_name)
    sent = AsyncMock(return_value=SendResult(success=True))
    adapter = SimpleNamespace(send=sent, name=plugin, _log_tag=plugin, _app=True)
    url = "https://images.example/x.png?X-Amz-Signature=opaqueImageCredential&width=1024"
    if plugin == "google_chat":
        adapter._resolve_thread_id = lambda *a, **kw: None
        adapter._consume_typing_card_with_text = sent
    elif plugin in {"qqbot", "wecom"}:
        adapter._send_media = AsyncMock(return_value=SendResult(success=False, error="failed"))
        adapter._send_media_source = adapter._send_media
        adapter._is_url = adapter._looks_like_url = lambda _: True
    elif plugin == "mattermost":
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda _: False)
        await cls._send_url_as_file(adapter, "chat", url, "preview", None)
    elif plugin == "slack":
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda _: True)
        def fail_client(**kwargs):
            raise RuntimeError("download failed")
        monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client", fail_client)
    if plugin != "mattermost":
        await cls.send_image(adapter, "chat", url, caption="preview")
    sent.assert_awaited_once()
    call = sent.await_args
    text = call.kwargs.get("content", call.args[1] if len(call.args) > 1 else "")
    assert "opaqueImageCredential" not in text
    assert "X-Amz-Signature=***&width=1024" in text
