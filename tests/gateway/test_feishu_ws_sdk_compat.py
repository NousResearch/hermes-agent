"""Channel signaling stays required for the shipped SDK while legacy clients can connect."""
import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu import adapter as feishu
from plugins.platforms.feishu.ws_client import supports_keyword_argument


def test_declared_sdk_accepts_channel_signaling():
    from lark_oapi.ws import Client

    assert "extra_ua_tags" in inspect.signature(Client).parameters
    assert supports_keyword_argument(Client, "extra_ua_tags")


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_websocket_constructor_preserves_modern_and_legacy_contracts(monkeypatch, legacy):
    captured = {}

    class LegacyClient:
        def __init__(self, *, app_id, app_secret, log_level, event_handler, domain):
            captured.update(app_id=app_id, event_handler=event_handler, domain=domain)

    class ModernClient(LegacyClient):
        def __init__(self, *, extra_ua_tags, **kwargs):
            super().__init__(**kwargs)
            captured["extra_ua_tags"] = extra_ua_tags

    future = asyncio.get_running_loop().create_future()
    future.set_result(None)
    loop = SimpleNamespace(is_closed=lambda: False, run_in_executor=lambda *_args: future)
    adapter = feishu.FeishuAdapter(PlatformConfig())
    adapter._loop = loop
    adapter._app_id = "test-app"
    adapter._app_secret = "test-secret"
    adapter._event_handler = object()
    monkeypatch.setattr(feishu, "FEISHU_WEBSOCKET_AVAILABLE", True)
    monkeypatch.setattr(feishu, "lark", SimpleNamespace(LogLevel=SimpleNamespace(INFO="INFO")))
    monkeypatch.setattr(feishu, "FeishuWSClient", LegacyClient if legacy else ModernClient)
    monkeypatch.setattr(adapter, "_prepare_client", lambda: "test-domain")
    monkeypatch.setattr(adapter, "_hydrate_bot_identity", AsyncMock())
    monkeypatch.setattr(adapter, "_get_sdk_executor", lambda: None)

    await adapter._connect_websocket()

    assert captured["app_id"] == "test-app"
    assert captured["event_handler"] is adapter._event_handler
    assert captured["domain"] == "test-domain"
    assert adapter._ws_future is future
    if legacy:
        assert "extra_ua_tags" not in captured
    else:
        assert captured["extra_ua_tags"] == ["channel"]
