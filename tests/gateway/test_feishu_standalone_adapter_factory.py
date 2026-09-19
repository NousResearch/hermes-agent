"""_standalone_send must build the adapter through the platform registry factory.

Regression: it hardcoded FeishuAdapter(pconfig), so a deployment that registered
its own adapter_factory silently got stock-adapter behaviour for every
out-of-process (cron) send.
"""
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platform_registry import platform_registry
from gateway.platforms.base import SendResult
from plugins.platforms.feishu import adapter as mod


class _StubAdapter:
    """Adapter stub that records which factory produced it."""

    def __init__(self, pconfig, *, used):
        self.pconfig = pconfig
        self._used = used
        self._client = None

    def _build_lark_client(self, domain):
        return SimpleNamespace()

    async def send(self, chat_id, message, metadata=None):
        _used.append((self._used, chat_id, message, metadata))
        return SendResult(success=True, message_id="om_sent", raw_response={})


_used: list = []


@pytest.mark.asyncio
async def test_standalone_send_uses_registered_adapter_factory(monkeypatch):
    _used.clear()

    def registered_factory(pconfig):
        return _StubAdapter(pconfig, used="registered")

    def stock_factory(pconfig):
        return _StubAdapter(pconfig, used="stock")

    monkeypatch.setattr(mod, "FeishuAdapter", stock_factory)
    monkeypatch.setattr(mod, "_load_lark_oapi", lambda: True)
    monkeypatch.setattr(
        platform_registry, "get",
        lambda name: SimpleNamespace(adapter_factory=registered_factory) if name == "feishu" else None,
    )

    result = await mod._standalone_send(PlatformConfig(), "oc_chat", "hello")

    assert [entry[0] for entry in _used] == ["registered"], (
        "standalone send must go through the registered platform factory; using the stock "
        "adapter silently drops plugin behaviour for out-of-process sends"
    )
    assert result.get("success") is True
    assert result.get("message_id") == "om_sent"


@pytest.mark.asyncio
async def test_standalone_send_falls_back_to_stock_adapter(monkeypatch):
    """No registry entry (e.g. plugins not loaded) must still deliver."""
    _used.clear()
    monkeypatch.setattr(mod, "FeishuAdapter", lambda pconfig: _StubAdapter(pconfig, used="stock"))
    monkeypatch.setattr(mod, "_load_lark_oapi", lambda: True)
    monkeypatch.setattr(platform_registry, "get", lambda name: None)

    result = await mod._standalone_send(PlatformConfig(), "oc_chat", "hello")

    assert [entry[0] for entry in _used] == ["stock"]
    assert result.get("success") is True
