"""_standalone_send must build the adapter through the platform registry factory.

Regression: it hardcoded FeishuAdapter(pconfig), so a deployment that registered
its own adapter_factory silently got stock-adapter behaviour for every
out-of-process (cron) send.
"""
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import SendResult
from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
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
async def test_standalone_send_uses_registered_adapter_factory(monkeypatch, tmp_path):
    _used.clear()

    def stock_factory(pconfig):
        return _StubAdapter(pconfig, used="stock")

    monkeypatch.setattr(mod, "FeishuAdapter", stock_factory)
    monkeypatch.setattr(mod, "_load_lark_oapi", lambda: True)
    homes = [tmp_path / name for name in ("a", "b")]
    for name, home in zip(("a", "b"), homes):
        home.mkdir()
        platform_registry.register(
            PlatformEntry(
                name="feishu", label="Feishu", source="plugin", check_fn=lambda: True,
                adapter_factory=lambda pconfig, used=name: _StubAdapter(pconfig, used=used),
            ),
            scope=hermes_home_key(home),
        )

    try:
        for home in (homes[0], homes[1], homes[0]):
            token = set_hermes_home_override(str(home))
            try:
                result = await mod._standalone_send(PlatformConfig(), "oc_chat", "hello")
            finally:
                reset_hermes_home_override(token)
            assert result.get("success") is True
            assert result.get("message_id") == "om_sent"

        assert [entry[0] for entry in _used] == ["a", "b", "a"], (
            "standalone send must use each profile's registered factory; using the stock "
            "adapter silently drops plugin behaviour for out-of-process sends"
        )
    finally:
        for home in homes:
            platform_registry.unregister("feishu", scope=hermes_home_key(home))


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
