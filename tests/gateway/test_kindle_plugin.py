from __future__ import annotations

import asyncio

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_kindle_mod = load_plugin_adapter("kindle")
KindleAdapter = _kindle_mod.KindleAdapter


@pytest.mark.asyncio
async def test_send_waits_for_final_notification(monkeypatch) -> None:
    monkeypatch.setenv("KINDLE_INSECURE", "true")
    adapter = KindleAdapter(PlatformConfig(enabled=True))
    pending = asyncio.get_running_loop().create_future()
    adapter._pending["scribe"] = pending

    preview = await adapter.send("scribe", "working", metadata={"notify": False})
    assert preview.success is True
    assert pending.done() is False

    final = await adapter.send("scribe", "finished", metadata={"notify": True})
    assert final.success is True
    assert await pending == "finished"


@pytest.mark.asyncio
async def test_send_without_waiter_reports_failure(monkeypatch) -> None:
    monkeypatch.setenv("KINDLE_INSECURE", "true")
    adapter = KindleAdapter(PlatformConfig(enabled=True))

    result = await adapter.send("missing", "reply", metadata={"notify": True})

    assert result.success is False
    assert result.error == "no pending Kindle request"
