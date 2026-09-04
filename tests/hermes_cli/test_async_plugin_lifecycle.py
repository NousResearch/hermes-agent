from __future__ import annotations

import asyncio

import pytest

from hermes_cli.plugins import PluginManager


@pytest.mark.asyncio
async def test_invoke_hook_async_preserves_registration_order_and_contains_failures() -> None:
    manager = PluginManager()
    order: list[str] = []

    async def first(**_kwargs):
        order.append("first:start")
        await asyncio.sleep(0)
        order.append("first:end")
        return "async-result"

    def second(**_kwargs):
        order.append("second")
        return "sync-result"

    async def failing(**_kwargs):
        order.append("failing")
        raise RuntimeError("boom")

    manager._hooks["gateway_ready"] = [first, second, failing]

    results = await manager.invoke_hook_async("gateway_ready", gateway=object())

    assert results == ["async-result", "sync-result"]
    assert order == ["first:start", "first:end", "second", "failing"]


@pytest.mark.asyncio
async def test_invoke_hook_async_filters_additive_kwargs_for_legacy_callbacks() -> None:
    manager = PluginManager()

    async def legacy(gateway):
        return gateway

    manager._hooks["gateway_ready"] = [legacy]

    gateway = object()
    results = await manager.invoke_hook_async(
        "gateway_ready",
        gateway=gateway,
        adapters={},
        profile_adapters={},
    )

    assert results == [gateway]


@pytest.mark.asyncio
async def test_invoke_hook_async_times_out_each_callback_independently() -> None:
    manager = PluginManager()

    async def hanging(**_kwargs):
        await asyncio.sleep(10)

    async def later(**_kwargs):
        return "later-result"

    manager._hooks["gateway_stopping"] = [hanging, later]

    results = await manager.invoke_hook_async(
        "gateway_stopping",
        callback_timeout=0.001,
    )

    assert results == ["later-result"]


@pytest.mark.asyncio
async def test_invoke_hook_async_rejects_blocking_sync_callback_before_invocation() -> None:
    manager = PluginManager()
    called = False

    def blocking(**_kwargs):
        nonlocal called
        called = True

    async def later(**_kwargs):
        return "later-result"

    manager._hooks["gateway_stopping"] = [blocking, later]

    results = await manager.invoke_hook_async(
        "gateway_stopping",
        callback_timeout=0.001,
    )

    assert results == ["later-result"]
    assert called is False
