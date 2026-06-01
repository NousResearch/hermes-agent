import pytest


@pytest.mark.asyncio
async def test_invoke_hook_async_awaits_sync_and_async_callbacks():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest, VALID_HOOKS

    assert "telegram_raw_update" in VALID_HOOKS

    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="raw-hook-test", source="test"), manager)
    calls = []

    def sync_callback(**kwargs):
        calls.append(("sync", kwargs["update"], kwargs["context"]))
        return None

    async def async_callback(**kwargs):
        calls.append(("async", kwargs["update"], kwargs["context"]))
        return {"action": "handled", "reason": "captured"}

    ctx.register_hook("telegram_raw_update", sync_callback)
    ctx.register_hook("telegram_raw_update", async_callback)

    results = await manager.invoke_hook_async(
        "telegram_raw_update",
        update="update-object",
        context="context-object",
        adapter="telegram-adapter",
        handler="text",
    )

    assert results == [{"action": "handled", "reason": "captured"}]
    assert calls == [
        ("sync", "update-object", "context-object"),
        ("async", "update-object", "context-object"),
    ]
