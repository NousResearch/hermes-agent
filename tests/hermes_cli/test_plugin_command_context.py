import asyncio

import pytest

from hermes_cli.plugins import (
    PluginCommandContext,
    call_plugin_command_handler,
    get_plugin_command_context,
)


def test_plugin_command_context_is_bound_for_sync_handler_and_cleared_afterward():
    context = PluginCommandContext(
        platform="telegram",
        user_id="user-1",
        chat_id="chat-1",
        chat_type="group",
        scope_id="workspace-1",
        profile="profile-1",
    )

    seen = []

    def handler(raw_args):
        seen.append((raw_args, get_plugin_command_context()))
        return "ok"

    assert asyncio.run(call_plugin_command_handler(handler, "args", context=context)) == "ok"
    assert seen == [("args", context)]
    assert get_plugin_command_context() is None


def test_plugin_command_context_supports_async_handler_and_resets_on_exception():
    context = PluginCommandContext("discord", "user-2", "chat-2", "dm", None, None)

    async def handler(_raw_args):
        await asyncio.sleep(0)
        assert get_plugin_command_context() == context
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(call_plugin_command_handler(handler, "", context=context))
    assert get_plugin_command_context() is None


def test_plugin_command_context_is_isolated_between_concurrent_handlers():
    first = PluginCommandContext("telegram", "u1", "c1", "dm")
    second = PluginCommandContext("slack", "u2", "c2", "channel", "w2", "p2")

    async def handler(raw_args):
        await asyncio.sleep(0)
        observed = get_plugin_command_context()
        await asyncio.sleep(0)
        assert get_plugin_command_context() == observed
        return raw_args, observed

    async def run():
        return await asyncio.gather(
            call_plugin_command_handler(handler, "one", context=first),
            call_plugin_command_handler(handler, "two", context=second),
        )

    assert asyncio.run(run()) == [("one", first), ("two", second)]
    assert get_plugin_command_context() is None


def test_plugin_command_context_is_none_without_gateway_context():
    async def handler(_raw_args):
        return get_plugin_command_context()

    assert asyncio.run(call_plugin_command_handler(handler, "")) is None
    assert get_plugin_command_context() is None
