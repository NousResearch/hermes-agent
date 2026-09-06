import asyncio

import pytest

from hermes_cli.plugins import (
    PluginCommandContext,
    _invoke_plugin_command_handler,
)


def test_public_plugin_module_has_no_ambient_context_api():
    import hermes_cli.plugins as plugins

    assert not hasattr(plugins, "get_plugin_command_context")
    assert not hasattr(plugins, "call_plugin_command_handler")


def test_legacy_handler_receives_only_raw_args():
    context = PluginCommandContext("telegram", "u", "c", "dm")
    seen = []

    def handler(raw_args):
        seen.append(raw_args)
        return "ok"

    entry = {"handler": handler, "authenticated_context": False}
    assert asyncio.run(_invoke_plugin_command_handler(entry, "args", context=context)) == "ok"
    assert seen == ["args"]


def test_opted_in_sync_handler_receives_explicit_context():
    context = PluginCommandContext("telegram", "u", "c", "group", "w", "p")
    seen = []

    def handler(raw_args, *, command_context):
        seen.append((raw_args, command_context))
        return "ok"

    entry = {"handler": handler, "authenticated_context": True}
    assert asyncio.run(_invoke_plugin_command_handler(entry, "args", context=context)) == "ok"
    assert seen == [("args", context)]


def test_opted_in_async_child_task_has_no_ambient_context():
    context = PluginCommandContext("discord", "u", "c", "dm")
    seen = []

    async def handler(_raw_args, *, command_context):
        async def child():
            seen.append(command_context)

        task = asyncio.create_task(child())
        await task
        return command_context

    entry = {"handler": handler, "authenticated_context": True}
    assert asyncio.run(_invoke_plugin_command_handler(entry, "", context=context)) == context
    # The child only sees the object because the handler explicitly passed/captured it;
    # there is no ambient getter or inherited ContextVar identity.
    assert seen == [context]


def test_opted_in_cli_invocation_receives_none():
    seen = []

    def handler(_raw_args, *, command_context):
        seen.append(command_context)
        return "unavailable" if command_context is None else "ok"

    entry = {"handler": handler, "authenticated_context": True}
    assert asyncio.run(_invoke_plugin_command_handler(entry, "", context=None)) == "unavailable"
    assert seen == [None]


def test_opted_in_handler_exception_does_not_leave_state_behind():
    context = PluginCommandContext("slack", "u", "c", "channel")

    def handler(_raw_args, *, command_context):
        assert command_context == context
        raise RuntimeError("boom")

    entry = {"handler": handler, "authenticated_context": True}
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(_invoke_plugin_command_handler(entry, "", context=context))


def test_concurrent_opted_in_handlers_keep_explicit_contexts_isolated():
    first = PluginCommandContext("telegram", "u1", "c1", "dm")
    second = PluginCommandContext("slack", "u2", "c2", "group")

    async def handler(raw_args, *, command_context):
        await asyncio.sleep(0)
        return raw_args, command_context

    entry = {"handler": handler, "authenticated_context": True}

    async def run():
        return await asyncio.gather(
            _invoke_plugin_command_handler(entry, "one", context=first),
            _invoke_plugin_command_handler(entry, "two", context=second),
        )

    assert asyncio.run(run()) == [("one", first), ("two", second)]
