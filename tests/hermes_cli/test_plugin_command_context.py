import asyncio

import pytest

from hermes_cli.plugins import (
    PluginCommandContext,
    _dispatch_plugin_command,
    _invoke_plugin_command_handler,
    get_plugin_command_handler,
    resolve_plugin_command_result,
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


def test_central_dispatch_denies_authenticated_command_without_gateway_context(monkeypatch):
    seen = []

    def handler(_raw_args, *, command_context):
        seen.append(command_context)
        return "unsafe"

    entry = {"handler": handler, "authenticated_context": True}
    monkeypatch.setattr("hermes_cli.plugins._get_plugin_command_entry", lambda _name: entry)

    result = asyncio.run(_dispatch_plugin_command("secure", "args"))

    assert result.found is True
    assert result.denied is True
    assert result.output is None
    assert seen == []


def test_central_dispatch_authorizes_before_building_context_or_invoking(monkeypatch):
    events = []
    entry = {
        "handler": lambda _args, *, command_context: events.append(("handler", command_context)),
        "authenticated_context": True,
    }
    monkeypatch.setattr("hermes_cli.plugins._get_plugin_command_entry", lambda _name: entry)

    result = asyncio.run(_dispatch_plugin_command(
        "secure",
        "args",
        authorize=lambda name: events.append(("authorize", name)) or "denied by policy",
        context_factory=lambda: events.append(("context", None)) or PluginCommandContext(
            "slack", "u", "c", "channel"
        ),
    ))

    assert result.found is True
    assert result.denied is True
    assert result.denial_message == "denied by policy"
    assert events == [("authorize", "secure")]


def test_central_dispatch_invokes_authenticated_handler_after_admission(monkeypatch):
    context = PluginCommandContext("slack", "u", "c", "channel")
    events = []

    def handler(raw_args, *, command_context):
        events.append(("handler", raw_args, command_context))
        return "ok"

    entry = {"handler": handler, "authenticated_context": True}
    monkeypatch.setattr("hermes_cli.plugins._get_plugin_command_entry", lambda _name: entry)

    result = asyncio.run(_dispatch_plugin_command(
        "secure",
        "args",
        authorize=lambda name: events.append(("authorize", name)),
        context_factory=lambda: events.append(("context", None)) or context,
    ))

    assert result.found is True
    assert result.denied is False
    assert result.output == "ok"
    assert events == [
        ("authorize", "secure"),
        ("context", None),
        ("handler", "args", context),
    ]


def test_central_dispatch_preserves_legacy_cli_and_tui_commands(monkeypatch):
    seen = []
    entry = {
        "handler": lambda args: seen.append(args) or "ok",
        "authenticated_context": False,
    }
    monkeypatch.setattr("hermes_cli.plugins._get_plugin_command_entry", lambda _name: entry)

    result = asyncio.run(_dispatch_plugin_command("legacy", "args"))

    assert result.found is True
    assert result.denied is False
    assert result.output == "ok"
    assert seen == ["args"]


def test_central_dispatch_contains_handler_failure_for_known_command(monkeypatch):
    def handler(_args):
        raise RuntimeError("sensitive failure detail")

    entry = {"handler": handler, "authenticated_context": False}
    monkeypatch.setattr("hermes_cli.plugins._get_plugin_command_entry", lambda _name: entry)

    result = asyncio.run(_dispatch_plugin_command("legacy", "args"))

    assert result.found is True
    assert result.failed is True
    assert result.error_message == "Plugin command failed."


def test_central_dispatch_fails_closed_when_command_registry_lookup_raises(monkeypatch):
    def broken_lookup(_name):
        raise RuntimeError("sensitive registry detail")

    monkeypatch.setattr("hermes_cli.plugins._get_plugin_command_entry", broken_lookup)

    result = asyncio.run(_dispatch_plugin_command("possibly-secure", "secret args"))

    # Registry uncertainty blocks fallback rather than classifying the command
    # as absent and sending it through another slash-command or agent path.
    assert result.found is True
    assert result.failed is True
    assert result.error_message == "Plugin command failed."


def test_legacy_public_handler_lookup_cannot_bypass_authenticated_dispatch(monkeypatch):
    seen = []
    entry = {
        "handler": lambda _args, *, command_context: seen.append(command_context),
        "authenticated_context": True,
    }

    class Manager:
        _plugin_commands = {"secure": entry}

    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda: Manager())
    handler = get_plugin_command_handler("secure")

    assert handler is not None
    with pytest.raises(PermissionError, match="authenticated gateway request"):
        resolve_plugin_command_result(handler("args"))
    assert seen == []


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
