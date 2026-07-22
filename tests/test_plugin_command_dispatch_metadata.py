from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _context():
    manager = PluginManager()
    manifest = PluginManifest(name="example", key="example")
    return PluginContext(manifest, manager), manager


def test_plugin_command_registration_records_active_session_bypass():
    ctx, manager = _context()

    def handler(raw_args):
        return f"handled {raw_args}"

    ctx.register_command("queue-status", handler, active_session_bypass=True)

    assert manager._plugin_commands["queue-status"]["active_session_bypass"] is True


def test_plugin_command_registration_defaults_to_no_active_session_bypass():
    ctx, manager = _context()

    def handler(raw_args):
        return f"handled {raw_args}"

    ctx.register_command("ordinary", handler)

    assert manager._plugin_commands["ordinary"]["active_session_bypass"] is False


def test_plugin_command_can_request_gateway_context(monkeypatch):
    ctx, manager = _context()
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda force=False: manager)
    seen = {}

    def handler(raw_args, *, event=None, gateway=None):
        seen["raw_args"] = raw_args
        seen["event"] = event
        seen["gateway"] = gateway
        return "ok"

    ctx.register_command("context-cmd", handler, wants_context=True)

    from hermes_cli.plugins import invoke_plugin_command

    result = invoke_plugin_command("context-cmd", "now", event="event-object", gateway="gateway-object")

    assert result == "ok"
    assert seen == {
        "raw_args": "now",
        "event": "event-object",
        "gateway": "gateway-object",
    }


def test_plugin_command_without_context_keeps_legacy_one_arg_call(monkeypatch):
    ctx, manager = _context()
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda force=False: manager)
    seen = []

    def handler(raw_args):
        seen.append(raw_args)
        return "ok"

    ctx.register_command("legacy", handler)

    from hermes_cli.plugins import invoke_plugin_command

    result = invoke_plugin_command("legacy", "args", event="ignored")

    assert result == "ok"
    assert seen == ["args"]


def test_async_plugin_command_context_result_can_be_resolved(monkeypatch):
    ctx, manager = _context()
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda force=False: manager)

    async def handler(raw_args, *, event=None):
        return f"{raw_args}:{event}"

    ctx.register_command("async-cmd", handler, wants_context=True)

    from hermes_cli.plugins import invoke_plugin_command, resolve_plugin_command_result

    result = invoke_plugin_command("async-cmd", "go", event="evt")

    assert resolve_plugin_command_result(result) == "go:evt"
