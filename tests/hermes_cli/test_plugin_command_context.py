from types import SimpleNamespace

import hermes_cli.plugins as plugins


def test_invoke_plugin_command_preserves_legacy_one_argument_handler(monkeypatch):
    monkeypatch.setattr(plugins, "get_plugin_command_handler", lambda _name: lambda args: f"legacy:{args}")
    assert plugins.invoke_plugin_command("demo", "x", event=object(), gateway=object()) == "legacy:x"


def test_invoke_plugin_command_passes_event_context_when_accepted(monkeypatch):
    def handler(args, *, event=None):
        return f"{event.source.platform.value}:{args}"

    monkeypatch.setattr(plugins, "get_plugin_command_handler", lambda _name: handler)
    event = SimpleNamespace(source=SimpleNamespace(platform=SimpleNamespace(value="telegram")))
    assert plugins.invoke_plugin_command("demo", "x", event=event, gateway=object()) == "telegram:x"
