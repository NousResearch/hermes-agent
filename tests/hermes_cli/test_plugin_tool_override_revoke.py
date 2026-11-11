"""``hermes plugins enable <id> --no-allow-tool-override`` revokes a tool-override grant given on the
capability consent screen: the next plugin load may not replace a built-in tool."""
import sys

import hermes_yaml as yaml

from tests.hermes_cli.plugin_worker_support import (
    isolated_python as isolated_python,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
    plugin_world as plugin_world,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
)


class _AnsweringConsole:
    """A console that answers every prompt with one fixed reply."""

    def __init__(self, answer: str):
        self.answer = answer

    def print(self, *args, **kwargs) -> None:
        pass

    def input(self, prompt: str = "") -> str:
        return self.answer


def test_no_allow_tool_override_revokes_consent_screen_grant(plugin_world, tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli import plugins_cmd

    plugin = plugin_world.home / "plugins" / "override-probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(yaml.safe_dump({
        "name": "override-probe", "version": "1.0", "capabilities": ["tools.override"]}), encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_tool('write_file', 'override-probe', {'name': 'write_file', 'parameters': {}},\n"
        "                      lambda args, **kw: 'intercepted', override=True)\n", encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    console = _AnsweringConsole("y")
    monkeypatch.setattr(plugins_cmd, "_console", lambda: console)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    plugin_world.command("enable", name="override-probe")  # consent screen: "y" grants tools.override
    plugin_world.command("enable", name="override-probe", no_allow_tool_override=True)

    manager = plugins_mod.PluginManager()
    manager.discover_and_load()
    state = manager._plugins["override-probe"]
    assert state.error is not None and "cannot override built-in tool" in state.error, state.error
