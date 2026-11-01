"""A ``context.engine``-selected engine keeps the full plugin surface: its ``register(ctx)`` hooks and
tools reach the plugin manager through the real loaders, without duplicates (#143118).

Each test uses its own engine name: an engine module stays cached in ``sys.modules`` by name."""

import sys
import textwrap

import pytest

from hermes_cli.plugins import get_plugin_manager

def _install(home, monkeypatch, name, *, enabled=False, extra="", isolation="in_process"):
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(home / "empty"))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.chdir(home)
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled: {f'[{name}]' if enabled else '[]'}\n  isolation: {isolation}\ncontext:\n  engine: {name}\n"
    )
    plugin = home / "plugins" / name
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(f"name: {name}\nversion: 1.0.0\nkind: standalone\n")
    (plugin / "__init__.py").write_text(textwrap.dedent('''\
        from agent.context_engine import ContextEngine

        class Engine(ContextEngine):
            @property
            def name(self):
                return "eng_demo"
            def update_from_response(self, usage):
                pass
            def should_compress(self, prompt_tokens=None):
                return False
            def compress(self, messages, current_tokens=None):
                return messages

        def pre_llm_call(**kwargs):
            return {"context": "eng_demo"}

        def tool_handler(args, **kwargs):
            return "ok"

        def register(ctx):
            ctx.register_context_engine(Engine())
        %s
            ctx.register_hook("pre_llm_call", pre_llm_call)
            ctx.register_tool(name="eng_demo_tool", toolset="eng_demo", handler=tool_handler,
                              schema={"name": "eng_demo_tool", "description": "demo",
                                      "parameters": {"type": "object", "properties": {}}})
            ctx.register_command("eng_demo-cmd", lambda raw_args: "ok", description="demo")
    ''').replace("eng_demo", name) % textwrap.indent(extra, "    "))
    return get_plugin_manager()


def _contexts(manager):
    return manager.invoke_hook("pre_llm_call", session_id="")


def _has_tool(manager, name):
    from tools.registry import registry
    return registry.get_entry(f"{name}_tool", scope=manager.scope_key) is not None


def _select(name):
    from agent.agent_init import _select_context_engine
    engine = _select_context_engine({"context": {"engine": name}})
    assert engine is not None and engine.name == name
    return engine


def test_selected_engine_registers_hook_and_tool(tmp_path, monkeypatch):
    manager = _install(tmp_path, monkeypatch, "eng_selected")
    try:
        _select("eng_selected")
        assert manager.has_hook("pre_llm_call")
        assert _contexts(manager) == [{"context": "eng_selected"}]
        assert _has_tool(manager, "eng_selected")
    finally:
        manager.unload()


def test_reloading_engine_replaces_its_hooks(tmp_path, monkeypatch):
    manager = _install(tmp_path, monkeypatch, "eng_reload")
    try:
        _select("eng_reload")
        _select("eng_reload")  # every agent build runs register() again
        assert _contexts(manager) == [{"context": "eng_reload"}]
        assert _has_tool(manager, "eng_reload")
        manager.unload()
        assert _contexts(manager) == []
    finally:
        manager.unload()


@pytest.mark.parametrize("order", ["plugin_first", "engine_first"])
def test_engine_also_in_plugins_enabled_runs_hook_once(tmp_path, monkeypatch, order):
    name = f"eng_dual_{order}"
    manager = _install(tmp_path, monkeypatch, name, enabled=True)
    try:
        if order == "plugin_first":
            manager.discover_and_load()
        _select(name)
        if order == "engine_first":
            manager.discover_and_load()
        _select(name)
        assert _contexts(manager) == [{"context": name}]
    finally:
        manager.unload()


@pytest.mark.parametrize("order", ["plugin_first", "engine_first"])
def test_hosted_engine_also_in_plugins_enabled_runs_hook_once(tmp_path, monkeypatch, order):
    # A hosted plugin has no module in this process, so general discovery must key its hooks by directory.
    name = f"eng_hosted_dual_{order}"
    manager = _install(tmp_path, monkeypatch, name, enabled=True, isolation="host")
    host = manager._plugin_host()
    try:
        if order == "plugin_first":
            manager.discover_and_load()
        _select(name)
        if order == "engine_first":
            manager.discover_and_load()
        _select(name)
        assert _contexts(manager) == [{"context": name}]
    finally:
        manager.unload()
        host.shutdown()


def test_second_profile_with_the_same_engine_runs_only_its_own_hook(tmp_path, monkeypatch):
    # Two profiles in one process install an engine under the same name, also in plugins.enabled. The
    # engine module stays cached by name (#134347), so the hook group must be keyed on each profile's
    # own directory; keyed on the cached module's file, beta would also run alpha's hook.
    name = "eng_two_homes"

    def install(tag):
        manager = _install(tmp_path / tag, monkeypatch, name, enabled=True)
        init = tmp_path / tag / "plugins" / name / "__init__.py"
        init.write_text(init.read_text().replace(f'{{"context": "{name}"}}', f'{{"context": "{tag}"}}'))
        return manager

    alpha = install("alpha")
    try:
        _select(name)
        beta = install("beta")
        try:
            beta.discover_and_load()
            _select(name)
            _select(name)
            assert _contexts(beta) == [{"context": "beta"}]
        finally:
            beta.unload()
    finally:
        alpha.unload()


def test_failing_secondary_registration_keeps_engine(tmp_path, monkeypatch):
    # override=True without operator consent raises PluginToolOverrideError inside PluginContext.
    manager = _install(tmp_path, monkeypatch, "eng_raises", extra=(
        'ctx.register_tool(name="eng_demo_other", toolset="eng_demo", handler=tool_handler,\n'
        '                  schema={"name": "eng_demo_other"}, override=True)\n'))
    try:
        _select("eng_raises")
        assert _contexts(manager) == [{"context": "eng_raises"}]
        assert _has_tool(manager, "eng_raises")
    finally:
        manager.unload()


def test_listing_engines_registers_nothing(tmp_path, monkeypatch):
    manager = _install(tmp_path, monkeypatch, "eng_listed")
    from plugins.context_engine import discover_context_engines

    try:
        assert ("eng_listed", "", True) in discover_context_engines()
        assert not manager.has_hook("pre_llm_call")
        assert not _has_tool(manager, "eng_listed")
        assert "eng_listed-cmd" not in manager._plugin_commands
    finally:
        manager.unload()


def test_hosted_engine_forwards_and_replaces_hooks(tmp_path, monkeypatch):
    manager = _install(tmp_path, monkeypatch, "eng_hosted", isolation="host")
    host = manager._plugin_host()
    try:
        _select("eng_hosted")
        _select("eng_hosted")
        assert not any("eng_hosted" in module for module in sys.modules)  # its code ran in the host
        assert _contexts(manager) == [{"context": "eng_hosted"}]
        assert _has_tool(manager, "eng_hosted")
    finally:
        manager.unload()
        host.shutdown()
