"""``register_hook("transform_llm_output", fn, hold_stream=True)``: the declaration and its scope.

A plugin that rewrites the reply (redaction) must see it before the user does. Declaring
``hold_stream=True`` holds the streamed reply of every session in that plugin's profile; a plain
registration keeps streaming. The hold lives as long as the registration and never crosses into
another profile's plugin manager.
"""

import hermes_yaml as yaml
import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

REDACTOR = (
    "def register(ctx):\n"
    "    ctx.register_hook('transform_llm_output', lambda response_text, **_: response_text, hold_stream=True)\n"
)


def test_the_hold_lasts_as_long_as_the_registration_that_declared_it():
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="redactor", source="user"), manager)
    ctx.register_hook("transform_llm_output", lambda **_: None)
    assert manager.holds_stream() is False  # a rewrite that did not ask keeps streaming

    held = ctx.register_hook("transform_llm_output", lambda **_: None, hold_stream=True)
    assert manager.holds_stream() is True

    held.dispose()
    assert manager.holds_stream() is False
    assert manager.has_hook("transform_llm_output")  # the undeclared registration is still live


def test_only_the_declaring_profile_holds(tmp_path, monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.plugins_dispatch import stream_held

    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "no-bundled"))
    homes = {name: tmp_path / name for name in ("alpha", "beta")}
    for name, home in homes.items():
        plugin = home / "plugins" / "redactor"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(yaml.safe_dump({"name": "redactor", "version": "0.1.0"}))
        (plugin / "__init__.py").write_text(REDACTOR if name == "alpha" else REDACTOR.replace(", hold_stream=True", ""))
        (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["redactor"]}}))

    plugins._reset_plugin_managers_for_tests()
    try:
        seen = []
        for name in ("alpha", "beta", "alpha"):
            monkeypatch.setenv("HERMES_HOME", str(homes[name]))
            seen.append(stream_held())
        assert seen == [True, False, True]
    finally:
        plugins._reset_plugin_managers_for_tests()


@pytest.mark.parametrize("hook", ["post_llm_call", "transform_tool_result"])
def test_a_hold_on_a_hook_that_does_not_rewrite_the_reply_holds_nothing(hook, caplog):
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="observer", source="user"), manager)
    ctx.register_hook(hook, lambda **_: None, hold_stream=True)
    assert manager.holds_stream() is False
    assert "hold_stream" in caplog.text


def test_a_turn_starting_during_a_plugin_reload_sees_the_settled_hold(tmp_path, monkeypatch):
    """``reload-plugins`` unloads every registration, then loads them again. A turn that asks in
    between must wait for the reload instead of reading the empty ledger and streaming raw text."""
    import sys
    import threading
    import types

    from hermes_cli import plugins

    gate = types.ModuleType("hold_stream_reload_gate")
    gate.loading, gate.entered, gate.release = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setitem(sys.modules, gate.__name__, gate)
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "no-bundled"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "redactor"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(yaml.safe_dump({"name": "redactor", "version": "0.1.0"}))
    (plugin / "__init__.py").write_text(
        "import hold_stream_reload_gate as gate\n"
        "def register(ctx):\n"
        "    if gate.loading.is_set():\n"
        "        gate.entered.set()\n"
        "        gate.release.wait(10)\n"
        "    ctx.register_hook('transform_llm_output', lambda response_text, **_: response_text, hold_stream=True)\n")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["redactor"]}}))

    plugins._reset_plugin_managers_for_tests()
    try:
        manager = plugins.get_plugin_manager()
        manager.discover_and_load()
        assert manager.holds_stream() is True
        gate.loading.set()
        reload = threading.Thread(target=manager.discover_and_load, kwargs={"force": True})
        reload.start()
        assert gate.entered.wait(10)  # the reload unloaded everything and is re-registering
        answers: list[bool] = []
        reading = threading.Event()

        def read() -> None:
            reading.set()
            answers.append(manager.holds_stream())

        turn = threading.Thread(target=read)
        turn.start()
        assert reading.wait(10)  # the read has started: from here only the reload lock can hold it
        turn.join(2.0)
        assert answers == [] and turn.is_alive()  # the turn waits while the plugin is mid-reload
        gate.release.set()
        reload.join(10)
        turn.join(10)
        assert answers == [True]
    finally:
        gate.release.set()
        plugins._reset_plugin_managers_for_tests()
