"""PluginContext lives in hermes_cli/plugin_context.py and stays importable and patchable through
the hermes_cli.plugins facade."""

import typing

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_plugin_context_is_defined_in_its_own_module_and_reexported():
    import hermes_cli.plugin_context as plugin_context_mod

    assert plugin_context_mod.PluginContext is PluginContext
    assert plugin_context_mod._UNSET is plugins_mod._UNSET
    assert plugin_context_mod._ignore_after_abandoned_load is plugins_mod._ignore_after_abandoned_load


def test_ctx_methods_still_read_patches_on_the_facade(monkeypatch):
    config = {"plugins": {"entries": {"demo": {"settings": {"color": "blue"}}}}}
    monkeypatch.setattr(plugins_mod, "load_config_readonly", lambda *a, **k: config)
    ctx = PluginContext(PluginManifest(name="demo"), PluginManager())
    assert ctx.get_config("color") == "blue"


def test_init_annotations_still_resolve():
    hints = typing.get_type_hints(PluginContext.__init__)
    assert hints == {"manifest": PluginManifest, "manager": PluginManager}
