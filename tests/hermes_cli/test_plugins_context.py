"""PluginContext lives in hermes_cli/plugins_context.py and stays importable and patchable through
the hermes_cli.plugins facade."""

import logging
import typing

import pytest

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_plugin_context_is_defined_in_its_own_module_and_reexported():
    import hermes_cli.plugins_context as plugin_context_mod

    assert plugin_context_mod.PluginContext is PluginContext
    assert plugin_context_mod._UNSET is plugins_mod._UNSET


def test_ctx_methods_still_read_patches_on_the_facade(monkeypatch):
    config = {"plugins": {"entries": {"demo": {"settings": {"color": "blue"}}}}}
    monkeypatch.setattr(plugins_mod, "load_config_readonly", lambda *a, **k: config)
    ctx = PluginContext(PluginManifest(name="demo"), PluginManager())
    assert ctx.get_config("color") == "blue"


def test_ctx_logs_through_the_facade_logger(monkeypatch, caplog):
    patched = logging.getLogger("tests.plugin_context.facade_logger")
    monkeypatch.setattr(plugins_mod, "logger", patched)
    ctx = PluginContext(PluginManifest(name="demo"), PluginManager())
    with caplog.at_level(logging.WARNING, logger=patched.name):
        ctx.register_middleware("not_a_middleware_kind", lambda *a, **k: None)
    assert [r.name for r in caplog.records if "unknown middleware" in r.getMessage()] == [patched.name]


def test_ctx_leases_through_the_facade_replacement_coordinator(monkeypatch):
    real = plugins_mod.replacement_coordinator
    acquired = []

    class Recording:
        def acquire(self, slot, **kwargs):
            acquired.append(slot[0])
            return real.acquire(slot, **kwargs)

    monkeypatch.setattr(plugins_mod, "replacement_coordinator", Recording())
    ctx = PluginContext(PluginManifest(name="demo"), PluginManager())
    ctx.register_command("demo-cmd", lambda raw: None).dispose()
    assert acquired == ["manager_mapping"]


def test_ctx_reads_the_section_limit_from_the_facade(monkeypatch):
    monkeypatch.setattr(plugins_mod, "MAX_SYSTEM_PROMPT_SECTION_CHARS", 1)
    ctx = PluginContext(PluginManifest(name="demo"), PluginManager())
    with pytest.raises(ValueError, match="between 1 and 1"):
        ctx.register_system_prompt_section("demo.section", "x", max_chars=2)


def test_init_annotations_still_resolve():
    hints = typing.get_type_hints(PluginContext.__init__)
    assert hints == {"manifest": PluginManifest, "manager": PluginManager}
