"""Tests for the generic ``transform_usage_extra`` plugin lifecycle hook.

This hook lets a plugin append an extra section to the end of the /usage
command render (after the account/credits blocks). It is observer-only: a
plugin returns a string block; the call site prints non-empty blocks and skips
the hook entirely when nothing is registered (zero idle cost).
"""

import types

from hermes_cli import plugins
from hermes_cli.plugins import PluginManager, VALID_HOOKS


def _fresh_manager(monkeypatch):
    """Install a clean PluginManager singleton and return it."""
    mgr = PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: mgr)
    return mgr


def test_valid_hooks_include_transform_usage_extra():
    assert "transform_usage_extra" in VALID_HOOKS


def test_has_hook_false_when_nothing_registered(monkeypatch):
    _fresh_manager(monkeypatch)
    assert plugins.has_hook("transform_usage_extra") is False


def test_transform_usage_extra_hook_returns_registered_blocks(monkeypatch):
    mgr = _fresh_manager(monkeypatch)

    def usage_block(**kwargs):
        return "📊 quota:\n• grok: Weekly 0% (reset today 18:00)"

    mgr._hooks.setdefault("transform_usage_extra", []).append(usage_block)

    assert plugins.has_hook("transform_usage_extra") is True
    results = plugins.invoke_hook(
        "transform_usage_extra", provider="grok", base_url="x", api_key="y", session_id="s1"
    )
    assert results == ["📊 quota:\n• grok: Weekly 0% (reset today 18:00)"]


def test_register_hook_accepts_new_hook_without_warning(monkeypatch, caplog):
    mgr = _fresh_manager(monkeypatch)
    ctx = plugins.PluginContext.__new__(plugins.PluginContext)
    ctx._manager = mgr
    ctx.manifest = types.SimpleNamespace(name="quota", key="quota")

    ctx.register_hook("transform_usage_extra", lambda **kw: "ue")

    assert mgr.has_hook("transform_usage_extra") is True
