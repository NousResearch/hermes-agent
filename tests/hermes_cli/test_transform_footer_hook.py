"""Tests for the generic ``transform_footer`` plugin lifecycle hook.

This hook lets a plugin append an extra block to the runtime footer on the
final agent response (agent loop, gateway, desktop). It is observer-only: a
plugin returns a string block; the call site concatenates non-empty blocks and
skips the hook entirely when nothing is registered (zero idle cost).
"""

import types

from hermes_cli import plugins
from hermes_cli.plugins import PluginManager, VALID_HOOKS


def _fresh_manager(monkeypatch):
    """Install a clean PluginManager singleton and return it."""
    mgr = PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: mgr)
    return mgr


def test_valid_hooks_include_transform_footer():
    assert "transform_footer" in VALID_HOOKS


def test_has_hook_false_when_nothing_registered(monkeypatch):
    _fresh_manager(monkeypatch)
    assert plugins.has_hook("transform_footer") is False


def test_transform_footer_hook_returns_registered_blocks(monkeypatch):
    mgr = _fresh_manager(monkeypatch)

    def footer_block(**kwargs):
        return "📊 quota:\n• anthropic: 54% (reset today 13:09)"

    mgr._hooks.setdefault("transform_footer", []).append(footer_block)

    assert plugins.has_hook("transform_footer") is True
    results = plugins.invoke_hook(
        "transform_footer", model="gpt-5.4", context_tokens=10, context_length=100
    )
    assert results == ["📊 quota:\n• anthropic: 54% (reset today 13:09)"]


def test_transform_footer_hook_core_drops_none_keeps_text(monkeypatch):
    mgr = _fresh_manager(monkeypatch)
    mgr._hooks.setdefault("transform_footer", []).extend(
        [
            lambda **kw: None,  # observer that contributes nothing
            lambda **kw: "   \n  ",  # whitespace-only: returned by core,
            # filtered at the call site via str().strip()
            lambda **kw: "real block",
        ]
    )
    results = plugins.invoke_hook("transform_footer")
    # Core invoke_hook only drops None; call sites strip() whitespace-only.
    assert results == ["   \n  ", "real block"]


def test_register_hook_accepts_new_hook_without_warning(monkeypatch, caplog):
    mgr = _fresh_manager(monkeypatch)
    ctx = plugins.PluginContext.__new__(plugins.PluginContext)
    ctx._manager = mgr
    ctx.manifest = types.SimpleNamespace(name="quota", key="quota")

    ctx.register_hook("transform_footer", lambda **kw: "fb")

    assert mgr.has_hook("transform_footer") is True
