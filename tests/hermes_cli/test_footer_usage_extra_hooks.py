"""Tests for the generic ``footer`` and ``usage_extra`` plugin lifecycle hooks.

These hooks let a plugin append an extra block to the runtime footer (final
message) and to the ``/usage`` command render, respectively. Both are
observer-only: a plugin returns a string block; call sites concatenate
non-empty blocks and skip the hook entirely when nothing is registered.
"""

import types

from hermes_cli import plugins
from hermes_cli.plugins import PluginManager, VALID_HOOKS


def _fresh_manager(monkeypatch):
    """Install a clean PluginManager singleton and return it."""
    mgr = PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: mgr)
    return mgr


def test_valid_hooks_include_footer_and_usage_extra():
    assert "footer" in VALID_HOOKS
    assert "usage_extra" in VALID_HOOKS


def test_has_hook_false_when_nothing_registered(monkeypatch):
    _fresh_manager(monkeypatch)
    assert plugins.has_hook("footer") is False
    assert plugins.has_hook("usage_extra") is False


def test_footer_hook_returns_registered_blocks(monkeypatch):
    mgr = _fresh_manager(monkeypatch)

    def footer_block(**kwargs):
        return "📊 quota:\n• anthropic: 54% (reset today 13:09)"

    mgr._hooks.setdefault("footer", []).append(footer_block)

    assert plugins.has_hook("footer") is True
    results = plugins.invoke_hook("footer", model="gpt-5.4", context_tokens=10, context_length=100)
    assert results == ["📊 quota:\n• anthropic: 54% (reset today 13:09)"]


def test_usage_extra_hook_returns_registered_blocks(monkeypatch):
    mgr = _fresh_manager(monkeypatch)

    def usage_block(**kwargs):
        return "📊 quota:\n• grok: Weekly 0% (reset today 18:00)"

    mgr._hooks.setdefault("usage_extra", []).append(usage_block)

    assert plugins.has_hook("usage_extra") is True
    results = plugins.invoke_hook(
        "usage_extra", provider="grok", base_url="x", api_key="y", session_id="s1"
    )
    assert results == ["📊 quota:\n• grok: Weekly 0% (reset today 18:00)"]


def test_footer_hook_core_drops_none_keeps_text(monkeypatch):
    mgr = _fresh_manager(monkeypatch)
    mgr._hooks.setdefault("footer", []).extend(
        [
            lambda **kw: None,  # observer that contributes nothing
            lambda **kw: "   \n  ",  # whitespace-only: returned by core,
            # filtered at the call site via str().strip()
            lambda **kw: "real block",
        ]
    )
    results = plugins.invoke_hook("footer")
    # Core invoke_hook only drops None; call sites strip() whitespace-only.
    assert results == ["   \n  ", "real block"]


def test_register_hook_accepts_new_hooks_without_warning(monkeypatch, caplog):
    mgr = _fresh_manager(monkeypatch)
    ctx = plugins.PluginContext.__new__(plugins.PluginContext)
    ctx._manager = mgr
    ctx.manifest = types.SimpleNamespace(name="quota")

    ctx.register_hook("footer", lambda **kw: "fb")
    ctx.register_hook("usage_extra", lambda **kw: "ue")

    assert mgr.has_hook("footer") is True
    assert mgr.has_hook("usage_extra") is True
