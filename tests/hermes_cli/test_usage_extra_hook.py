"""Tests for the ``transform_usage_extra`` plugin hook.

The hook lets a plugin append a section to the end of ``/usage`` — per-provider
quota summaries, credit balances, spend estimates. It renders after the
account/credits blocks on every path, including the no-live-agent path the
TUI/Desktop slash worker uses (#42904), and it is fail-open: a broken plugin can
never abort ``/usage``.
"""

import types

from hermes_cli import plugins
from hermes_cli.cli_info_mixin import CLIInfoMixin
from hermes_cli.plugins import PluginManager, VALID_HOOKS


def _patch_hook(monkeypatch, callbacks):
    """Route lifecycle dispatch to *callbacks*; empty list means nothing registered."""
    from hermes_cli import lifecycle

    def _has(hook_name):
        return hook_name == "transform_usage_extra" and bool(callbacks)

    def _invoke(hook_name, **kwargs):
        assert hook_name == "transform_usage_extra"
        out = []
        for cb in callbacks:
            result = cb(**kwargs)
            if result is not None:
                out.append(result)
        return out

    monkeypatch.setattr(lifecycle, "has_hook", _has)
    monkeypatch.setattr(lifecycle, "invoke_hook", _invoke)


class _UsageHost:
    """Minimal host for ``CLIInfoMixin._print_usage_extra`` / ``_show_usage``."""

    def __init__(self, agent=None):
        self.agent = agent
        self.session_id = "local-usage-extra"

    _print_usage_extra = CLIInfoMixin._print_usage_extra
    _show_usage = CLIInfoMixin._show_usage

    def _print_account_limits(self):
        return False

    def _print_nous_credits_block(self):
        return False

    def _print_usage_cta(self):
        pass


def test_valid_hooks_include_transform_usage_extra():
    assert "transform_usage_extra" in VALID_HOOKS


def test_register_hook_accepts_transform_usage_extra_without_warning(caplog):
    mgr = PluginManager()
    ctx = plugins.PluginContext.__new__(plugins.PluginContext)
    ctx._manager = mgr
    ctx.manifest = types.SimpleNamespace(name="quota", key="quota")

    with caplog.at_level("WARNING"):
        ctx.register_hook("transform_usage_extra", lambda **kw: "block")

    assert mgr.has_hook("transform_usage_extra") is True
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_prints_registered_blocks_and_drops_empty(capsys, monkeypatch):
    _patch_hook(monkeypatch, [
        lambda **kw: None,
        lambda **kw: "   ",
        lambda **kw: "📊 quotas\n• anthropic: 54% left",
        lambda **kw: "second block",
    ])
    _UsageHost()._print_usage_extra()
    out = capsys.readouterr().out
    assert "• anthropic: 54% left" in out and "second block" in out
    assert out.index("anthropic") < out.index("second block")


def test_silent_when_nothing_registered(capsys, monkeypatch):
    _patch_hook(monkeypatch, [])
    _UsageHost()._print_usage_extra()
    assert capsys.readouterr().out == ""


def test_forwards_route_kwargs_without_credentials(capsys, monkeypatch):
    seen = {}
    _patch_hook(monkeypatch, [lambda **kw: seen.update(kw) or ""])
    agent = types.SimpleNamespace(provider="openai-codex", base_url=None, session_id="sess-42")
    _UsageHost(agent=agent)._print_usage_extra()
    assert seen == {"provider": "openai-codex", "base_url": None, "session_id": "sess-42"}
    assert "api_key" not in seen


def test_session_id_falls_back_to_the_cli_when_the_agent_has_none(capsys, monkeypatch):
    seen = {}
    _patch_hook(monkeypatch, [lambda **kw: seen.update(kw) or ""])
    agent = types.SimpleNamespace(provider="openai", base_url="https://x", session_id=None)
    _UsageHost(agent=agent)._print_usage_extra()
    assert seen["session_id"] == "local-usage-extra"


def test_fails_open_when_a_callback_raises(capsys, monkeypatch):
    """A raising plugin is isolated by PluginManager: the section still renders."""
    mgr = PluginManager()
    ctx = plugins.PluginContext.__new__(plugins.PluginContext)
    ctx._manager = mgr
    ctx.manifest = types.SimpleNamespace(name="quota", key="quota")

    def _boom(**kwargs):
        raise RuntimeError("plugin blew up")

    ctx.register_hook("transform_usage_extra", _boom)
    ctx.register_hook("transform_usage_extra", lambda **kw: "SURVIVED")
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: mgr)
    # _delivery_manager() boots discovery on first use; this manager is already built.
    monkeypatch.setattr(plugins, "_discover_once", lambda *a, **k: None, raising=False)

    _UsageHost()._print_usage_extra()
    assert "SURVIVED" in capsys.readouterr().out


def test_show_usage_renders_the_plugin_section_with_no_live_agent(capsys, monkeypatch):
    """/usage without an agent (TUI/Desktop slash worker) still shows the plugin block."""
    _patch_hook(monkeypatch, [lambda **kw: "QUOTA-BLOCK-42904"])
    _UsageHost()._show_usage()
    assert "QUOTA-BLOCK-42904" in capsys.readouterr().out
