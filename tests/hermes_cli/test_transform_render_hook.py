"""Tests for the ``transform_render`` plugin hook.

One render-extension surface for every place Hermes appends plugin text to
something it already rendered: ``section="footer"`` (the final gateway reply)
and ``section="usage"`` (the ``/usage`` command output). Callbacks receive
``section`` plus that surface's payload, return a string block, and non-empty
blocks are joined with one blank line. The shared machinery lives in
``hermes_cli.lifecycle.collect_render_blocks``; the surface-specific call sites
are covered where they live (``tests/gateway/test_runtime_footer.py`` and
``test_show_usage_renders_the_plugin_section_with_no_live_agent`` below).
"""

import types

from hermes_cli import plugins
from hermes_cli.cli_info_mixin import CLIInfoMixin
from hermes_cli.lifecycle import collect_render_blocks
from hermes_cli.plugins import PluginManager, VALID_HOOKS


def _patch_hook(monkeypatch, callbacks):
    """Route lifecycle dispatch to *callbacks*; empty list means nothing registered."""
    from hermes_cli import lifecycle

    def _has(hook_name):
        return hook_name == "transform_render" and bool(callbacks)

    def _invoke(hook_name, **kwargs):
        assert hook_name == "transform_render"
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


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------


def test_valid_hooks_carry_one_render_hook_not_two():
    assert "transform_render" in VALID_HOOKS
    # The split hook names never shipped; they must not linger as inert surfaces.
    assert "transform_footer" not in VALID_HOOKS
    assert "transform_usage_extra" not in VALID_HOOKS


def test_register_hook_accepts_transform_render_without_warning(caplog):
    mgr = PluginManager()
    ctx = plugins.PluginContext.__new__(plugins.PluginContext)
    ctx._manager = mgr
    ctx.manifest = types.SimpleNamespace(name="quota", key="quota")

    with caplog.at_level("WARNING"):
        ctx.register_hook("transform_render", lambda **kw: "block")

    assert mgr.has_hook("transform_render") is True
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_unknown_hook_still_warns_so_the_allowlist_keeps_teeth(caplog):
    mgr = PluginManager()
    ctx = plugins.PluginContext.__new__(plugins.PluginContext)
    ctx._manager = mgr
    ctx.manifest = types.SimpleNamespace(name="quota", key="quota")

    with caplog.at_level("WARNING"):
        ctx.register_hook("transform_render_typo", lambda **kw: "block")

    assert [r for r in caplog.records if r.levelname == "WARNING"]


# ---------------------------------------------------------------------------
# collect_render_blocks — the shared machinery
# ---------------------------------------------------------------------------


def test_section_and_payload_reach_every_callback(monkeypatch):
    seen = []

    def _capture(**kwargs):
        seen.append(kwargs)
        return ""

    _patch_hook(monkeypatch, [_capture])
    collect_render_blocks("usage", provider="openai-codex", base_url=None, session_id="s-1")
    assert seen == [
        {"section": "usage", "provider": "openai-codex", "base_url": None, "session_id": "s-1"},
    ]


def test_blocks_join_in_registration_order_and_empty_ones_are_dropped(monkeypatch):
    _patch_hook(monkeypatch, [
        lambda **kw: None,
        lambda **kw: "   \n  ",
        lambda **kw: "📊 quotas\n• anthropic: 54%",
        lambda **kw: "second block",
    ])
    assert collect_render_blocks("footer", model="m") == "📊 quotas\n• anthropic: 54%\n\nsecond block"


def test_no_registered_callback_costs_one_lookup_and_returns_empty(monkeypatch):
    _patch_hook(monkeypatch, [])
    assert collect_render_blocks("footer", model="m") == ""


def test_dispatch_failure_returns_empty_instead_of_raising(monkeypatch):
    from hermes_cli import lifecycle

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: True)

    def _boom(hook_name, **kwargs):
        raise RuntimeError("dispatch blew up")

    monkeypatch.setattr(lifecycle, "invoke_hook", _boom)
    assert collect_render_blocks("footer", model="m") == ""


def test_a_raising_callback_is_isolated_by_the_real_dispatcher(monkeypatch, capsys):
    mgr = PluginManager()
    ctx = plugins.PluginContext.__new__(plugins.PluginContext)
    ctx._manager = mgr
    ctx.manifest = types.SimpleNamespace(name="quota", key="quota")

    def _boom(**kwargs):
        raise RuntimeError("plugin blew up")

    ctx.register_hook("transform_render", _boom)
    ctx.register_hook("transform_render", lambda **kw: "SURVIVED")
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: mgr)
    monkeypatch.setattr(plugins, "_discover_once", lambda *a, **k: None, raising=False)

    _UsageHost()._print_usage_extra()
    assert "SURVIVED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# /usage call site — section="usage"
# ---------------------------------------------------------------------------


def test_usage_section_prints_blocks(capsys, monkeypatch):
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return "📊 quotas\n• anthropic: 54% left"

    _patch_hook(monkeypatch, [_capture])
    agent = types.SimpleNamespace(provider="openai-codex", base_url=None, session_id="sess-42")
    _UsageHost(agent=agent)._print_usage_extra()
    assert "• anthropic: 54% left" in capsys.readouterr().out
    assert seen["section"] == "usage"
    assert "api_key" not in seen


def test_usage_silent_when_nothing_registered(capsys, monkeypatch):
    _patch_hook(monkeypatch, [])
    _UsageHost()._print_usage_extra()
    assert capsys.readouterr().out == ""


def test_usage_session_id_falls_back_to_the_cli_when_the_agent_has_none(monkeypatch):
    seen = {}
    _patch_hook(monkeypatch, [lambda **kw: seen.update(kw) or ""])
    agent = types.SimpleNamespace(provider="openai", base_url="https://x", session_id=None)
    _UsageHost(agent=agent)._print_usage_extra()
    assert seen["session_id"] == "local-usage-extra"


def test_show_usage_renders_the_plugin_section_with_no_live_agent(capsys, monkeypatch):
    """/usage without an agent (TUI/Desktop slash worker) still shows the plugin block."""
    _patch_hook(monkeypatch, [lambda **kw: "QUOTA-BLOCK-42904"])
    _UsageHost()._show_usage()
    assert "QUOTA-BLOCK-42904" in capsys.readouterr().out
