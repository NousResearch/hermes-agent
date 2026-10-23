"""Tests for the bundled you-should-know plugin.

Behavior contracts: the observer flags a tool error the agent glossed over,
trivial/disabled sessions never trigger an auxiliary call, the digest is never
re-observed, and a denied model override fails closed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PLUGIN_INIT = Path(__file__).resolve().parents[2] / "plugins" / "you-should-know" / "__init__.py"
MODULE_NAME = "hermes_plugins.you_should_know"  # the loader's slug for plugins/you-should-know/


def _fresh_plugin():
    """Import the plugin module fresh (clears per-session buffers)."""
    sys.modules.pop(MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, PLUGIN_INIT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeObserver:
    """Canned ctx.llm: records calls, answers with a fixed digest."""

    def __init__(self, items, *, allow_model_override=False):
        from agent.plugin_llm import _TrustPolicy, make_plugin_llm_for_test

        self.calls = []
        self.items = items

        def _caller(**kw):
            self.calls.append(kw)
            response = SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"items": self.items})))],
                model="fake-observer",
                usage=None,
            )
            return ("fake-provider", "fake-observer", response)

        policy = _TrustPolicy(plugin_id="you-should-know",
                              allow_model_override=allow_model_override)
        self.llm = make_plugin_llm_for_test(
            plugin_id="you-should-know", policy=policy, sync_caller=_caller)

    @property
    def transcripts(self):
        # complete_structured messages: [system?, user]; user parts: [header, *blocks]
        return [kw["messages"][-1]["content"][1]["text"] for kw in self.calls]


class _FakeCtx:
    def __init__(self, settings, plugin_llm):
        self._settings = settings
        self.llm = plugin_llm
        self.hooks = {}
        self.aux_tasks = []

    def get_config(self, key, default=None):
        return self._settings.get(key, default)

    def register_hook(self, name, fn):
        self.hooks[name] = fn

    def register_auxiliary_task(self, key, *, display_name, description, defaults=None):
        self.aux_tasks.append(key)


def _setup(monkeypatch, settings=None, items=None, allow_model_override=False):
    """Build a fresh plugin module + fake ctx. The aux-task ownership seam is
    patched to reflect what the host does at register(): the plugin's own
    ``you_should_know_observer`` slot is owned by this plugin id (the
    registry-matching itself is framework code, covered by
    tests/agent/test_plugin_llm_task_routing.py)."""
    monkeypatch.setattr(
        "agent.plugin_llm._resolve_task_ownership",
        lambda plugin_id: (frozenset({"you_should_know_observer"}), frozenset()))
    mod = _fresh_plugin()
    observer = _FakeObserver(items or [], allow_model_override=allow_model_override)
    ctx = _FakeCtx(settings or {}, observer.llm)
    mod.register(ctx)
    return mod, ctx, observer


class TestDiscovery:
    def test_discovered_but_not_auto_loaded(self, tmp_path, monkeypatch):
        """Real discovery path with an isolated HERMES_HOME: the manifest is
        found, but the plugin stays opt-in (enable via `hermes plugins enable`)."""
        from hermes_cli import plugins as plugins_mod

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()

        loaded = manager._plugins.get("you-should-know")
        assert loaded is not None, "plugin not discovered"
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "").lower()

    def test_register_wires_hooks_and_aux_task(self, monkeypatch):
        mod, ctx, _ = _setup(monkeypatch)
        assert set(ctx.hooks) == {"transform_llm_output", "post_llm_call", "post_tool_call"}
        assert ctx.aux_tasks == ["you_should_know_observer"]


class TestObserverDigest:
    def test_glossed_over_tool_error_is_flagged(self, monkeypatch):
        items = [{"severity": "warning", "title": "3 tests failed in CI",
                  "detail": "The test run reported failures that the summary did not mention."}]
        mod, ctx, observer = _setup(monkeypatch, settings={"min_observe_chars": 10}, items=items)
        sid = "sess-1"

        # Turn 1, in real hook order: tool runs, transform observes, post_llm_call accumulates.
        mod.on_post_tool_call(session_id=sid, tool_name="bash",
                              result="ERROR: 3 tests failed (test_login, test_checkout, test_refund).")
        out = mod.on_transform_llm_output(response_text="All done — everything works!",
                                          session_id=sid, turn_id="t1")
        assert len(observer.calls) == 1, "exactly one auxiliary call for the epoch"
        assert "3 tests failed" in observer.transcripts[0], \
            "the glossed-over tool error reached the observer"
        assert out is not None
        assert out.startswith("All done — everything works!")
        assert "You should know" in out and "3 tests failed in CI" in out

        # The transformed text flows back through post_llm_call (as in production).
        mod.on_post_llm_call(session_id=sid, turn_id="t1", assistant_response=out)

        # Turn 2: the observer must not re-read its own digest (no feedback loop).
        mod.on_post_tool_call(session_id=sid, tool_name="bash", result="ok")
        out2 = mod.on_transform_llm_output(response_text="Second task finished cleanly, all green.",
                                            session_id=sid, turn_id="t2")
        assert len(observer.calls) == 2
        assert "You should know" not in observer.transcripts[1]
        assert "Second task finished cleanly" in observer.transcripts[1]
        assert out2 is not None and "You should know" in out2  # canned items still render

    def test_trivial_session_makes_no_observer_call(self, monkeypatch):
        mod, ctx, observer = _setup(monkeypatch)  # defaults: min_observe_chars=1500
        out = mod.on_transform_llm_output(response_text="4", session_id="s", turn_id="t")
        assert out is None
        assert observer.calls == []

    def test_disabled_plugin_is_inert(self, monkeypatch):
        mod, ctx, observer = _setup(monkeypatch, settings={"enabled": False, "min_observe_chars": 1})
        long_text = "substantive output " * 500
        mod.on_post_tool_call(session_id="s", tool_name="bash", result="boom")
        mod.on_post_llm_call(session_id="s", turn_id="t", assistant_response=long_text)
        assert mod.on_transform_llm_output(response_text=long_text, session_id="s", turn_id="t") is None
        assert observer.calls == []
        assert mod._BUFFERS == {}

    def test_model_override_denied_fails_closed(self, monkeypatch):
        mod, ctx, observer = _setup(monkeypatch, settings={"model": "cheap-model", "min_observe_chars": 1},
                                           allow_model_override=False)
        out = mod.on_transform_llm_output(response_text="substantive output " * 100,
                                          session_id="s", turn_id="t")
        assert out is None, "no digest without trust"
        assert observer.calls == [], "denied before any provider call (no silent downgrade)"
