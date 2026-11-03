"""Behaviour of the adaptive-context wiring helpers: config gate, compressor rebudget, /usage line."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import adaptive_context
from agent.adaptive_context import (
    AdaptiveContextTracker, init_adaptive_context, rebudget_on_backend_change, usage_line,
)


class _Compressor:
    def __init__(self):
        self.calls = []

    def update_model(self, **kwargs):
        self.calls.append(kwargs)


def _agent(tracker=None):
    return SimpleNamespace(
        _adaptive_context=tracker, context_compressor=_Compressor(), base_url="https://r.example/v1",
        api_key="k", provider="openrouter", api_mode="chat_completions",
    )


@pytest.mark.parametrize("cfg,enabled", [
    ({}, False),
    ({"compression": {}}, False),
    ({"compression": {"adaptive_context_window": False}}, False),
    ({"compression": {"adaptive_context_window": "false"}}, False),
    ({"compression": {"adaptive_context_window": True}}, True),
])
def test_init_is_gated_on_config(cfg, enabled):
    agent = SimpleNamespace()
    init_adaptive_context(agent, cfg)
    assert isinstance(agent._adaptive_context, AdaptiveContextTracker) is enabled


def test_rebudget_only_on_backend_change(monkeypatch):
    lookups = []

    def fake_ctx(model, base_url="", api_key="", provider=""):
        lookups.append((model, base_url, api_key, provider))
        return 131072

    monkeypatch.setattr("agent.model_metadata.get_model_context_length", fake_ctx)
    agent = _agent(AdaptiveContextTracker())
    rebudget_on_backend_change(agent, SimpleNamespace(model="llama-3.3-70b"))
    rebudget_on_backend_change(agent, SimpleNamespace(model="llama-3.3-70b"))
    assert agent.context_compressor.calls == []
    rebudget_on_backend_change(agent, SimpleNamespace(model="qwen-2.5-72b"))
    assert lookups == [("qwen-2.5-72b", "https://r.example/v1", "k", "openrouter")]
    assert agent.context_compressor.calls == [{
        "model": "qwen-2.5-72b", "context_length": 131072, "base_url": "https://r.example/v1",
        "api_key": "k", "provider": "openrouter", "api_mode": "chat_completions",
    }]


def test_rebudget_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr("agent.model_metadata.get_model_context_length",
                        lambda *a, **k: pytest.fail("lookup must not run when disabled"))
    agent = _agent(None)
    rebudget_on_backend_change(agent, SimpleNamespace(model="a"))
    rebudget_on_backend_change(agent, SimpleNamespace(model="b"))
    assert agent.context_compressor.calls == []


def test_failed_lookup_keeps_current_budget(monkeypatch):
    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr("agent.model_metadata.get_model_context_length", boom)
    agent = _agent(AdaptiveContextTracker())
    rebudget_on_backend_change(agent, SimpleNamespace(model="a"))
    rebudget_on_backend_change(agent, SimpleNamespace(model="b"))
    assert agent.context_compressor.calls == []
    assert agent._adaptive_context.change_count == 1


def test_usage_line_states(monkeypatch):
    seen = []
    monkeypatch.setattr("agent.i18n.t", lambda key, **kw: seen.append((key, kw)) or key)
    tracker = AdaptiveContextTracker()
    usage_line(tracker)
    tracker.observe("a")
    usage_line(tracker)
    tracker.observe("b")
    usage_line(tracker)
    assert seen == [
        ("gateway.usage.label_adaptive_context_pending", {}),
        ("gateway.usage.label_adaptive_context_baseline", {"model": "a"}),
        ("gateway.usage.label_adaptive_context_changes", {"model": "b", "count": 1}),
    ]


def test_usage_line_renders_english_catalog():
    tracker = AdaptiveContextTracker()
    assert adaptive_context.usage_line(tracker).startswith("Adaptive ctx:")
