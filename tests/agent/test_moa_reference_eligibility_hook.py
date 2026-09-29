"""moa_reference_eligibility lets a plugin narrow the MoA reference slots before fan-out."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import plugins as plugins_mod

SLOTS = [
    {"provider": "openrouter", "model": "a/cheap"},
    {"provider": "openrouter", "model": "b/pricey"},
]


def _response(content: str = "ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake")


@pytest.fixture
def reference_models_called(monkeypatch):
    called: list[str] = []

    def fake_call_llm(**kwargs):
        if kwargs.get("task") == "moa_reference":
            called.append(kwargs.get("model"))
        return _response("x")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    return called


def _register(monkeypatch, callback):
    manager = plugins_mod.PluginManager()
    manager._hooks.setdefault("moa_reference_eligibility", []).append(callback)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)


def _aggregate():
    from agent.moa_loop import aggregate_moa_context

    aggregate_moa_context(
        user_prompt="q", api_messages=[{"role": "user", "content": "q"}],
        reference_models=list(SLOTS),
        aggregator={"provider": "openrouter", "model": "agg/model"},
        preset_name="p",
    )


def test_hook_drops_slots_before_fanout(monkeypatch, reference_models_called):
    seen = {}

    def hook(slots, preset, **_):
        seen["preset"] = preset
        return [s for s in slots if s["model"] != "b/pricey"]

    _register(monkeypatch, hook)
    _aggregate()
    assert reference_models_called == ["a/cheap"]
    assert seen["preset"] == "p"


def test_hook_failure_or_non_list_keeps_all_slots(monkeypatch, reference_models_called):
    def boom(**_):
        raise RuntimeError("nope")

    _register(monkeypatch, boom)
    _aggregate()
    assert sorted(reference_models_called) == ["a/cheap", "b/pricey"]
