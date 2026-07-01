"""Tests for the per-turn model-override branch of the ``pre_llm_call`` hook.

A ``pre_llm_call`` plugin may return ``{"model": ..., "provider": ...}`` to
route the current turn to a different model. ``_apply_pre_llm_call_model_override``
resolves credentials via the shared ``/model`` pipeline and applies the runtime
swap, swallowing any failure so routing can never abort a turn.
"""

import types

import agent.turn_context as tc


class _FakeAgent:
    def __init__(self):
        self.provider = "gemini"
        self.model = "gemini-2.5-flash"
        self.base_url = ""
        self.api_key = "k"


def _install_stubs(monkeypatch, *, result, record):
    """Stub the resolver + runtime swap that the override helper imports."""
    def _fake_resolve(**kwargs):
        record["resolve_kwargs"] = kwargs
        return result

    def _fake_runtime_switch(agent, new_model, new_provider, **kw):
        record["applied"] = (new_model, new_provider, kw)
        agent.model, agent.provider = new_model, new_provider

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_resolve)
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.switch_model", _fake_runtime_switch
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        "hermes_cli.config.get_compatible_custom_providers", lambda *a, **k: []
    )


def test_successful_override_applies_resolved_credentials(monkeypatch):
    record = {}
    result = types.SimpleNamespace(
        success=True, new_model="openai/gpt-oss-120b:free",
        target_provider="openrouter", api_key="rk",
        base_url="https://openrouter.ai/api/v1", api_mode="chat_completions",
        error_message="",
    )
    _install_stubs(monkeypatch, result=result, record=record)

    agent = _FakeAgent()
    tc._apply_pre_llm_call_model_override(agent, "gpt-oss-120b", "openrouter")

    assert record["applied"][0] == "openai/gpt-oss-120b:free"
    assert record["applied"][1] == "openrouter"
    assert record["applied"][2]["api_key"] == "rk"
    assert record["applied"][2]["base_url"] == "https://openrouter.ai/api/v1"
    assert agent.model == "openai/gpt-oss-120b:free"


def test_failed_resolution_keeps_current_model(monkeypatch):
    record = {}
    result = types.SimpleNamespace(
        success=False, new_model="", target_provider="", api_key="",
        base_url="", api_mode="", error_message="unknown model",
    )
    _install_stubs(monkeypatch, result=result, record=record)

    agent = _FakeAgent()
    tc._apply_pre_llm_call_model_override(agent, "nope", "openrouter")

    assert "applied" not in record          # runtime swap never attempted
    assert agent.model == "gemini-2.5-flash"  # unchanged


def test_resolver_exception_is_swallowed(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("resolver blew up")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _boom)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: {})
    monkeypatch.setattr(
        "hermes_cli.config.get_compatible_custom_providers", lambda *a, **k: []
    )

    agent = _FakeAgent()
    # Must not raise — a bad override can never abort the turn.
    tc._apply_pre_llm_call_model_override(agent, "x", "y")
    assert agent.model == "gemini-2.5-flash"
