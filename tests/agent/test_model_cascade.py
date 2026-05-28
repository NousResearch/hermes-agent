import json
from types import SimpleNamespace

from agent.model_cascade import (
    classify_for_model_cascade,
    maybe_apply_model_cascade,
    resolve_model_cascade_target,
)


def test_resolve_model_cascade_target_disabled_keeps_classification_only():
    decision = resolve_model_cascade_target(
        {
            "enabled": False,
            "models": {"full": "gpt-5.4"},
        },
        "debug this failing test",
    )

    assert decision == {"tier": "full", "model": "", "provider": ""}


def test_resolve_model_cascade_target_uses_tier_model_and_provider():
    decision = resolve_model_cascade_target(
        {
            "enabled": True,
            "models": {
                "nano": "gpt-4.1-nano",
                "mini": "gpt-5.4-mini",
                "full": "gpt-5.4",
                "frontier": "gpt-5.5",
            },
            "providers": {"frontier": "openrouter"},
        },
        "!urgent complete rewrite of the gateway",
    )

    assert decision == {
        "tier": "frontier",
        "model": "gpt-5.5",
        "provider": "openrouter",
    }


def test_maybe_apply_model_cascade_calls_switch_model_for_configured_tier(monkeypatch):
    calls = []

    def fake_switch(agent, model, *, reason=None, provider=None):
        calls.append((agent, model, reason, provider))
        agent.model = "resolved/gpt-5.4"
        agent.provider = "openrouter"
        return json.dumps(
            {
                "success": True,
                "new_model": "resolved/gpt-5.4",
                "provider": "openrouter",
            }
        )

    monkeypatch.setattr("tools.switch_model_tool.switch_model_for_agent", fake_switch)
    agent = SimpleNamespace(
        model="gpt-4.1-nano",
        provider="openai",
        _model_cascade_config={
            "enabled": True,
            "models": {"full": "gpt-5.4"},
            "providers": {"full": "openrouter"},
        },
    )

    decision = maybe_apply_model_cascade(agent, "explain the architecture here")

    assert calls == [
        (agent, "gpt-5.4", "model_cascade:full", "openrouter")
    ]
    assert decision["applied"] is True
    assert decision["tier"] == "full"
    assert decision["resolved_model"] == "resolved/gpt-5.4"
    assert decision["resolved_provider"] == "openrouter"
    assert agent._model_cascade_last_decision == decision


def test_maybe_apply_model_cascade_skips_unconfigured_tier(monkeypatch):
    monkeypatch.setattr(
        "tools.switch_model_tool.switch_model_for_agent",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not switch")),
    )
    agent = SimpleNamespace(
        model="gpt-5.4",
        provider="openai",
        _model_cascade_config={"enabled": True, "models": {"frontier": "gpt-5.5"}},
    )

    assert maybe_apply_model_cascade(agent, "hello") is None
    assert agent._model_cascade_last_decision == {
        "tier": "nano",
        "model": "",
        "provider": "",
    }


def test_classify_for_model_cascade_delegates_to_complexity_classifier():
    assert classify_for_model_cascade("write a small script") == "mini"


def test_default_config_contains_disabled_model_cascade_shape():
    from hermes_cli.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG["model_cascade"]
    assert cfg["enabled"] is False
    assert set(cfg["models"]) == {"nano", "mini", "full", "frontier"}
    assert set(cfg["providers"]) == {"nano", "mini", "full", "frontier"}
