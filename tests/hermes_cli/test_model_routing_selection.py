from __future__ import annotations

from hermes_cli.model_routing import select_model_route


def _config():
    return {
        "routing": {
            "enabled": True,
            "model_selection": {
                "enabled": True,
                "default_tier": "balanced",
                "low_complexity_tier": "fast",
                "custodial_tier": "custodial_direct",
                "tier_map": {
                    "none": "balanced",
                    "think": "balanced",
                    "megathink": "deep",
                    "ultrathink": "deep",
                },
            },
            "tiers": {
                "fast": {"provider": "opencode-go", "model": "deepseek-v4-flash"},
                "balanced": {"provider": "opencode-go", "model": "deepseek-v4-pro"},
                "deep": {"provider": "openai-codex", "model": "gpt-5.5"},
                "custodial_direct": {
                    "provider": "openai-codex",
                    "model": "gpt-5.5",
                    "allow_fallback": False,
                },
            },
            "reasoning": {
                "default_tier": "none",
                "rules": [
                    {
                        "tier": "ultrathink",
                        "reason": "architecture",
                        "patterns": ["architect|system design|security audit"],
                    },
                    {
                        "tier": "megathink",
                        "reason": "debug-refactor",
                        "patterns": ["debug|root cause|refactor"],
                    },
                    {
                        "tier": "think",
                        "reason": "standard-coding",
                        "patterns": ["write|create|update|compare"],
                    },
                ],
            },
            "delegation": {
                "default_complexity": 4,
                "critical_threshold": -5,
                "complexity_rules": [
                    {
                        "name": "low-lookup-extraction",
                        "complexity": 2,
                        "patterns": ["^(what|where|who|when).{0,30}\\b|lookup|get me the|^list\\s|extract|format (this|the)|simple|quick\\s"],
                    },
                    {
                        "name": "high-architecture-debugging",
                        "complexity": 8,
                        "patterns": ["architect|design (a |the )?system|debug|refactor.*(system|architecture)|integrate|orchestrat|security (audit|review)"],
                    },
                ],
                "score_rules": [
                    {"name": "research-investigation", "score": 2, "patterns": ["research|compare|investigate"]},
                    {"name": "financial-sensitive", "score": -3, "patterns": ["payment|financial|invoice"]},
                    {"name": "secrets-security", "score": -3, "patterns": ["password|api.key|secret|credential"]},
                    {"name": "destructive-production", "score": -5, "patterns": ["deploy|delete|production|rm -rf"]},
                    {"name": "citadel-trading", "score": -10, "patterns": ["citadel|trading|exchange order"]},
                    {"name": "email-approval-flow", "score": -2, "patterns": ["send (an )?email|draft.*email"]},
                ],
            },
        }
    }


def test_standard_routing_uses_balanced_opencode_pro_for_normal_work():
    route = select_model_route(
        "Compare these two approaches and recommend the better Hermes routing policy.",
        _config(),
    )

    assert route is not None
    assert route["tier"] == "balanced"
    assert route["provider"] == "opencode-go"
    assert route["model"] == "deepseek-v4-pro"
    assert route["reasoning_tier"] == "think"


def test_low_complexity_lookup_uses_fast_opencode_flash():
    route = select_model_route("quick lookup: what timezone is Seattle in?", _config())

    assert route is not None
    assert route["tier"] == "fast"
    assert route["provider"] == "opencode-go"
    assert route["model"] == "deepseek-v4-flash"


def test_architecture_and_security_route_to_deep_chatgpt():
    route = select_model_route("Architect a secure multi-agent routing system.", _config())

    assert route is not None
    assert route["tier"] == "deep"
    assert route["provider"] == "openai-codex"
    assert route["model"] == "gpt-5.5"
    assert route["reasoning_tier"] == "ultrathink"


def test_custodial_routing_overrides_cheap_lanes_for_sensitive_work():
    route = select_model_route("Rotate the API key and deploy the production change.", _config())

    assert route is not None
    assert route["tier"] == "custodial_direct"
    assert route["provider"] == "openai-codex"
    assert route["model"] == "gpt-5.5"
    assert route["allow_fallback"] is False


def test_model_selection_disabled_returns_none():
    config = _config()
    config["routing"]["model_selection"]["enabled"] = False

    assert select_model_route("Compare these two approaches", config) is None
