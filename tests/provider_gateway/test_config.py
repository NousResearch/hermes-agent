from __future__ import annotations

from hermes_cli.config import DEFAULT_CONFIG
from provider_gateway.config import GatewayConfig, load_gateway_config


def test_gateway_config_defaults_disabled() -> None:
    config = GatewayConfig.from_dict({})

    assert config.enabled is False
    assert config.backend == "native"
    assert config.track_usage is True
    assert config.track_cost is True
    assert config.routing_strategy == "round-robin"
    assert config.fallback_models == []


def test_gateway_config_parses_nested_routing() -> None:
    config = GatewayConfig.from_dict(
        {
            "enabled": True,
            "backend": "native",
            "track_usage": False,
            "track_cost": False,
            "routing": {
                "strategy": "lowest-cost",
                "fallback_models": ["anthropic/claude-sonnet-4.6", "ollama/llama3.2"],
            },
        }
    )

    assert config.enabled is True
    assert config.backend == "native"
    assert config.track_usage is False
    assert config.track_cost is False
    assert config.routing_strategy == "lowest-cost"
    assert config.fallback_models == [
        "anthropic/claude-sonnet-4.6",
        "ollama/llama3.2",
    ]


def test_load_gateway_config_from_root_config_dict() -> None:
    config = load_gateway_config(
        {
            "provider_gateway": {
                "enabled": True,
                "backend": "litellm",
                "routing": {"strategy": "lowest-latency"},
            }
        }
    )

    assert config.enabled is True
    assert config.backend == "litellm"
    assert config.routing_strategy == "lowest-latency"


def test_default_config_exposes_provider_gateway_default_off() -> None:
    provider_gateway = DEFAULT_CONFIG["provider_gateway"]

    assert provider_gateway["enabled"] is False
    assert provider_gateway["backend"] == "native"
    assert provider_gateway["track_usage"] is True
    assert provider_gateway["track_cost"] is True
    assert provider_gateway["routing"]["strategy"] == "round-robin"
    assert provider_gateway["routing"]["fallback_models"] == []


# --- New validation tests below ---


def test_invalid_backend_falls_back_to_native() -> None:
    """Unknown backend values should silently fall back to 'native'."""
    config = GatewayConfig.from_dict({"backend": "invalid_backend_xyz"})

    assert config.backend == "native"


def test_invalid_routing_strategy_falls_back_to_round_robin() -> None:
    """Unknown routing strategy should silently fall back to 'round-robin'."""
    config = GatewayConfig.from_dict(
        {"routing": {"strategy": "non_existent_strategy"}}
    )

    assert config.routing_strategy == "round-robin"


def test_none_input_produces_defaults() -> None:
    """Passing None to from_dict should produce default config."""
    config = GatewayConfig.from_dict(None)

    assert config.enabled is False
    assert config.backend == "native"
    assert config.routing_strategy == "round-robin"
    assert config.fallback_models == []


def test_non_mapping_routing_produces_defaults() -> None:
    """Non-dict routing value should not crash, producing default strategy."""
    config = GatewayConfig.from_dict({"routing": "not_a_dict"})

    assert config.routing_strategy == "round-robin"
    assert config.fallback_models == []


def test_non_list_fallback_models_produces_empty() -> None:
    """Non-list fallback_models should be safely ignored."""
    config = GatewayConfig.from_dict(
        {"routing": {"fallback_models": "not_a_list"}}
    )

    assert config.fallback_models == []


def test_fallback_models_strips_whitespace_and_filters_empty() -> None:
    """Whitespace-only and empty entries in fallback_models should be removed."""
    config = GatewayConfig.from_dict(
        {
            "routing": {
                "fallback_models": [
                    "  openai/gpt-4o  ",
                    "",
                    "   ",
                    "anthropic/claude-opus-4.6",
                    None,
                    42,
                ]
            }
        }
    )

    assert config.fallback_models == [
        "openai/gpt-4o",
        "anthropic/claude-opus-4.6",
    ]


def test_empty_string_backend_falls_back_to_native() -> None:
    """Empty string backend should default to 'native'."""
    config = GatewayConfig.from_dict({"backend": ""})

    assert config.backend == "native"


def test_load_gateway_config_without_root_config_returns_defaults() -> None:
    """load_gateway_config with empty root config should return defaults."""
    config = load_gateway_config({})

    assert config.enabled is False
    assert config.backend == "native"


def test_gateway_config_is_frozen() -> None:
    """GatewayConfig should be immutable (frozen dataclass)."""
    config = GatewayConfig()

    try:
        config.enabled = True  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass
