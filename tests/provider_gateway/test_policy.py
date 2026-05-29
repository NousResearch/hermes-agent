from __future__ import annotations

from types import SimpleNamespace

from agent.error_classifier import FailoverReason
from provider_gateway.config import GatewayConfig
from provider_gateway.policy import (
    ProviderGatewayPolicy,
    ProviderRouteCandidate,
    build_gateway_policy,
    should_consider_gateway_fallback,
)


def test_policy_keeps_disabled_gateway_as_primary_only() -> None:
    agent = SimpleNamespace(
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[
            {"provider": "openai", "model": "gpt-4o"},
        ],
    )

    policy = build_gateway_policy(agent, GatewayConfig(enabled=False))

    assert policy.enabled is False
    assert policy.routing_strategy == "round-robin"
    assert policy.candidates == (
        ProviderRouteCandidate(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            source="primary",
            base_url="https://openrouter.ai/api/v1",
        ),
    )


def test_policy_orders_primary_gateway_models_then_existing_fallback_chain() -> None:
    agent = SimpleNamespace(
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[
            {
                "provider": "openai",
                "model": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "key_env": "OPENAI_API_KEY",
            },
            {"provider": "zai", "model": "glm-4.7"},
        ],
    )
    config = GatewayConfig(
        enabled=True,
        routing_strategy="lowest-cost",
        fallback_models=["openai/gpt-5.4", "anthropic/claude-haiku-4.5"],
    )

    policy = build_gateway_policy(agent, config)

    assert policy.enabled is True
    assert policy.routing_strategy == "lowest-cost"
    assert policy.candidates == (
        ProviderRouteCandidate(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            source="primary",
            base_url="https://openrouter.ai/api/v1",
        ),
        ProviderRouteCandidate(
            provider="openrouter",
            model="openai/gpt-5.4",
            source="provider_gateway.routing.fallback_models",
            base_url="https://openrouter.ai/api/v1",
        ),
        ProviderRouteCandidate(
            provider="openrouter",
            model="anthropic/claude-haiku-4.5",
            source="provider_gateway.routing.fallback_models",
            base_url="https://openrouter.ai/api/v1",
        ),
        ProviderRouteCandidate(
            provider="openai",
            model="gpt-4o",
            source="fallback_chain",
            base_url="https://api.openai.com/v1",
            key_env="OPENAI_API_KEY",
        ),
        ProviderRouteCandidate(
            provider="zai",
            model="glm-4.7",
            source="fallback_chain",
        ),
    )


def test_policy_deduplicates_candidates_by_provider_model_and_base_url() -> None:
    agent = SimpleNamespace(
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1/",
        _fallback_chain=[
            {"provider": "openrouter", "model": "openai/gpt-5.4"},
            {"provider": "openrouter", "model": "openai/gpt-5.4"},
        ],
    )
    config = GatewayConfig(
        enabled=True,
        fallback_models=[
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5.4",
            "openai/gpt-5.4",
        ],
    )

    policy = build_gateway_policy(agent, config)

    assert [candidate.model for candidate in policy.candidates] == [
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
    ]


def test_policy_can_select_next_candidate_after_current_route() -> None:
    agent = SimpleNamespace(
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[{"provider": "openai", "model": "gpt-4o"}],
    )
    config = GatewayConfig(
        enabled=True,
        fallback_models=["openai/gpt-5.4"],
    )

    policy = build_gateway_policy(agent, config)

    assert policy.next_after("openrouter", "anthropic/claude-sonnet-4.6") == (
        ProviderRouteCandidate(
            provider="openrouter",
            model="openai/gpt-5.4",
            source="provider_gateway.routing.fallback_models",
            base_url="https://openrouter.ai/api/v1",
        )
    )
    assert policy.next_after("openrouter", "openai/gpt-5.4") == (
        ProviderRouteCandidate(
            provider="openai",
            model="gpt-4o",
            source="fallback_chain",
        )
    )
    assert policy.next_after("openai", "gpt-4o") is None


def test_reason_gate_matches_recoverable_provider_failures() -> None:
    assert should_consider_gateway_fallback(FailoverReason.rate_limit) is True
    assert should_consider_gateway_fallback(FailoverReason.billing) is True
    assert should_consider_gateway_fallback(FailoverReason.overloaded) is True
    assert should_consider_gateway_fallback(FailoverReason.model_not_found) is True
    assert should_consider_gateway_fallback(FailoverReason.content_policy_blocked) is False
    assert should_consider_gateway_fallback(FailoverReason.auth_permanent) is False


# --- New edge-case tests below ---


def test_reason_gate_accepts_none_reason() -> None:
    """None reason (unknown error) should allow fallback consideration."""
    assert should_consider_gateway_fallback(None) is True


def test_policy_with_empty_fallback_chain_and_no_gateway_models() -> None:
    """Enabled gateway with no fallback sources should only have primary."""
    agent = SimpleNamespace(
        provider="anthropic",
        model="claude-opus-4.6",
        base_url="https://api.anthropic.com/v1",
        _fallback_chain=[],
    )
    config = GatewayConfig(enabled=True, fallback_models=[])

    policy = build_gateway_policy(agent, config)

    assert len(policy.candidates) == 1
    assert policy.candidates[0].source == "primary"
    assert policy.next_after("anthropic", "claude-opus-4.6") is None


def test_policy_with_missing_fallback_chain_attribute() -> None:
    """Agent without _fallback_chain attribute should not crash."""
    agent = SimpleNamespace(
        provider="openai",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
    )
    config = GatewayConfig(
        enabled=True,
        fallback_models=["gpt-4o-mini"],
    )

    policy = build_gateway_policy(agent, config)

    assert len(policy.candidates) == 2
    assert policy.candidates[0].model == "gpt-4o"
    assert policy.candidates[1].model == "gpt-4o-mini"


def test_next_after_unknown_route_returns_first_candidate() -> None:
    """If the current route is not found in candidates, return the first one."""
    agent = SimpleNamespace(
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[],
    )
    config = GatewayConfig(
        enabled=True,
        fallback_models=["openai/gpt-5.4"],
    )

    policy = build_gateway_policy(agent, config)
    candidate = policy.next_after("totally_unknown", "nonexistent_model")

    assert candidate is not None
    assert candidate == policy.candidates[0]


def test_policy_ignores_non_dict_fallback_chain_entries() -> None:
    """Non-dict entries in _fallback_chain should be safely skipped."""
    agent = SimpleNamespace(
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[
            "not_a_dict",
            42,
            None,
            {"provider": "openai", "model": "gpt-4o"},
        ],
    )
    config = GatewayConfig(enabled=True)

    policy = build_gateway_policy(agent, config)

    assert len(policy.candidates) == 2
    assert policy.candidates[1].provider == "openai"
    assert policy.candidates[1].model == "gpt-4o"


def test_reason_gate_covers_all_expected_fallback_reasons() -> None:
    """All server/infra-related failure reasons should allow fallback."""
    fallback_reasons = [
        FailoverReason.billing,
        FailoverReason.rate_limit,
        FailoverReason.overloaded,
        FailoverReason.server_error,
        FailoverReason.timeout,
        FailoverReason.model_not_found,
        FailoverReason.provider_policy_blocked,
        FailoverReason.unknown,
    ]

    for reason in fallback_reasons:
        assert should_consider_gateway_fallback(reason) is True, (
            f"Expected {reason} to allow fallback"
        )
