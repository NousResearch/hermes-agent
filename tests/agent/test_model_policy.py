from copy import deepcopy

import pytest

from agent.model_policy import parse_model_variant, resolve_model_policy


@pytest.fixture
def route_policies():
    return {
        "coding": {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4.6",
            "fallback_models": [
                "openai/gpt-5.4-mini",
                {"provider": "openrouter", "model": "qwen/qwen3.6-plus"},
            ],
            "providerOptions": {
                "temperature": 0.1,
                "Authorization": "Bearer should-not-leak",
            },
        }
    }


@pytest.fixture
def named_agent_config():
    return {
        "name": "builder",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6(effort=medium)",
        "fallback_models": ["claude-haiku-4.5", "openai/gpt-5.4-mini"],
        "providerOptions": {
            "temperature": 0.2,
            "api_key": "secret-api-key",
        },
        "ultrawork": {
            "provider": "openrouter",
            "model": "openai/gpt-5.4",
            "fallback_models": ["anthropic/claude-sonnet-4.6", "gpt-5.4-mini"],
            "providerOptions": {
                "service_tier": "priority",
                "token": "secret-token",
            },
        },
    }


@pytest.fixture
def system_defaults():
    return {
        "provider": "nous",
        "model": "moonshotai/kimi-k2.6",
        "fallback_models": [
            "openrouter:openai/gpt-5.4-mini",
            {"provider": "anthropic", "model": "claude-haiku-4.5"},
        ],
        "providerOptions": {
            "api_key": "system-secret",
            "region": "global",
        },
        "provider_chain": ["openrouter", "anthropic"],
    }


def _strip_source(chain):
    return [{k: v for k, v in entry.items() if k in {"provider", "model"}} for entry in chain]


def test_explicit_user_override_wins_over_all_other_sources(route_policies, named_agent_config, system_defaults):
    policy = resolve_model_policy(
        model="gpt-5.4-nano(reasoning=high)",
        provider="openai",
        fallback_models=["gpt-5.4-mini", "anthropic:claude-haiku-4.5"],
        named_agent=named_agent_config,
        route_category="coding",
        route_policies=route_policies,
        runtime_mode="ultrawork",
        defaults=system_defaults,
    )

    assert policy.primary_provider == "openai"
    assert policy.primary_model == "gpt-5.4-nano"
    assert policy.variant == "reasoning=high"
    assert policy.trace[0]["stage"] == "selection"
    assert policy.trace[0]["source"] == "explicit_override"


def test_ultrawork_activates_named_agent_override_without_mutation(named_agent_config, route_policies, system_defaults):
    original = deepcopy(named_agent_config)

    policy = resolve_model_policy(
        named_agent=named_agent_config,
        runtime_mode="ultrawork",
        route_category="coding",
        route_policies=route_policies,
        defaults=system_defaults,
    )

    assert policy.primary_provider == "openrouter"
    assert policy.primary_model == "openai/gpt-5.4"
    assert policy.variant is None
    assert policy.provider_options == {"service_tier": "priority"}
    assert named_agent_config == original


def test_normal_mode_keeps_named_agent_base_model(named_agent_config, route_policies, system_defaults):
    policy = resolve_model_policy(
        named_agent=named_agent_config,
        runtime_mode="default",
        route_category="coding",
        route_policies=route_policies,
        defaults=system_defaults,
    )

    assert policy.primary_provider == "anthropic"
    assert policy.primary_model == "claude-sonnet-4-6"
    assert policy.variant == "effort=medium"
    assert policy.provider_options == {"temperature": 0.2}


def test_disabled_named_agent_policy_falls_through_to_route_category(route_policies, system_defaults):
    policy = resolve_model_policy(
        named_agent={
            "name": "oracle",
            "mode": "disabled",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        },
        route_category="coding",
        route_policies=route_policies,
        defaults=system_defaults,
    )

    assert policy.primary_provider == "openrouter"
    assert policy.primary_model == "anthropic/claude-sonnet-4.6"
    assert policy.trace[0]["source"] == "route_category"


def test_unsupported_named_agent_runtime_mode_falls_through_to_defaults(system_defaults):
    policy = resolve_model_policy(
        named_agent={
            "name": "sisyphus",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "supported_runtime_modes": ["default"],
        },
        runtime_mode="ultrawork",
        defaults=system_defaults,
    )

    assert policy.primary_provider == "nous"
    assert policy.primary_model == "moonshotai/kimi-k2.6"
    assert policy.trace[0]["source"] == "system_default"


def test_route_category_default_supplies_model_when_no_explicit_or_named_agent(route_policies, system_defaults):
    policy = resolve_model_policy(
        route_category="coding",
        route_policies=route_policies,
        defaults=system_defaults,
    )

    assert policy.primary_provider == "openrouter"
    assert policy.primary_model == "anthropic/claude-sonnet-4.6"
    assert policy.variant is None
    assert policy.provider_options == {"temperature": 0.1}
    assert policy.trace[0]["source"] == "route_category"


def test_fallback_chain_combines_all_sources_without_duplicates(route_policies, named_agent_config, system_defaults):
    policy = resolve_model_policy(
        fallback_models=["gpt-5.4-mini", "anthropic:claude-haiku-4.5", "gpt-5.4-mini"],
        named_agent=named_agent_config,
        route_category="coding",
        route_policies=route_policies,
        runtime_mode="ultrawork",
        defaults=system_defaults,
    )

    assert _strip_source(policy.fallback_chain) == [
        {"provider": "openrouter", "model": "gpt-5.4-mini"},
        {"provider": "anthropic", "model": "claude-haiku-4.5"},
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
        {"provider": "openrouter", "model": "qwen/qwen3.6-plus"},
        {"provider": "nous", "model": "moonshotai/kimi-k2.6"},
        {"provider": "anthropic", "model": "openai/gpt-5.4"},
    ]
    assert [entry["source"] for entry in policy.fallback_chain[:3]] == [
        "user_fallback",
        "user_fallback",
        "named_agent",
    ]


def test_same_provider_shorthand_fallback_entries_use_active_provider(system_defaults):
    policy = resolve_model_policy(
        provider="zai",
        model="glm-5.1",
        fallback_models=["glm-5-turbo", "openrouter:openai/gpt-5.4-mini"],
        defaults=system_defaults,
    )

    assert _strip_source(policy.fallback_chain[:2]) == [
        {"provider": "zai", "model": "glm-5-turbo"},
        {"provider": "openrouter", "model": "openai/gpt-5.4-mini"},
    ]


def test_provider_chain_supplements_declared_fallbacks_without_replacing_them():
    policy = resolve_model_policy(
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        fallback_models=["openai/gpt-5.4-mini"],
        provider_chain=["openrouter", "anthropic"],
    )

    assert _strip_source(policy.fallback_chain) == [
        {"provider": "openrouter", "model": "openai/gpt-5.4-mini"},
        {"provider": "anthropic", "model": "anthropic/claude-sonnet-4.6"},
    ]
    assert policy.fallback_chain[1]["source"] == "provider_chain"


def test_accepts_wave2_fallback_models_and_provider_options_aliases(system_defaults):
    policy = resolve_model_policy(
        named_agent={
            "name": "sisyphus",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "fallback_models": ["claude-haiku-4.5"],
            "providerOptions": {"temperature": 0},
        },
        defaults=system_defaults,
    )

    assert policy.provider_options == {"temperature": 0}
    assert policy.fallback_chain[0] == {
        "provider": "anthropic",
        "model": "claude-haiku-4.5",
        "source": "named_agent",
    }


@pytest.mark.parametrize(
    ("raw", "expected_model", "expected_variant"),
    [
        ("gpt-5.4(reasoning=high)", "gpt-5.4", "reasoning=high"),
        ("claude-sonnet-4-6(effort=medium)", "claude-sonnet-4-6", "effort=medium"),
        ("anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet-4.6", None),
        ("minimax/minimax-m2.5:free", "minimax/minimax-m2.5:free", None),
        (None, None, None),
    ],
)
def test_parse_model_variant_parses_inline_variant_once(raw, expected_model, expected_variant):
    assert parse_model_variant(raw) == (expected_model, expected_variant)


def test_trace_and_provider_options_are_safely_redacted(route_policies, named_agent_config, system_defaults):
    policy = resolve_model_policy(
        named_agent=named_agent_config,
        runtime_mode="ultrawork",
        route_category="coding",
        route_policies=route_policies,
        defaults=system_defaults,
        providerOptions={"api_key": "explicit-secret", "safe": "ok"},
    )

    rendered_trace = repr(policy.trace)
    rendered_options = repr(policy.provider_options)
    for forbidden in [
        "secret-api-key",
        "secret-token",
        "system-secret",
        "should-not-leak",
        "Bearer",
        "explicit-secret",
    ]:
        assert forbidden not in rendered_trace
        assert forbidden not in rendered_options
    assert policy.provider_options == {"safe": "ok"}
