"""Per-primary fallback chains (#110822).

``fallback_providers`` is one global ordered chain, so a session that switched its primary model
(``/model``, aliases, channel/profile overrides, delegation) still walked the same chain. The tests
below cover ``fallback_routes``: a route matching the effective primary wins, no match keeps the
global chain byte-for-byte, and the existing same-backend skip / retry semantics are untouched.
"""

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.fallback_config import (
    get_fallback_chain,
    get_fallback_routes,
    match_fallback_route,
)
from agent.error_classifier import FailoverReason
from run_agent import AIAgent

GLOBAL_CHAIN = [{"provider": "zai", "model": "glm-5.2"}]

ROUTES = [
    {
        "when": {"provider": "openrouter", "model": "vendor/model-a"},
        "fallback_providers": [
            {"provider": "openai-codex", "model": "model-b"},
            {"provider": "xai-oauth", "model": "model-c"},
        ],
    },
    {
        "when": {"provider": "openai-codex", "model": "model-b"},
        "fallback_providers": [{"provider": "xai-oauth", "model": "model-c"}],
    },
    {"when": {"provider": "ollama"}, "fallback_providers": []},
]


# ── Resolver ──────────────────────────────────────────────────────────────


def test_matching_route_wins_in_declared_order():
    config = {"fallback_providers": GLOBAL_CHAIN, "fallback_routes": ROUTES}

    chain = match_fallback_route(config, "openrouter", "vendor/model-a")

    assert [entry["provider"] for entry in chain] == ["openai-codex", "xai-oauth"]


def test_one_provider_two_models_route_differently():
    config = {"fallback_routes": ROUTES}

    assert [e["provider"] for e in match_fallback_route(config, "openrouter", "vendor/model-a")] == [
        "openai-codex", "xai-oauth",
    ]
    assert [e["provider"] for e in match_fallback_route(config, "openai-codex", "model-b")] == [
        "xai-oauth",
    ]


def test_unmatched_primary_has_no_route():
    config = {"fallback_providers": GLOBAL_CHAIN, "fallback_routes": ROUTES}

    # Unknown model on a routed provider, and an unrouted provider, both defer to the global chain.
    assert match_fallback_route(config, "openai-codex", "model-z") is None
    assert match_fallback_route(config, "deepseek", "deepseek-v4-flash") is None
    assert get_fallback_chain(config) == GLOBAL_CHAIN


def test_provider_only_route_matches_every_model():
    config = {"fallback_routes": ROUTES}

    assert match_fallback_route(config, "ollama", "qwen3:32b") == []
    assert match_fallback_route(config, "ollama", "") == []


def test_matching_is_case_insensitive():
    config = {"fallback_routes": ROUTES}

    assert match_fallback_route(config, "OpenRouter", "Vendor/Model-A") is not None


@pytest.mark.parametrize(
    "declared",
    [
        "not-a-list",
        [],
        [{"when": {"model": "vendor/model-a"}, "fallback_providers": [{"provider": "zai", "model": "glm-5.2"}]}],
        [{"when": {"provider": "openrouter"}, "fallback_providers": "openai-codex"}],
        [{"when": {"provider": "openrouter"}}],
        # Non-empty list, but every entry is unusable → invalid value, not "disable fallback".
        [{"when": {"provider": "openrouter"}, "fallback_providers": [{"provider": "zai"}]}],
        [{"when": {"provider": "openrouter"}, "fallback_providers": ["zai"]}],
    ],
)
def test_invalid_routes_degrade_to_global_chain(declared):
    config = {"fallback_providers": GLOBAL_CHAIN, "fallback_routes": declared}

    assert get_fallback_routes(config) == []
    assert match_fallback_route(config, "openrouter", "vendor/model-a") is None


# ── Agent fallback walk ───────────────────────────────────────────────────


def _make_agent(fallback_model=None):
    """Minimal AIAgent with an installed (global) fallback chain."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _agent_with_primary(provider, model, fallback_model=None):
    agent = _make_agent(fallback_model=fallback_model)
    agent.provider, agent.model = provider, model
    agent.requested_provider = provider
    agent._rate_limited_until = 0
    return agent


def _mock_client(base_url="https://api.example.com/v1"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = "fb-key"
    return mock


def _config_with_routes(*extra_routes, global_chain=GLOBAL_CHAIN):
    return {"fallback_providers": list(global_chain), "fallback_routes": list(extra_routes)}


def test_matching_route_replaces_global_chain_for_that_primary():
    agent = _agent_with_primary("openrouter", "vendor/model-a", GLOBAL_CHAIN)
    config = _config_with_routes({
        "when": {"provider": "openrouter", "model": "vendor/model-a"},
        "fallback_providers": [{"provider": "xai-oauth", "model": "model-c"}],
    })

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(_mock_client("https://api.x.ai/v1"), "model-c")) as resolver,
    ):
        assert agent._try_activate_fallback(FailoverReason.rate_limit) is True

    assert (agent.provider, agent.model) == ("xai-oauth", "model-c")
    assert resolver.call_args.args[0] == "xai-oauth"


def test_unmatched_primary_keeps_global_chain():
    agent = _agent_with_primary("deepseek", "deepseek-v4-flash", GLOBAL_CHAIN)
    config = _config_with_routes({
        "when": {"provider": "openrouter", "model": "vendor/model-a"},
        "fallback_providers": [{"provider": "xai-oauth", "model": "model-c"}],
    })

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(_mock_client("https://api.z.ai/v1"), "glm-5.2")),
    ):
        assert agent._try_activate_fallback(FailoverReason.rate_limit) is True

    assert (agent.provider, agent.model) == ("zai", "glm-5.2")


def test_empty_route_disables_fallback_for_that_primary():
    agent = _agent_with_primary("custom", "llama-3.1-70b", GLOBAL_CHAIN)
    config = _config_with_routes(
        {"when": {"provider": "custom"}, "fallback_providers": []},
        global_chain=GLOBAL_CHAIN,
    )

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client.resolve_provider_client") as resolver,
    ):
        assert agent._has_pending_fallback() is False
        assert agent._try_activate_fallback(FailoverReason.overloaded) is False

    resolver.assert_not_called()
    assert (agent.provider, agent.model) == ("custom", "llama-3.1-70b")
    # An opted-out route must not arm the chain-exhausted cooldown either.
    assert agent._rate_limited_until == 0


def test_route_chain_beyond_global_chain_is_walked():
    agent = _agent_with_primary("openrouter", "vendor/model-a", None)
    config = {
        "fallback_routes": [{
            "when": {"provider": "openrouter", "model": "vendor/model-a"},
            "fallback_providers": [
                {"provider": "openai-codex", "model": "model-b"},
                {"provider": "xai-oauth", "model": "model-c"},
            ],
        }],
    }

    clients = [_mock_client("https://chatgpt.com/backend-api/codex"), _mock_client("https://api.x.ai/v1")]
    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=[
            (clients[0], "model-b"), (clients[1], "model-c"),
        ]),
    ):
        assert agent._has_pending_fallback() is True
        assert agent._try_activate_fallback(FailoverReason.rate_limit) is True
        assert (agent.provider, agent.model) == ("openai-codex", "model-b")
        # Second rung of the route chain, even though the installed chain is empty.
        assert agent._has_pending_fallback() is True
        assert agent._try_activate_fallback(FailoverReason.overloaded) is True

    assert (agent.provider, agent.model) == ("xai-oauth", "model-c")


def test_same_backend_skip_still_applies_within_a_route():
    agent = _agent_with_primary("openrouter", "vendor/model-a", GLOBAL_CHAIN)
    config = _config_with_routes({
        "when": {"provider": "openrouter", "model": "vendor/model-a"},
        # Same provider + model as the primary: falling back to it would loop the failure.
        "fallback_providers": [{"provider": "openrouter", "model": "vendor/model-a"}],
    })

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client.resolve_provider_client") as resolver,
    ):
        assert agent._try_activate_fallback(FailoverReason.rate_limit) is False

    resolver.assert_not_called()
    assert (agent.provider, agent.model) == ("openrouter", "vendor/model-a")


def test_walk_does_not_reroute_mid_chain():
    """A route is one chain: the fallback's own route does not hijack an in-flight walk."""
    agent = _agent_with_primary("openrouter", "vendor/model-a", None)
    config = {
        "fallback_routes": [
            {"when": {"provider": "openrouter", "model": "vendor/model-a"},
             "fallback_providers": [{"provider": "openai-codex", "model": "model-b"}]},
            {"when": {"provider": "openai-codex", "model": "model-b"},
             "fallback_providers": [{"provider": "xai-oauth", "model": "model-c"}]},
        ],
    }

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(_mock_client("https://chatgpt.com/backend-api/codex"), "model-b")),
    ):
        assert agent._try_activate_fallback(FailoverReason.rate_limit) is True
        assert agent._try_activate_fallback(FailoverReason.overloaded) is False

    assert (agent.provider, agent.model) == ("openai-codex", "model-b")


@pytest.mark.parametrize(
    ("provider", "model", "expected_provider"),
    [
        ("openrouter", "vendor/model-a", "openai-codex"),
        ("openai-codex", "model-b", "xai-oauth"),
    ],
)
def test_two_primaries_take_different_chains(provider, model, expected_provider):
    agent = _agent_with_primary(provider, model, GLOBAL_CHAIN)
    config = _config_with_routes(*ROUTES)
    routed = match_fallback_route(config, provider, model)
    resolved = _mock_client("https://api.example.com/v1")

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(resolved, routed[0]["model"])),
    ):
        assert agent._try_activate_fallback(FailoverReason.rate_limit) is True

    assert agent.provider == expected_provider
    assert agent.provider != "zai", "global chain must not be used when a route matches"


def test_delegated_child_uses_its_own_primary_route():
    """A child inherits the parent chain, but its own primary's route wins."""
    from tools.delegate_tool_config import _resolve_child_fallback_chain

    parent = _agent_with_primary("openrouter", "vendor/model-a", GLOBAL_CHAIN)
    inherited = _resolve_child_fallback_chain(parent, None, pinned=False)
    assert inherited == GLOBAL_CHAIN

    child = _agent_with_primary("openai-codex", "model-b", inherited)
    config = _config_with_routes(*ROUTES)

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(_mock_client("https://api.x.ai/v1"), "model-c")),
    ):
        assert child._try_activate_fallback(FailoverReason.rate_limit) is True

    assert (child.provider, child.model) == ("xai-oauth", "model-c")


def test_config_without_routes_is_unchanged():
    agent = _agent_with_primary("openrouter", "vendor/model-a", GLOBAL_CHAIN)

    with (
        patch("hermes_cli.config.load_config_readonly", return_value={"fallback_providers": GLOBAL_CHAIN}),
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(_mock_client("https://api.z.ai/v1"), "glm-5.2")),
    ):
        assert agent._has_pending_fallback() is True
        assert agent._try_activate_fallback(FailoverReason.rate_limit) is True

    assert (agent.provider, agent.model) == ("zai", "glm-5.2")
