"""Tests that switch_model does not inherit stale context_length overrides."""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.agent_init import _normalize_route_base_url
from agent.context_compressor import ContextCompressor


class _StubStartupCompressor:
    def __init__(self, *args, **kwargs):
        self.context_length = kwargs.get("config_context_length") or 272_000
        self.config_context_length = kwargs.get("config_context_length")
        self.threshold_tokens = int(self.context_length * 0.95)
        self.threshold_percent = 0.95

    def get_tool_schemas(self):
        return []

    def on_session_start(self, *args, **kwargs):
        return None


def test_route_url_normalization_preserves_path_slash_before_query():
    """A path slash before a query changes OpenAI SDK URL joining."""
    assert _normalize_route_base_url(
        "https://example.com/v1/?tenant=large"
    ) != _normalize_route_base_url("https://example.com/v1?tenant=large")


def test_route_url_normalization_preserves_trailing_whitespace():
    """Whitespace can alter the request target and must not collapse routes."""
    assert _normalize_route_base_url(
        "https://example.com/v1 "
    ) != _normalize_route_base_url("https://example.com/v1")


def test_route_url_normalization_preserves_bracketed_host_syntax():
    """Invalid bracketed host syntax must not collapse onto a valid DNS host."""
    assert _normalize_route_base_url(
        "http://[v1.Foo]/v1"
    ) != _normalize_route_base_url("http://v1.foo/v1")


def test_route_url_normalization_preserves_malformed_trailing_slash():
    """Malformed URLs are kept byte-exact rather than partially normalized."""
    assert _normalize_route_base_url(
        "http://[bad/v1/"
    ) != _normalize_route_base_url("http://[bad/v1")


@pytest.mark.parametrize(
    "raw",
    ["http://[bad/v1/", "example.com/v1/", "https:///v1/"],
)
def test_route_url_normalization_keeps_unparseable_routes_byte_exact(raw):
    assert _normalize_route_base_url(raw) == raw


@pytest.mark.parametrize(
    ("configured", "active"),
    [
        ("http://EXAMPLE.COM:80/v1/", "http://example.com/v1"),
        (
            "https://EXAMPLE.COM:443/v1/#configured-fragment",
            "https://example.com/v1#active-fragment",
        ),
        ("http://[2001:DB8::1]:80/v1/", "http://[2001:db8::1]/v1"),
    ],
)
def test_route_url_normalization_accepts_isolated_safe_equivalences(
    configured, active
):
    """Default ports, fragments, and IPv6 hex case do not change HTTP routes."""
    assert _normalize_route_base_url(configured) == _normalize_route_base_url(active)


@pytest.mark.parametrize(
    ("configured", "active"),
    [
        ("https://example.com/V1", "https://example.com/v1"),
        ("https://example.com:8443/v1", "https://example.com/v1"),
        ("https://example.com/v1?tenant=large", "https://example.com/v1"),
        ("http://example.com/v1", "https://example.com/v1"),
        ("https://example.com:notaport/v1", "https://example.com/v1"),
    ],
)
def test_route_url_normalization_preserves_significant_components(
    configured, active
):
    """Path case, route data, schemes, and ambiguous ports stay distinct."""
    assert _normalize_route_base_url(configured) != _normalize_route_base_url(active)


def _make_direct_start_agent(
    cfg: dict, *, model: str, provider: str, base_url: str
) -> AIAgent:
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.agent_init.ContextCompressor", new=_StubStartupCompressor),
    ):
        return AIAgent(
            model=model,
            provider=provider,
            api_key="fake-test-token",
            base_url=base_url,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def _make_agent_with_compressor(config_context_length=None) -> AIAgent:
    """Build a minimal AIAgent with a context_compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)

    # Primary model settings
    agent.model = "primary-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True

    # Store the initial config_context_length override used at agent construction.
    agent._config_context_length = config_context_length

    # Context compressor with primary model values
    compressor = ContextCompressor(
        model="primary-model",
        threshold_percent=0.50,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-primary",
        provider="openrouter",
        quiet_mode=True,
        config_context_length=config_context_length,
    )
    agent.context_compressor = compressor

    # For switch_model
    agent._primary_runtime = {}

    return agent


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
def test_switch_model_clears_previous_config_context_length(mock_ctx_len):
    """Switching models must not reuse the previous model.context_length override."""
    agent = _make_agent_with_compressor(config_context_length=32_768)

    assert agent.context_compressor.model == "primary-model"
    assert agent.context_compressor.context_length == 32_768  # From config override

    # Switch model
    agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

    # Verify the old config override is not passed to the new model.
    mock_ctx_len.assert_called_once()
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") is None

    # Verify compressor was updated from the newly resolved model metadata.
    assert agent.context_compressor.model == "new-model"
    assert agent.context_compressor.context_length == 131_072


def test_switch_model_without_config_context_length():
    """When switching models without config override, config_context_length should be None."""
    agent = _make_agent_with_compressor(config_context_length=None)

    with patch("agent.model_metadata.get_model_context_length", return_value=128_000) as mock_ctx_len:
        # Switch model
        agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

        # Verify get_model_context_length was called with None
        mock_ctx_len.assert_called_once()
        call_kwargs = mock_ctx_len.call_args.kwargs
        assert call_kwargs.get("config_context_length") is None


def test_direct_start_model_override_does_not_inherit_profile_context_length():
    """A CLI ``--model`` startup override must not inherit another model's window."""
    cfg = {
        "model": {
            "default": "kimi-k3",
            "provider": "custom:kimi-coding-1m",
            "base_url": "https://api.kimi.com/coding",
            "context_length": 1_048_576,
        },
        "custom_providers": [
            {
                "name": "kimi-coding-1m",
                "base_url": "https://api.kimi.com/coding",
                "models": {"kimi-k3": {"context_length": 1_048_576}},
            }
        ],
    }
    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert agent.context_compressor.config_context_length is None
    assert agent.context_compressor.context_length == 272_000


def test_direct_start_preserves_context_for_normalized_default_model_alias():
    """Equivalent vendor-prefixed defaults still own their explicit window."""
    cfg = {
        "model": {
            "default": "openai/gpt-5.6-sol",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "context_length": 272_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert agent.context_compressor.config_context_length == 272_000
    assert agent.context_compressor.context_length == 272_000


def test_direct_start_same_model_on_different_route_drops_context_override():
    """Context pins are route-specific even when the model slug is unchanged."""
    cfg = {
        "model": {
            "default": "gpt-5.6-sol",
            "provider": "custom:large-sol-route",
            "base_url": "https://large-sol.example/v1",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert agent.context_compressor.config_context_length is None
    assert agent.context_compressor.context_length == 272_000


def test_direct_start_drops_context_when_configured_route_has_no_active_url():
    """A configured endpoint cannot own a runtime whose endpoint is unknown."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": "https://large.example/v1",
            "context_length": 1_048_576,
        }
    }
    routed_client = MagicMock(api_key="fake-test-token", base_url="")

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(routed_client, "shared-model"),
    ):
        agent = _make_direct_start_agent(
            cfg,
            model="shared-model",
            provider="custom",
            base_url="",
        )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_preserves_context_for_bare_aggregator_model():
    """Aggregator normalization must compare both sides, not rewrite one side."""
    cfg = {
        "model": {
            "default": "gpt-5.4",
            "provider": "openrouter",
            "context_length": 1_000_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.4",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )

    assert agent.context_compressor.config_context_length == 1_000_000


def test_direct_start_drops_context_for_same_provider_custom_base_url():
    """An explicit endpoint override changes the route even if provider matches."""
    cfg = {
        "model": {
            "default": "gpt-5.4",
            "provider": "openrouter",
            "context_length": 1_000_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.4",
        provider="openrouter",
        base_url="https://small.example/v1",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_drops_context_for_provider_name_lookalike_host():
    """A hostname containing a provider domain is not that provider's route."""
    cfg = {
        "model": {
            "default": "gpt-5.4",
            "provider": "openrouter",
            "context_length": 1_000_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.4",
        provider="openrouter",
        base_url="https://evil-openrouter.ai/v1",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_preserves_context_for_codex_default_endpoint():
    """ChatGPT's Codex endpoint belongs to the openai-codex route."""
    cfg = {
        "model": {
            "default": "gpt-5.6-sol",
            "provider": "openai-codex",
            "context_length": 272_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert agent.context_compressor.config_context_length == 272_000


def test_direct_start_drops_context_for_codex_wrong_path():
    """A known host with a different route path is not the Codex endpoint."""
    cfg = {
        "model": {
            "default": "gpt-5.6-sol",
            "provider": "openai-codex",
            "context_length": 272_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/unrelated",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_drops_context_for_overridden_provider_wrong_path():
    """Providers with an explicit default route require that complete route."""
    cfg = {
        "model": {
            "default": "grok-4",
            "provider": "xai",
            "context_length": 256_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="grok-4",
        provider="xai",
        base_url="https://api.x.ai/not-v1",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_preserves_context_for_equivalent_base_url_spellings():
    """Route identity ignores URL casing, default ports, and trailing slashes."""
    cfg = {
        "model": {
            "default": "gpt-5.4",
            "provider": "openrouter",
            "base_url": "HTTPS://OPENROUTER.AI:443/api/v1/",
            "context_length": 1_000_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.4",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )

    assert agent.context_compressor.config_context_length == 1_000_000


def test_direct_start_drops_context_when_path_parameter_segment_changes():
    """Trailing-slash normalization must not move params to another segment."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": "https://example.com/v1/;tenant=large",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://example.com/v1;tenant=large",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_drops_context_when_empty_path_parameter_changes():
    """An explicit empty path-parameter delimiter is not discarded."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": "https://example.com/v1;",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://example.com/v1",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_drops_context_when_empty_query_delimiter_changes():
    """An explicit empty query changes OpenAI SDK base-URL joining semantics."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": "https://example.com/v1?",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://example.com/v1",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_drops_context_when_query_path_slash_changes():
    """A path slash before a query remains part of the effective SDK route."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": "https://example.com/v1/?tenant=large",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://example.com/v1?tenant=large",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_drops_context_when_empty_query_path_slash_changes():
    """An empty query still preserves the path slash immediately before it."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": "https://example.com/v1/?",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://example.com/v1?",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_drops_context_when_active_query_changes():
    """Query parameters remain part of the effective route identity."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": "https://example.com/v1",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://example.com/v1?tenant=small",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_preserves_context_for_matching_query_route():
    """SDK query extraction must not hide an otherwise matching route."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": "https://example.com/v1?tenant=large",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://example.com/v1?tenant=large",
    )

    assert agent.context_compressor.config_context_length == 1_048_576


def test_direct_start_drops_context_when_extra_trailing_segment_changes():
    """Only one conventional trailing slash is ignored for route identity."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": "https://example.com/v1//",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://example.com/v1",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_does_not_reapply_custom_context_across_extra_slash():
    """Per-model custom overrides use the same fail-closed route identity."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
        },
        "custom_providers": [
            {
                "name": "large-route",
                "base_url": "https://example.com/v1//",
                "models": {
                    "shared-model": {"context_length": 1_048_576}
                },
            }
        ],
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://example.com/v1",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_drops_context_when_url_userinfo_changes():
    """Credentials embedded in a URL remain part of route identity."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": "https://large-tenant:secret@example.com/v1",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://small-tenant:secret@example.com/v1",
    )

    assert agent.context_compressor.config_context_length is None


@pytest.mark.parametrize("suffix", [" ", "\t", "\n", "\r", "%20"])
def test_direct_start_drops_context_for_trailing_url_data(suffix):
    """Whitespace, controls, and encoded spaces remain route-significant."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": f"https://example.com/v1{suffix}",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://example.com/v1",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_drops_context_when_ipv6_zone_case_changes():
    """IPv6 address hex is case-insensitive, but its zone identifier is not."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "base_url": "http://[FE80::1%25ETH0]/v1",
            "context_length": 1_048_576,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="http://[fe80::1%25eth0]/v1",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_preserves_context_for_provider_alias():
    """Canonical provider aliases identify the same route when no URL is pinned."""
    cfg = {
        "model": {
            "default": "gemini-2.5-pro",
            "provider": "google",
            "context_length": 1_000_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gemini-2.5-pro",
        provider="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )

    assert agent.context_compressor.config_context_length == 1_000_000


def test_direct_start_preserves_context_for_registry_provider_alias():
    """Legacy and models.dev provider IDs may identify the same route."""
    cfg = {
        "model": {
            "default": "kimi-k3",
            "provider": "kimi-for-coding",
            "context_length": 1_048_576,
        }
    }

    routed_client = MagicMock(api_key="fake-test-token", base_url="")
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(routed_client, "kimi-k3"),
    ):
        agent = _make_direct_start_agent(
            cfg,
            model="kimi-k3",
            provider="kimi-coding",
            base_url="",
        )

    assert agent.context_compressor.config_context_length == 1_048_576


def test_direct_start_preserves_context_for_profile_route_on_shared_host():
    """Exact provider-profile routes disambiguate providers sharing a hostname."""
    cfg = {
        "model": {
            "default": "gpt-5.4",
            "provider": "opencode-zen",
            "context_length": 1_000_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.4",
        provider="opencode",
        base_url="https://opencode.ai/zen/v1",
    )

    assert agent.context_compressor.config_context_length == 1_000_000


def test_direct_start_drops_context_for_profile_wrong_path():
    """A shared hostname cannot substitute for a profile's complete route."""
    cfg = {
        "model": {
            "default": "gpt-5.4",
            "provider": "opencode-go",
            "context_length": 1_000_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.4",
        provider="opencode-go",
        base_url="https://opencode.ai/unrelated",
    )

    assert agent.context_compressor.config_context_length is None


def test_direct_start_named_custom_route_resolves_configured_base_url():
    """Named custom providers must not collapse to one generic custom route."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom:large-route",
            "context_length": 1_048_576,
        },
        "custom_providers": [
            {
                "name": "Large Route",
                "base_url": "https://legacy-large.example/v1",
            }
        ],
        "providers": {
            "large-route": {
                "name": "Large Route",
                "api": "https://large.example/v1",
            }
        },
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://small.example/v1",
    )

    assert agent.context_compressor.config_context_length is None
    assert agent.context_compressor.context_length == 272_000

    matching_agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="HTTPS://LARGE.EXAMPLE:443/v1/",
    )

    assert matching_agent.context_compressor.config_context_length == 1_048_576

    legacy_agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://legacy-large.example/v1",
    )

    assert legacy_agent.context_compressor.config_context_length is None


def test_direct_start_named_custom_provider_key_uses_canonical_slug():
    """Raw, canonical, and prefixed provider keys/names share runtime identity."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom:Route Key",
            "context_length": 1_048_576,
        },
        "providers": {
            "Route Key": {
                "name": "Friendly Label",
                "api": "https://key.example/v1",
            },
            "custom:Prefixed Key": {
                "name": "custom:Prefixed Label",
                "api": "https://prefixed.example/v1",
            },
        },
    }

    for configured_provider in (
        "custom:Route Key",
        "custom:Friendly Label",
        "Route Key",
        "route-key",
        "Friendly Label",
        "friendly-label",
    ):
        cfg["model"]["provider"] = configured_provider
        agent = _make_direct_start_agent(
            cfg,
            model="shared-model",
            provider="custom",
            base_url="https://key.example/v1",
        )

        assert agent.context_compressor.config_context_length == 1_048_576

    for configured_provider in (
        "custom:Prefixed Key",
        "custom:Prefixed Label",
        "custom:custom:Prefixed Key",
        "custom:custom:Prefixed Label",
    ):
        cfg["model"]["provider"] = configured_provider
        agent = _make_direct_start_agent(
            cfg,
            model="shared-model",
            provider="custom",
            base_url="https://prefixed.example/v1",
        )

        assert agent.context_compressor.config_context_length == 1_048_576

    for configured_provider in (
        "custom: Prefixed Key",
        "custom:\tPrefixed Key",
    ):
        cfg["model"]["provider"] = configured_provider
        agent = _make_direct_start_agent(
            cfg,
            model="shared-model",
            provider="custom",
            base_url="https://prefixed.example/v1",
        )

        assert agent.context_compressor.config_context_length is None


def test_direct_start_named_custom_raw_legacy_display_name_matches():
    """Legacy display names accepted by runtime also identify the scoped route."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "Legacy Route",
            "context_length": 1_048_576,
        },
        "custom_providers": [
            {
                "name": "Legacy Route",
                "base_url": "https://legacy.example/v1",
            }
        ],
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://legacy.example/v1",
    )

    assert agent.context_compressor.config_context_length == 1_048_576


def test_direct_start_literal_bare_custom_entry_matches_runtime():
    """A providers.custom entry makes bare custom a complete route identity."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom",
            "context_length": 1_048_576,
        },
        "providers": {
            "custom": {
                "api": "https://literal.example/v1",
            }
        },
    }

    agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://literal.example/v1",
    )

    assert agent.context_compressor.config_context_length == 1_048_576


def test_direct_start_disabled_modern_custom_falls_back_only_to_legacy():
    """Disabled modern entries cannot retain pins, but legacy fallback can."""
    cfg = {
        "model": {
            "default": "shared-model",
            "provider": "custom:route-key",
            "context_length": 1_048_576,
        },
        "providers": {
            "route-key": {
                "name": "Route Key",
                "api": "https://disabled.example/v1",
                "enabled": False,
            }
        },
    }

    disabled_agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://disabled.example/v1",
    )
    assert disabled_agent.context_compressor.config_context_length is None

    cfg["custom_providers"] = [
        {
            "name": "Route Key",
            "base_url": "https://legacy.example/v1",
        }
    ]
    legacy_agent = _make_direct_start_agent(
        cfg,
        model="shared-model",
        provider="custom",
        base_url="https://legacy.example/v1",
    )
    assert legacy_agent.context_compressor.config_context_length == 1_048_576


def test_direct_start_runtime_first_provider_names_require_explicit_custom_prefix():
    """Auto, MoA, and Vertex routes cannot be shadowed by raw custom names."""
    for provider_name in (
        "auto",
        "moa",
        "vertex",
        "google-vertex",
        "vertex-ai",
        "gcp-vertex",
        "vertexai",
    ):
        base_url = f"https://{provider_name}.shadow.example/v1"
        cfg = {
            "model": {
                "default": "shared-model",
                "provider": provider_name,
                "context_length": 1_048_576,
            },
            "providers": {
                provider_name: {
                    "api": base_url,
                }
            },
        }

        raw_agent = _make_direct_start_agent(
            cfg,
            model="shared-model",
            provider="custom",
            base_url=base_url,
        )
        assert raw_agent.context_compressor.config_context_length is None

        cfg["model"]["provider"] = f"custom:{provider_name}"
        custom_agent = _make_direct_start_agent(
            cfg,
            model="shared-model",
            provider="custom",
            base_url=base_url,
        )
        assert custom_agent.context_compressor.config_context_length == 1_048_576
