"""Tests for #101424: model-identifier normalization at the API surface.

The WebUI model picker advertises ``@anthropic:claude-sonnet-5``-style ids
(and the browser runtime uses ``anthropic::claude-sonnet-5``).  Neither
form is a valid wire model id — the prefix is a routing hint — yet
``_request_agent_overrides`` forwarded them verbatim, so the provider API
404'd on the prefixed string.  ``default`` was also forwarded verbatim
instead of resolving to the configured default model.
"""

from __future__ import annotations

import pytest

from gateway.platforms.api_server import (
    _request_agent_overrides,
    _split_prefixed_model_identifier,
)


# ---------------------------------------------------------------------------
# _split_prefixed_model_identifier
# ---------------------------------------------------------------------------


class TestSplitPrefixedModelIdentifier:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # WebUI picker form
            ("@anthropic:claude-sonnet-5", ("anthropic", "claude-sonnet-5")),
            # browser-runtime / session form
            ("anthropic::claude-sonnet-5", ("anthropic", "claude-sonnet-5")),
            # bare ids pass through untouched
            ("claude-sonnet-5", ("", "claude-sonnet-5")),
            # vendor/slash ids are NOT this helper's job (normalize_model_for_provider)
            ("openai/gpt-5", ("", "openai/gpt-5")),
            ("", ("", "")),
            (None, ("", "")),
        ],
    )
    def test_forms(self, model, expected):
        assert _split_prefixed_model_identifier(model) == expected

    @pytest.mark.parametrize(
        "model",
        [
            # @ without a colon, malformed slug, empty bare part
            "@anthropicclaude-sonnet-5",
            "@:claude-sonnet-5",
            "@anthropic:",
            "@an thropic:claude-sonnet-5",
            "not-a-prefix@at-middle:claude",
        ],
    )
    def test_malformed_never_split(self, model):
        provider, bare = _split_prefixed_model_identifier(model)
        assert provider == ""
        assert bare == model


# ---------------------------------------------------------------------------
# _request_agent_overrides normalization
# ---------------------------------------------------------------------------


class TestPrefixedModelNormalization:
    def test_matching_prefix_is_stripped_and_provider_kept(self):
        overrides = _request_agent_overrides(
            {"provider": "anthropic", "model": "@anthropic:claude-sonnet-5"}
        )
        assert overrides["requested_provider"] == "anthropic"
        assert overrides["requested_model"] == "claude-sonnet-5"
        assert "request_error" not in overrides

    def test_prefix_routes_when_no_explicit_provider(self):
        overrides = _request_agent_overrides(
            {"model": "@anthropic:claude-sonnet-5"}
        )
        assert overrides["requested_provider"] == "anthropic"
        assert overrides["requested_model"] == "claude-sonnet-5"

    def test_double_colon_form_is_split_too(self):
        overrides = _request_agent_overrides(
            {"model": "anthropic::claude-sonnet-5"}
        )
        assert overrides["requested_provider"] == "anthropic"
        assert overrides["requested_model"] == "claude-sonnet-5"

    def test_disagreeing_prefix_and_provider_is_an_error(self):
        overrides = _request_agent_overrides(
            {"provider": "anthropic", "model": "@openai:gpt-5"}
        )
        assert "requested_model" not in overrides
        assert "requested_provider" not in overrides
        err = overrides.get("request_error") or ""
        assert "anthropic" in err and "openai" in err

    def test_bare_model_with_provider_still_passes_through(self):
        # The issue's controlled experiment: bare name + provider works, and
        # must keep working byte-for-byte.
        overrides = _request_agent_overrides(
            {"provider": "anthropic", "model": "claude-sonnet-5"}
        )
        assert overrides["requested_provider"] == "anthropic"
        assert overrides["requested_model"] == "claude-sonnet-5"

    def test_model_options_still_carried(self):
        overrides = _request_agent_overrides(
            {
                "provider": "anthropic",
                "model": "@anthropic:claude-sonnet-5",
                "model_options": {"reasoning_config": {"effort": "high"}},
            }
        )
        assert overrides["model_options"] == {
            "reasoning_config": {"effort": "high"}
        }


class TestDefaultAliasResolution:
    def test_default_with_provider_resolves_to_gateway_default(self):
        # "default" is a resolution instruction: no requested_model means the
        # agent falls back to model.default from config; the provider still
        # routes the request.
        overrides = _request_agent_overrides(
            {"provider": "anthropic", "model": "default"}
        )
        assert overrides["requested_provider"] == "anthropic"
        assert "requested_model" not in overrides

    def test_prefixed_default_routes_on_prefix(self):
        overrides = _request_agent_overrides(
            {"model": "@anthropic:default"}
        )
        assert overrides["requested_provider"] == "anthropic"
        assert "requested_model" not in overrides

    def test_virtual_alias_still_means_default(self):
        overrides = _request_agent_overrides(
            {"model": "hermes-agent"}, virtual_model="hermes-agent"
        )
        assert "requested_model" not in overrides


class TestBareModelGateUnchanged:
    def test_bare_model_dropped_when_disallowed(self):
        overrides = _request_agent_overrides(
            {"model": "openai/gpt-5"}, allow_bare_model=False
        )
        assert "requested_model" not in overrides

    def test_bare_model_honored_when_allowed(self):
        overrides = _request_agent_overrides(
            {"model": "openai/gpt-5"}, allow_bare_model=True
        )
        assert overrides["requested_model"] == "openai/gpt-5"
