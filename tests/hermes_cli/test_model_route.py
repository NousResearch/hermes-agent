"""Parity tests for ``hermes_cli.model_route.resolve_requested_route`` (the stage-1
extraction backing the Kanban per-provider concurrency budget, #123654).

Every row asserts the FULL ``RequestedRoute`` and — where a HermesCLI can be built cheaply —
that constructing the real CLI object against the same inputs yields the same
``model`` / ``requested_provider`` / explicit base URL values.
"""

from __future__ import annotations

import pytest

from hermes_cli.model_route import RequestedRoute, resolve_requested_route


def _cfg(model=None, provider=None, base_url=None, default=None):
    out = {}
    if model is not None:
        out["model"] = model
    if provider is not None:
        out["provider"] = provider
    if base_url is not None:
        out["base_url"] = base_url
    if default is not None:
        out["default"] = default
    return out


class TestPrecedence:
    def test_explicit_flag_provider_wins(self):
        route = resolve_requested_route(
            model="m", provider="anthropic", model_config=_cfg(provider="openrouter"),
        )
        assert route == RequestedRoute(model="m", requested_provider="anthropic")

    def test_moa_prefix_wins_over_provider_flag(self):
        route = resolve_requested_route(
            model="moa:fast", provider="anthropic", model_config=_cfg(provider="openrouter"),
        )
        assert route.model == "fast"
        assert route.requested_provider == "moa"

    def test_moa_prefix_without_preset_is_not_special(self):
        route = resolve_requested_route(
            model="moa:", provider="anthropic", model_config=_cfg(default="x"),
        )
        # No preset -> the CLI keeps the raw string and the explicit provider.
        assert route.model == "moa:"
        assert route.requested_provider == "anthropic"

    def test_nested_default_provider_beats_model_provider(self):
        route = resolve_requested_route(
            model=None, provider=None,
            model_config=_cfg(default={"model": "some-model", "provider": "anthropic"},
                             provider="openrouter"),
        )
        assert route.model == "some-model"
        assert route.requested_provider == "anthropic"

    def test_model_provider_when_no_flags(self):
        route = resolve_requested_route(
            model=None, provider=None, model_config=_cfg(default="m", provider="gemini"),
        )
        assert route.model == "m"
        assert route.requested_provider == "gemini"
        assert route.config_model == "m"

    def test_env_provider_after_model_provider(self):
        route = resolve_requested_route(
            model=None, provider=None, model_config=_cfg(default="m"),
            env_provider="mistral",
        )
        assert route.requested_provider == "mistral"

    def test_nothing_pinned_resolves_auto(self):
        route = resolve_requested_route(model=None, provider=None, model_config=_cfg(default="m"))
        assert route.model == "m"
        assert route.requested_provider == "auto"
        assert route.config_model == "m"


class TestStartupRoute:
    def test_provider_model_string(self):
        """``provider/model`` for a CONFIGURED provider routes to it (T5 arm 1).

        An aggregator-native slug (``anthropic/x`` under an openrouter
        current_provider) deliberately stays on the aggregator — that guard
        lives in ``resolve_startup_model_route`` and the extraction keeps it.
        """
        route = resolve_requested_route(
            model="acme-labs/acme-model", provider=None,
            model_config=_cfg(default="fallback", provider="openrouter"),
            user_providers={"acme-labs": {"name": "Acme", "base_url": "https://acme.example/v1"}},
        )
        assert route.model == "acme-model"
        assert route.requested_provider == "acme-labs"

    def test_aggregator_slug_stays_on_aggregator(self):
        """``anthropic/claude-opus-4.6`` under OpenRouter is aggregator-native (no steal)."""
        route = resolve_requested_route(
            model="anthropic/claude-opus-4.6", provider=None,
            model_config=_cfg(default="fallback", provider="openrouter"),
        )
        assert route.model == "anthropic/claude-opus-4.6"
        assert route.requested_provider == "openrouter"

    def test_direct_alias_with_base_url(self, monkeypatch):
        """A URL-bearing direct alias pins model/provider/base_url + its api key."""
        from hermes_cli import model_switch

        monkeypatch.setattr(
            model_switch, "DIRECT_ALIASES",
            {"myalias": model_switch.DirectAlias(
                "my-model-id", "custom", "http://alias.example:8000/v1",
                api_key="not-needed")},
        )
        route = resolve_requested_route(
            model="myalias", provider=None,
            model_config=_cfg(default="fallback", provider="openrouter"),
        )
        assert route.model == "my-model-id"
        assert route.requested_provider == "custom"
        assert route.explicit_base_url == "http://alias.example:8000/v1"
        assert route.explicit_api_key == "not-needed"

    def test_provider_flag_over_startup_alias(self):
        """The explicit --provider still outranks a resolved startup alias provider."""
        from hermes_cli import model_switch

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            model_switch, "DIRECT_ALIASES",
            {"myalias": model_switch.DirectAlias("my-model-id", "zai", "")},
        )
        try:
            route = resolve_requested_route(
                model="myalias", provider="anthropic",
                model_config=_cfg(default="fallback", provider="openrouter"),
            )
            assert route.model == "my-model-id"
            assert route.requested_provider == "anthropic"
            assert route.explicit_base_url is None
        finally:
            monkeypatch.undo()


class TestConfigDefaults:
    def test_dict_valued_default_uses_model_key_too(self):
        route = resolve_requested_route(
            model=None, provider=None,
            model_config=_cfg(default={"default": "m2", "provider": "openai"}),
        )
        assert route.model == "m2"
        assert route.requested_provider == "openai"

    def test_explicit_model_wins_over_config_default(self):
        route = resolve_requested_route(
            model="flag-model", provider=None, model_config=_cfg(default="cfg-model"),
        )
        assert route.model == "flag-model"

    def test_empty_everything_yields_auto(self):
        route = resolve_requested_route(model=None, provider=None, model_config={})
        assert route.model == ""
        assert route.requested_provider == "auto"


class TestPurity:
    def test_no_env_read_when_env_provider_omitted(self, monkeypatch):
        """The function must not fall back to os.environ when env_provider is None."""
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
        route = resolve_requested_route(model="m", provider=None, model_config={})
        assert route.requested_provider == "auto"

    def test_env_provider_used_when_passed(self, monkeypatch):
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
        route = resolve_requested_route(
            model="m", provider=None, model_config={}, env_provider="mistral",
        )
        assert route.requested_provider == "mistral"
