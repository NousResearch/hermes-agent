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


# ---------------------------------------------------------------------------
# HermesCLI parity (T25): each precedence row must also hold on a REAL
# HermesCLI construction, so the extraction cannot drift from the CLI ladder.
# Rows 1-3 are the ladder-collision cases where a startup-route provider
# string EQUALS a lower rung (env / config / nested default) — the CLI must
# keep the startup rung's provider exactly as origin/main's ladder did.
# ---------------------------------------------------------------------------


def _import_cli():
    import importlib
    import sys
    import types

    for name in list(sys.modules):
        if name in ("cli", "run_agent", "tools") or name.startswith("tools."):
            sys.modules.pop(name, None)

    if "firecrawl" not in sys.modules:
        sys.modules["firecrawl"] = types.SimpleNamespace(Firecrawl=object)
    try:
        importlib.import_module("prompt_toolkit")
    except ModuleNotFoundError:  # pragma: no cover - CI without prompt_toolkit
        from tests.hermes_cli.test_cli_provider_resolution import (
            _install_prompt_toolkit_stubs,
        )

        _install_prompt_toolkit_stubs()
    return importlib.import_module("cli")


@pytest.fixture(autouse=False)
def _restore_cli_modules():
    import sys

    prefixes = ("tools", "cli", "run_agent")
    original = {
        name: module
        for name, module in sys.modules.items()
        if any(name == p or name.startswith(p + ".") for p in prefixes)
    }
    try:
        yield
    finally:
        for name in list(sys.modules):
            if any(name == p or name.startswith(p + ".") for p in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(original)


class TestCliParity:
    """Real ``HermesCLI`` parity rows for the precedence ladder (T25).

    ``provider:model`` startup routes whose provider string collides with the
    env/cfg/nested rungs must keep the STARTUP provider (origin/main parity);
    the budget key derives from the same route (D2), so a regression here
    changes both the CLI's provider and its budget bucket.
    """

    ROWS = [
        # (model_flag, provider_flag, cfg_model_section, env, expected_model, expected_provider)
        # Row 1: env=zai, cfg provider=openrouter, -m zai:glm-5 -> zai wins (not cfg).
        ("zai:glm-5", None, {"default": "x", "provider": "openrouter"}, "zai",
         "glm-5", "zai"),
        # Row 2: nested default anthropic, cfg provider=zai, -m zai:glm-5 -> zai wins (not nested).
        ("zai:glm-5", None, {"default": {"model": "x", "provider": "anthropic"}, "provider": "zai"},
         None, "glm-5", "zai"),
        # Row 3: env=anthropic, cfg provider=openrouter, nested anthropic, -m anthropic:claude-x
        #   -> anthropic (the startup rung equals env AND nested; both keep it anthropic).
        ("anthropic:claude-x", None,
         {"default": {"model": "x", "provider": "anthropic"}, "provider": "openrouter"},
         "anthropic", "claude-x", "anthropic"),
        # Row 4 (arch A1 probe): env=anthropic, cfg provider=openrouter, NO
        # nested default, -m anthropic:claude-x -> anthropic (the startup rung
        # equals the env rung; it must not be dropped for the cfg rung).
        ("anthropic:claude-x", None,
         {"default": "x", "provider": "openrouter"},
         "anthropic", "claude-x", "anthropic"),
        # Row 5 (control): plain -m keeps the config provider when no startup route applies.
        ("glm-5", None, {"default": "x", "provider": "openrouter"}, None,
         "glm-5", "openrouter"),
        # Row 6 (control): explicit --provider beats a startup alias and cfg.
        ("zai:glm-5", "anthropic", {"default": "x", "provider": "openrouter"}, None,
         "zai:glm-5", "anthropic"),
    ]

    def test_cli_ladder_parity(self, monkeypatch, _restore_cli_modules):
        cli = _import_cli()
        for model_flag, provider_flag, model_cfg, env, want_model, want_provider in self.ROWS:
            monkeypatch.setitem(cli.CLI_CONFIG, "model", model_cfg)
            monkeypatch.setitem(cli.CLI_CONFIG, "providers", {})
            monkeypatch.setitem(cli.CLI_CONFIG, "custom_providers", [])
            monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", env or "")
            if env is None:
                monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
            shell = cli.HermesCLI(
                model=model_flag, provider=provider_flag, compact=True, max_turns=1)
            assert shell.model == want_model, (
                f"model mismatch for row (-m {model_flag!r}, cfg={model_cfg}, env={env!r}): "
                f"{shell.model!r} != {want_model!r}")
            assert shell.requested_provider == want_provider, (
                f"provider mismatch for row (-m {model_flag!r}, cfg={model_cfg}, env={env!r}): "
                f"{shell.requested_provider!r} != {want_provider!r}")

    def test_route_matches_cli_ladder(self, monkeypatch):
        """resolve_requested_route alone returns the same provider per row."""
        for model_flag, provider_flag, model_cfg, env, want_model, want_provider in self.ROWS:
            route = resolve_requested_route(
                model=model_flag, provider=provider_flag,
                model_config=model_cfg, env_provider=env,
            )
            assert route.model == want_model, (model_flag, model_cfg, env)
            assert route.requested_provider == want_provider, (
                model_flag, model_cfg, env, route.requested_provider)



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
