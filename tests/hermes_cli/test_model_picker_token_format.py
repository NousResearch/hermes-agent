"""Regression tests for #8826 — the /model picker must not advertise a
provider whose only credential has a definitively unusable format.

``list_authenticated_providers()`` gated provider rows on a bare truthiness
check over the provider's env vars, so ``GITHUB_TOKEN=ghp_...`` (a classic
PAT, which the Copilot API rejects) made GitHub Copilot look authenticated
in the picker and then fail on first use.

Only formats the provider itself documents as unusable are rejected — a
provider with no declared validator keeps its old behaviour, so this can
never hide an endpoint we have no rule for.

No network calls.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli.auth import PROVIDER_REGISTRY, provider_credential_format_ok
from hermes_cli.model_switch import list_authenticated_providers

# Shapes only — none of these are real credentials.
CLASSIC_PAT = "ghp_" + "A" * 36
OAUTH_TOKEN = "gho_" + "B" * 36
FINE_GRAINED_PAT = "github_pat_" + "C" * 70
APP_TOKEN = "ghu_" + "D" * 36

COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")


@pytest.fixture
def clean_env(monkeypatch):
    """Blank every credential env var the assertions depend on."""
    for var in COPILOT_ENV_VARS + ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _picker_slugs(**kwargs) -> set[str]:
    """Run the picker with every non-env credential source turned off."""
    with patch("hermes_cli.model_switch._credential_pool_is_usable", return_value=False), \
         patch("hermes_cli.auth._load_auth_store", return_value={}):
        rows = list_authenticated_providers(current_provider="openrouter", max_models=5, **kwargs)
    return {row["slug"] for row in rows}


def test_copilot_hidden_when_only_credential_is_a_classic_pat(clean_env):
    """The exact #8826 repro: ghp_* in GITHUB_TOKEN must not list Copilot."""
    clean_env.setenv("GITHUB_TOKEN", CLASSIC_PAT)

    assert "copilot" not in _picker_slugs()


@pytest.mark.parametrize("env_var", COPILOT_ENV_VARS)
def test_copilot_hidden_for_classic_pat_in_any_env_var(clean_env, env_var):
    """All three Copilot env vars go through the same format gate."""
    clean_env.setenv(env_var, CLASSIC_PAT)

    assert "copilot" not in _picker_slugs()


@pytest.mark.parametrize("token", [OAUTH_TOKEN, FINE_GRAINED_PAT, APP_TOKEN])
def test_copilot_listed_for_supported_token_shapes(clean_env, token):
    """Supported prefixes must keep listing Copilot — no false negatives."""
    clean_env.setenv("GITHUB_TOKEN", token)

    assert "copilot" in _picker_slugs()


def test_one_usable_token_is_enough(clean_env):
    """A rejected token in one var must not mask a usable one in another."""
    clean_env.setenv("GITHUB_TOKEN", CLASSIC_PAT)
    clean_env.setenv("COPILOT_GITHUB_TOKEN", OAUTH_TOKEN)

    assert "copilot" in _picker_slugs()


def test_provider_without_a_format_rule_is_unaffected(clean_env):
    """Providers that declare no validator keep the old truthiness behaviour."""
    clean_env.setenv("OPENAI_API_KEY", "not-a-recognizable-shape")

    assert "openai-api" in _picker_slugs()


def test_registry_helper_passes_providers_without_a_validator():
    """Unknown/undeclared providers must never be rejected on format."""
    assert provider_credential_format_ok("openai-api", "anything-at-all")
    assert provider_credential_format_ok("not-a-real-provider", "anything-at-all")


def test_registry_helper_rejects_blank_values():
    assert not provider_credential_format_ok("copilot", "")
    assert not provider_credential_format_ok("copilot", "   ")


def test_registry_helper_uses_the_copilot_validator():
    assert PROVIDER_REGISTRY["copilot"].token_format_validator is not None
    assert not provider_credential_format_ok("copilot", CLASSIC_PAT)
    assert provider_credential_format_ok("copilot", OAUTH_TOKEN)


def test_registry_helper_allows_token_when_validator_raises():
    """A broken validator must fail open — hiding a working provider is worse."""
    def _boom(_token):
        raise RuntimeError("validator exploded")

    with patch.object(PROVIDER_REGISTRY["copilot"], "token_format_validator", _boom):
        assert provider_credential_format_ok("copilot", CLASSIC_PAT)
