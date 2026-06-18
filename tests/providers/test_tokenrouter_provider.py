"""TokenRouter built-in provider wiring.

TokenRouter ships as a fast-path API-key provider plugin
(``plugins/model-providers/tokenrouter/``). Adding only that plugin must wire
the provider end-to-end via the auto-extend hooks, with no edits to the picker
or auth registry. These tests pin that contract so a future refactor of the
auto-extend paths can't silently drop the provider.
"""

from __future__ import annotations


def test_profile_registered_with_expected_identity():
    from providers import get_provider_profile

    prof = get_provider_profile("tokenrouter")
    assert prof is not None, "tokenrouter profile not discovered"
    assert prof.name == "tokenrouter"
    assert prof.base_url == "https://api.tokenrouter.com/v1"
    assert prof.display_name == "TokenRouter"
    assert prof.signup_url == "https://tokenrouter.com"
    assert "TOKENROUTER_API_KEY" in prof.env_vars
    # A base-URL override env var must be declared so auth can derive it.
    assert "TOKENROUTER_BASE_URL" in prof.env_vars
    # Curated fallback list is non-empty and namespaced like the live catalog.
    assert prof.fallback_models, "expected a curated fallback model list"


def test_alias_resolves():
    from providers import get_provider_profile

    assert get_provider_profile("token-router") is get_provider_profile("tokenrouter")


def test_appears_in_model_picker():
    """models.py auto-extends CANONICAL_PROVIDERS from the plugin registry."""
    from hermes_cli.models import _PROVIDER_LABELS

    assert _PROVIDER_LABELS.get("tokenrouter") == "TokenRouter"


def test_auth_registry_autowired():
    """auth.py auto-extends PROVIDER_REGISTRY; base-URL var derived from env_vars."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    assert "tokenrouter" in PROVIDER_REGISTRY
    cfg = PROVIDER_REGISTRY["tokenrouter"]
    assert cfg.auth_type == "api_key"
    assert cfg.inference_base_url == "https://api.tokenrouter.com/v1"
    assert cfg.api_key_env_vars == ("TOKENROUTER_API_KEY",)
    # The *_BASE_URL var is split out as the override, not treated as a key.
    assert cfg.base_url_env_var == "TOKENROUTER_BASE_URL"


def test_provider_identity_overlay():
    """hermes_cli/providers.py marks it an OpenAI-compatible aggregator."""
    from hermes_cli.providers import get_provider, is_aggregator, determine_api_mode

    pdef = get_provider("tokenrouter")
    assert pdef is not None
    assert pdef.name == "TokenRouter"
    assert pdef.base_url == "https://api.tokenrouter.com/v1"
    assert pdef.is_aggregator is True
    assert is_aggregator("tokenrouter") is True
    assert is_aggregator("token-router") is True
    assert determine_api_mode("tokenrouter") == "chat_completions"
