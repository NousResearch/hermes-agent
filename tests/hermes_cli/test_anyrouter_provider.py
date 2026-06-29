"""Tests for AnyRouter provider support — OpenRouter-compatible aggregator.

AnyRouter is a routing aggregator (vendor/model slugs) registered via a
provider profile plugin. It auto-wires into auth.py, config.py, models.py,
and model_metadata.py from the registry, and carries an ``is_aggregator``
overlay in hermes_cli.providers for routing-slug behavior.
"""

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_provider,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
)


_OTHER_PROVIDER_KEYS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "NOVITA_API_KEY", "XAI_API_KEY", "GLM_API_KEY", "ZAI_API_KEY",
)


# =============================================================================
# Provider profile
# =============================================================================


class TestAnyRouterProfile:
    def test_profile_registered(self):
        from providers import get_provider_profile
        p = get_provider_profile("anyrouter")
        assert p is not None
        assert p.name == "anyrouter"

    def test_base_url(self):
        from providers import get_provider_profile
        assert get_provider_profile("anyrouter").base_url == "https://anyrouter.dev/api/v1"

    def test_models_url(self):
        from providers import get_provider_profile
        assert get_provider_profile("anyrouter").models_url == "https://anyrouter.dev/api/v1/models"

    def test_env_vars(self):
        from providers import get_provider_profile
        assert get_provider_profile("anyrouter").env_vars == (
            "ANYROUTER_API_KEY", "ANYROUTER_BASE_URL",
        )


# =============================================================================
# Aggregator routing
# =============================================================================


class TestAnyRouterAggregator:
    def test_overlay_is_aggregator(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "anyrouter" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["anyrouter"]
        assert overlay.is_aggregator
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_override == "https://anyrouter.dev/api/v1"
        assert overlay.base_url_env_var == "ANYROUTER_BASE_URL"

    def test_is_aggregator_helpers(self):
        from hermes_cli.providers import is_aggregator, is_routing_aggregator
        assert is_aggregator("anyrouter")
        # Routing aggregator: selecting a model re-routes to other vendors'
        # endpoints, so the picker must treat it like OpenRouter (not a
        # flat-namespace reseller).
        assert is_routing_aggregator("anyrouter")

    def test_get_provider(self):
        from hermes_cli.providers import get_provider
        pd = get_provider("anyrouter")
        assert pd is not None
        assert pd.id == "anyrouter"
        assert pd.is_aggregator
        assert pd.base_url == "https://anyrouter.dev/api/v1"
        assert pd.api_key_env_vars == ("ANYROUTER_API_KEY",)


# =============================================================================
# Aliases / normalization
# =============================================================================


class TestAnyRouterNormalization:
    def test_normalize_providers_py(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("anyrouter") == "anyrouter"

    def test_normalize_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("anyrouter") == "anyrouter"


# =============================================================================
# Registry auto-wiring
# =============================================================================


class TestAnyRouterRegistry:
    def test_in_provider_registry(self):
        assert "anyrouter" in PROVIDER_REGISTRY
        cfg = PROVIDER_REGISTRY["anyrouter"]
        assert cfg.auth_type == "api_key"
        assert cfg.inference_base_url == "https://anyrouter.dev/api/v1"
        assert cfg.api_key_env_vars == ("ANYROUTER_API_KEY",)
        assert cfg.base_url_env_var == "ANYROUTER_BASE_URL"

    def test_canonical_provider_entry(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        assert "anyrouter" in [p.slug for p in CANONICAL_PROVIDERS]

    def test_label(self):
        from hermes_cli.models import _PROVIDER_LABELS
        assert _PROVIDER_LABELS["anyrouter"] == "AnyRouter"

    def test_env_vars_surfaced_in_config(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "ANYROUTER_API_KEY" in OPTIONAL_ENV_VARS
        assert "ANYROUTER_BASE_URL" in OPTIONAL_ENV_VARS

    def test_prefix_recognized(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "anyrouter" in _PROVIDER_PREFIXES

    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER
        assert _URL_TO_PROVIDER.get("anyrouter.dev") == "anyrouter"


# =============================================================================
# Credentials
# =============================================================================


class TestAnyRouterCredentials:
    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("ANYROUTER_API_KEY", "ar-test")
        assert get_api_key_provider_status("anyrouter")["configured"]

    def test_status_not_configured(self, monkeypatch):
        monkeypatch.delenv("ANYROUTER_API_KEY", raising=False)
        assert not get_api_key_provider_status("anyrouter")["configured"]

    def test_openrouter_key_does_not_configure_anyrouter(self, monkeypatch):
        """An OpenRouter key must NOT make AnyRouter look configured."""
        monkeypatch.delenv("ANYROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        assert not get_api_key_provider_status("anyrouter")["configured"]

    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("ANYROUTER_API_KEY", "ar-direct-key")
        monkeypatch.delenv("ANYROUTER_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("anyrouter")
        assert creds["api_key"] == "ar-direct-key"
        assert creds["base_url"] == "https://anyrouter.dev/api/v1"

    def test_custom_base_url_override(self, monkeypatch):
        monkeypatch.setenv("ANYROUTER_API_KEY", "ar-x")
        monkeypatch.setenv("ANYROUTER_BASE_URL", "https://proxy.anyrouter.example/v1")
        creds = resolve_api_key_provider_credentials("anyrouter")
        assert creds["base_url"] == "https://proxy.anyrouter.example/v1"

    def test_resolve_provider_with_key(self, monkeypatch):
        for key in _OTHER_PROVIDER_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("ANYROUTER_API_KEY", "ar-test-12345")
        assert resolve_provider("anyrouter") == "anyrouter"


# =============================================================================
# Request building — OpenRouter-compatible passthrough
# =============================================================================


class TestAnyRouterRequestBuilding:
    def _profile(self):
        from providers import get_provider_profile
        return get_provider_profile("anyrouter")

    def test_session_id_passthrough(self):
        """session_id is AnyRouter's sticky-session id (body wins over header)."""
        body = self._profile().build_extra_body(session_id="s1")
        assert body["session_id"] == "s1"

    def test_provider_preferences_passthrough(self):
        body = self._profile().build_extra_body(
            session_id="s1", provider_preferences={"sort": "ttft"}
        )
        assert body["session_id"] == "s1"
        assert body["provider"] == {"sort": "ttft"}

    def test_no_session_no_prefs(self):
        assert self._profile().build_extra_body() == {}

    def test_reasoning_passthrough(self):
        """AnyRouter accepts the reasoning object natively — pass it through."""
        extra_body, top = self._profile().build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            supports_reasoning=True,
            model="anthropic/claude-sonnet-4.6",
        )
        assert extra_body["reasoning"] == {"enabled": True, "effort": "high"}
        assert top == {}

    def test_reasoning_default_when_enabled_without_config(self):
        extra_body, _ = self._profile().build_api_kwargs_extras(
            reasoning_config=None, supports_reasoning=True, model="openai/gpt-5.4"
        )
        assert extra_body["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_reasoning_omitted_when_unsupported(self):
        extra_body, top = self._profile().build_api_kwargs_extras(
            reasoning_config={"effort": "high"}, supports_reasoning=False,
        )
        assert extra_body == {}
        assert top == {}


# =============================================================================
# App attribution
# =============================================================================


class TestAnyRouterAppAttribution:
    def test_attribution_headers(self):
        """Hermes traffic is credited in AnyRouter's app rankings."""
        from providers import get_provider_profile
        headers = get_provider_profile("anyrouter").default_headers
        assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
        assert headers["X-AnyRouter-Title"] == "Hermes Agent"
        assert headers["X-AnyRouter-Source"] == "cli-agent"
        assert headers["X-AnyRouter-Categories"] == "cli-agent"
