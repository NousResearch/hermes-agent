"""Unit tests for the Inworld Router provider profile.

Inworld exposes an OpenAI-compatible LLM router at ``https://api.inworld.ai/v1``.
The provider is registered as a declarative ``ProviderProfile`` plugin (no
adapter needed — standard Chat Completions wire format). These tests verify:

  1. The profile registers with the expected identity, auth, and endpoint.
  2. The default hooks don't emit request-side quirks (plain chat completions).
  3. The generic custom-provider path also covers the endpoint, including the
     ``extra_headers`` escape hatch for Inworld's Basic-auth scheme on its
     non-chat APIs.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def inworld_profile():
    """Resolve the registered Inworld profile via the provider registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    Inworld profile. Going through ``get_provider_profile`` keeps the test
    honest: if the registered class is ever swapped for a plain
    ``ProviderProfile`` the assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("inworld")
    assert profile is not None, "inworld provider profile must be registered"
    return profile


class TestInworldProfile:
    def test_registered_identity(self, inworld_profile):
        assert inworld_profile.name == "inworld"
        assert "inworld-router" in inworld_profile.aliases

    def test_openai_compatible_endpoint(self, inworld_profile):
        # Doc: https://docs.inworld.ai/router/openai-compatibility
        assert inworld_profile.api_mode == "chat_completions"
        assert inworld_profile.base_url == "https://api.inworld.ai/v1"
        assert inworld_profile.auth_type == "api_key"

    def test_env_vars(self, inworld_profile):
        assert inworld_profile.env_vars == ("INWORLD_API_KEY",)

    def test_plain_chat_completions_no_request_quirks(self, inworld_profile):
        """No extra_body / top-level kwargs unless something is configured.

        Inworld is a plain OpenAI-compatible router: no thinking flags, no
        provider-routing knobs should be emitted by the default hooks.
        """
        extra_body, top_level = inworld_profile.build_api_kwargs_extras(
            reasoning_config=None
        )
        assert extra_body == {}
        assert top_level == {}

    def test_hostname_derived_from_base_url(self, inworld_profile):
        assert inworld_profile.get_hostname() == "api.inworld.ai"


class TestInworldGenericCustomProviderPath:
    """The generic custom-provider mechanism already covers api.inworld.ai.

    A user can point a named custom provider at the Inworld endpoint without
    any code change — and, for Inworld's Basic-auth surfaces, the existing
    ``extra_headers`` mechanism provides the escape hatch.
    """

    def test_custom_provider_extra_headers_match_inworld_base_url(self):
        from hermes_cli.config import get_custom_provider_extra_headers

        providers = [
            {
                "name": "inworld",
                "base_url": "https://api.inworld.ai/v1",
                "extra_headers": {"Authorization": "Basic <base64-key>"},
            }
        ]
        headers = get_custom_provider_extra_headers(
            "https://api.inworld.ai/v1",
            custom_providers=providers,
        )
        assert headers == {"Authorization": "Basic <base64-key>"}

    def test_custom_provider_extra_headers_no_match_for_other_host(self):
        from hermes_cli.config import get_custom_provider_extra_headers

        providers = [
            {
                "name": "inworld",
                "base_url": "https://api.inworld.ai/v1",
                "extra_headers": {"Authorization": "Basic <base64-key>"},
            }
        ]
        # Prefix look-alike host must not match (no substring bypass).
        assert get_custom_provider_extra_headers(
            "https://api.inworld.ai.attacker.test/v1",
            custom_providers=providers,
        ) == {}
