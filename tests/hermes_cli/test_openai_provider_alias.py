"""resolve_provider must accept "openai" as an alias for "openai-api".

hermes-agent registers its built-in OpenAI provider as ``openai-api`` in
``PROVIDER_REGISTRY``.  Consumers (WebUI model picker, config.yaml, CLI
``--provider`` flag) may reasonably send the bare slug ``openai``.
Without an alias, ``resolve_provider("openai")`` raises ``AuthError``.
"""

import pytest

from hermes_cli.auth import resolve_provider, PROVIDER_REGISTRY


class TestOpenaiAlias:
    """The bare slug "openai" must resolve to the "openai-api" registry entry."""

    def test_openai_resolves_to_openai_api(self):
        assert resolve_provider("openai") == "openai-api"

    def test_case_insensitive(self):
        assert resolve_provider("OpenAI") == "openai-api"
        assert resolve_provider("OPENAI") == "openai-api"

    def test_canonical_openai_api_unchanged(self):
        assert resolve_provider("openai-api") == "openai-api"

    def test_openai_codex_unchanged(self):
        assert resolve_provider("openai-codex") == "openai-codex"

    def test_openai_api_in_registry(self):
        assert "openai-api" in PROVIDER_REGISTRY


class TestOpenaiModelsAlias:
    """models.py alias table must also map "openai" -> "openai-api"."""

    def test_models_alias(self):
        from hermes_cli.models import _PROVIDER_ALIASES
        assert _PROVIDER_ALIASES.get("openai") == "openai-api"

    def test_models_normalize_provider(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("openai") == "openai-api"
