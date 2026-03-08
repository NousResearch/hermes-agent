"""Tests for provider-aware `/model` validation in hermes_cli.models."""

from unittest.mock import patch

from hermes_cli.models import (
    fetch_api_models,
    normalize_provider,
    provider_model_ids,
    validate_requested_model,
)


# -- helpers -----------------------------------------------------------------

# Simulated API model list for mocking fetch_api_models
FAKE_API_MODELS = [
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.4-pro",
    "openai/gpt-5.4",
    "google/gemini-3-pro-preview",
]


def _validate(model, provider="openrouter", api_models=FAKE_API_MODELS, **kw):
    """Shortcut: call validate_requested_model with mocked API."""
    with patch("hermes_cli.models.fetch_api_models", return_value=api_models):
        return validate_requested_model(model, provider, **kw)


# -- normalize_provider ------------------------------------------------------

class TestNormalizeProvider:
    def test_defaults_to_openrouter(self):
        assert normalize_provider(None) == "openrouter"
        assert normalize_provider("") == "openrouter"

    def test_known_aliases(self):
        assert normalize_provider("glm") == "zai"
        assert normalize_provider("z-ai") == "zai"
        assert normalize_provider("z.ai") == "zai"
        assert normalize_provider("zhipu") == "zai"
        assert normalize_provider("kimi") == "kimi-coding"
        assert normalize_provider("moonshot") == "kimi-coding"
        assert normalize_provider("minimax-china") == "minimax-cn"

    def test_canonical_ids_pass_through(self):
        assert normalize_provider("openrouter") == "openrouter"
        assert normalize_provider("nous") == "nous"
        assert normalize_provider("openai-codex") == "openai-codex"

    def test_case_insensitive(self):
        assert normalize_provider("OpenRouter") == "openrouter"
        assert normalize_provider("GLM") == "zai"


# -- provider_model_ids ------------------------------------------------------

class TestProviderModelIds:
    def test_openrouter_returns_curated_list(self):
        ids = provider_model_ids("openrouter")
        assert len(ids) > 0
        assert all("/" in mid for mid in ids)

    def test_unknown_provider_returns_empty(self):
        assert provider_model_ids("some-unknown-provider") == []

    def test_zai_returns_glm_models(self):
        ids = provider_model_ids("zai")
        assert "glm-5" in ids

    def test_alias_resolves_correctly(self):
        assert provider_model_ids("glm") == provider_model_ids("zai")


# -- fetch_api_models --------------------------------------------------------

class TestFetchApiModels:
    def test_returns_none_when_no_base_url(self):
        assert fetch_api_models("key", None) is None
        assert fetch_api_models("key", "") is None

    def test_returns_none_on_network_error(self):
        with patch("hermes_cli.models.urllib.request.urlopen", side_effect=Exception("timeout")):
            assert fetch_api_models("key", "https://example.com/v1") is None


# -- validate_requested_model — format checks (no API needed) ----------------

class TestValidateFormatChecks:
    def test_empty_model_rejected(self):
        result = _validate("")
        assert result["accepted"] is False
        assert "empty" in result["message"]

    def test_whitespace_only_rejected(self):
        result = _validate("   ")
        assert result["accepted"] is False
        assert "empty" in result["message"]

    def test_model_with_spaces_rejected(self):
        result = _validate("anthropic/ claude-opus")
        assert result["accepted"] is False
        assert "spaces" in result["message"].lower()

    def test_no_slash_model_still_probes_api(self):
        """Models without '/' should still be checked via API (not all providers need it)."""
        result = _validate("gpt-5.4", api_models=["gpt-5.4", "gpt-5.4-pro"])
        assert result["accepted"] is True
        assert result["persist"] is True

    def test_no_slash_model_rejected_if_not_in_api(self):
        result = _validate("gpt-5.4", api_models=["openai/gpt-5.4"])
        assert result["accepted"] is False
        assert "not a valid model" in result["message"]


# -- validate_requested_model — API probe found model ------------------------

class TestValidateApiFound:
    def test_model_found_in_api_is_accepted_and_persisted(self):
        result = _validate("anthropic/claude-opus-4.6")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    def test_model_found_in_api_for_custom_endpoint(self):
        result = _validate(
            "my-model",
            provider="openrouter",
            api_models=["my-model", "other-model"],
            base_url="http://localhost:11434/v1",
        )
        assert result["accepted"] is True
        assert result["persist"] is True


# -- validate_requested_model — API probe model not found --------------------

class TestValidateApiNotFound:
    def test_model_not_in_api_is_rejected(self):
        result = _validate("anthropic/claude-nonexistent")
        assert result["accepted"] is False
        assert result["persist"] is False
        assert "not a valid model" in result["message"]

    def test_rejection_includes_suggestions(self):
        result = _validate("anthropic/claude-opus-4.5")  # close to claude-opus-4.6
        assert result["accepted"] is False
        assert "Did you mean" in result["message"]

    def test_completely_wrong_model_rejected(self):
        result = _validate("totally/fake-model-xyz")
        assert result["accepted"] is False
        assert "not a valid model" in result["message"]


# -- validate_requested_model — API unreachable (fallback) -------------------

class TestValidateApiFallback:
    def test_known_catalog_model_accepted_when_api_down(self):
        """If API is unreachable, fall back to hardcoded catalog."""
        result = _validate("anthropic/claude-opus-4.6", api_models=None)
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True

    def test_unknown_model_is_session_only_when_api_down(self):
        result = _validate("anthropic/claude-next-gen", api_models=None)
        assert result["accepted"] is True
        assert result["persist"] is False
        assert "Could not validate" in result["message"]
        assert "session only" in result["message"].lower()

    def test_zai_known_model_accepted_when_api_down(self):
        result = _validate("glm-5", provider="zai", api_models=None)
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True

    def test_zai_unknown_model_session_only_when_api_down(self):
        result = _validate("glm-99", provider="zai", api_models=None)
        assert result["accepted"] is True
        assert result["persist"] is False

    def test_unknown_provider_session_only_when_api_down(self):
        result = _validate("some-model", provider="totally-unknown", api_models=None)
        assert result["accepted"] is True
        assert result["persist"] is False
        assert result["message"] is not None
