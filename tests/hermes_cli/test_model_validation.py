"""Tests for provider-aware `/model` validation in hermes_cli.models."""

from unittest.mock import patch

from hermes_cli.models import (
    curated_models_for_provider,
    fetch_api_models,
    normalize_provider,
    parse_model_input,
    provider_label,
    provider_model_ids,
    validate_requested_model,
)


# -- helpers -----------------------------------------------------------------

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


# -- parse_model_input -------------------------------------------------------

class TestParseModelInput:
    def test_plain_model_keeps_current_provider(self):
        provider, model = parse_model_input("anthropic/claude-sonnet-4.5", "openrouter")
        assert provider == "openrouter"
        assert model == "anthropic/claude-sonnet-4.5"

    def test_provider_colon_model_switches_provider(self):
        provider, model = parse_model_input("openrouter:anthropic/claude-sonnet-4.5", "nous")
        assert provider == "openrouter"
        assert model == "anthropic/claude-sonnet-4.5"

    def test_provider_alias_resolved(self):
        provider, model = parse_model_input("glm:glm-5", "openrouter")
        assert provider == "zai"
        assert model == "glm-5"

    def test_no_slash_no_colon_keeps_provider(self):
        provider, model = parse_model_input("gpt-5.4", "openrouter")
        assert provider == "openrouter"
        assert model == "gpt-5.4"

    def test_nous_provider_switch(self):
        provider, model = parse_model_input("nous:hermes-3", "openrouter")
        assert provider == "nous"
        assert model == "hermes-3"

    def test_empty_model_after_colon_keeps_current(self):
        provider, model = parse_model_input("openrouter:", "nous")
        assert provider == "nous"
        assert model == "openrouter:"

    def test_colon_at_start_keeps_current(self):
        provider, model = parse_model_input(":something", "openrouter")
        assert provider == "openrouter"
        assert model == ":something"

    def test_unknown_prefix_colon_not_treated_as_provider(self):
        """Colons are only provider delimiters if the left side is a known provider."""
        provider, model = parse_model_input("anthropic/claude-3.5-sonnet:beta", "openrouter")
        assert provider == "openrouter"
        assert model == "anthropic/claude-3.5-sonnet:beta"

    def test_http_url_not_treated_as_provider(self):
        provider, model = parse_model_input("http://localhost:8080/model", "openrouter")
        assert provider == "openrouter"
        assert model == "http://localhost:8080/model"


# -- curated_models_for_provider ---------------------------------------------

class TestCuratedModelsForProvider:
    def test_openrouter_returns_curated_list(self):
        models = curated_models_for_provider("openrouter")
        assert len(models) > 0
        assert any("claude" in m[0] for m in models)

    def test_zai_returns_glm_models(self):
        models = curated_models_for_provider("zai")
        assert any("glm" in m[0] for m in models)

    def test_unknown_provider_returns_empty(self):
        assert curated_models_for_provider("totally-unknown") == []


# -- normalize_provider ------------------------------------------------------

class TestNormalizeProvider:
    def test_defaults_to_openrouter(self):
        assert normalize_provider(None) == "openrouter"
        assert normalize_provider("") == "openrouter"

    def test_known_aliases(self):
        assert normalize_provider("glm") == "zai"
        assert normalize_provider("kimi") == "kimi-coding"
        assert normalize_provider("moonshot") == "kimi-coding"

    def test_case_insensitive(self):
        assert normalize_provider("OpenRouter") == "openrouter"


class TestProviderLabel:
    def test_known_labels_and_auto(self):
        assert provider_label("anthropic") == "Anthropic"
        assert provider_label("kimi") == "Kimi / Moonshot"
        assert provider_label("auto") == "Auto"

    def test_unknown_provider_preserves_original_name(self):
        assert provider_label("my-custom-provider") == "my-custom-provider"


# -- provider_model_ids ------------------------------------------------------

class TestProviderModelIds:
    def test_openrouter_returns_curated_list(self):
        ids = provider_model_ids("openrouter")
        assert len(ids) > 0
        assert all("/" in mid for mid in ids)

    def test_unknown_provider_returns_empty(self):
        assert provider_model_ids("some-unknown-provider") == []

    def test_zai_returns_glm_models(self):
        assert "glm-5" in provider_model_ids("zai")


# -- fetch_api_models --------------------------------------------------------

class TestFetchApiModels:
    def test_returns_none_when_no_base_url(self):
        assert fetch_api_models("key", None) is None

    def test_returns_none_on_network_error(self):
        with patch("hermes_cli.models.urllib.request.urlopen", side_effect=Exception("timeout")):
            assert fetch_api_models("key", "https://example.com/v1") is None


# -- validate — format checks -----------------------------------------------

class TestValidateFormatChecks:
    def test_empty_model_rejected(self):
        result = _validate("")
        assert result["accepted"] is False
        assert "empty" in result["message"]

    def test_whitespace_only_rejected(self):
        result = _validate("   ")
        assert result["accepted"] is False

    def test_model_with_spaces_rejected(self):
        result = _validate("anthropic/ claude-opus")
        assert result["accepted"] is False

    def test_no_slash_model_still_probes_api(self):
        result = _validate("gpt-5.4", api_models=["gpt-5.4", "gpt-5.4-pro"])
        assert result["accepted"] is True
        assert result["persist"] is True

    def test_no_slash_model_rejected_if_not_in_api(self):
        result = _validate("gpt-5.4", api_models=["openai/gpt-5.4"])
        assert result["accepted"] is True
        assert "not found" in result["message"]


# -- validate — API found ----------------------------------------------------

class TestValidateApiFound:
    def test_model_found_in_api(self):
        result = _validate("anthropic/claude-opus-4.6")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True

    def test_model_found_for_custom_endpoint(self):
        result = _validate(
            "my-model", provider="openrouter",
            api_models=["my-model"], base_url="http://localhost:11434/v1",
        )
        assert result["accepted"] is True
        assert result["persist"] is True


# -- validate — API not found ------------------------------------------------

class TestValidateApiNotFound:
    def test_model_not_in_api_accepted_with_warning(self):
        result = _validate("anthropic/claude-nonexistent")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert "not found" in result["message"]

    def test_warning_includes_suggestions(self):
        result = _validate("anthropic/claude-opus-4.5")
        assert result["accepted"] is True
        assert "Similar models" in result["message"]


# -- validate — API unreachable — accept and persist everything ----------------

class TestValidateApiFallback:
    def test_any_model_accepted_when_api_down(self):
        result = _validate("anthropic/claude-opus-4.6", api_models=None)
        assert result["accepted"] is True
        assert result["persist"] is True

    def test_unknown_model_also_accepted_when_api_down(self):
        """No hardcoded catalog gatekeeping — accept, persist, and warn."""
        result = _validate("anthropic/claude-next-gen", api_models=None)
        assert result["accepted"] is True
        assert result["persist"] is True
        assert "could not reach" in result["message"].lower()

    def test_zai_model_accepted_when_api_down(self):
        result = _validate("glm-5", provider="zai", api_models=None)
        assert result["accepted"] is True
        assert result["persist"] is True

    def test_unknown_provider_accepted_when_api_down(self):
        result = _validate("some-model", provider="totally-unknown", api_models=None)
        assert result["accepted"] is True
        assert result["persist"] is True
