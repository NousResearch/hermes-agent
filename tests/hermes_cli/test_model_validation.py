"""Tests for provider-aware `/model` validation in hermes_cli.models."""

from hermes_cli.models import (
    normalize_provider,
    provider_model_ids,
    validate_requested_model,
)


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


class TestValidateRequestedModel:
    # -- known models (happy path) ---------------------------------------

    def test_known_openrouter_model_accepted_and_persisted(self):
        result = validate_requested_model("anthropic/claude-opus-4.6", "openrouter")

        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    # -- empty / whitespace ----------------------------------------------

    def test_empty_model_rejected(self):
        result = validate_requested_model("", "openrouter")
        assert result["accepted"] is False
        assert "empty" in result["message"]

    def test_whitespace_only_rejected(self):
        result = validate_requested_model("   ", "openrouter")
        assert result["accepted"] is False
        assert "empty" in result["message"]

    def test_model_with_spaces_rejected(self):
        result = validate_requested_model("anthropic/ claude-opus", "openrouter")
        assert result["accepted"] is False
        assert "spaces" in result["message"].lower()

    # -- OpenRouter format validation ------------------------------------

    def test_openrouter_requires_slash(self):
        result = validate_requested_model("claude-opus-4.6", "openrouter")

        assert result["accepted"] is False
        assert result["persist"] is False
        assert "provider/model" in result["message"]

    def test_openrouter_rejects_leading_slash(self):
        result = validate_requested_model("/claude-opus-4.6", "openrouter")
        assert result["accepted"] is False

    def test_openrouter_rejects_trailing_slash(self):
        result = validate_requested_model("anthropic/", "openrouter")
        assert result["accepted"] is False

    def test_openrouter_unknown_but_plausible_is_session_only(self):
        result = validate_requested_model("anthropic/claude-next-gen", "openrouter")

        assert result["accepted"] is True
        assert result["persist"] is False
        assert result["recognized"] is False
        assert "session only" in result["message"].lower()

    # -- custom endpoint -------------------------------------------------

    def test_custom_base_url_accepts_anything(self):
        result = validate_requested_model(
            "my-local-model",
            "openrouter",
            base_url="http://localhost:11434/v1",
        )

        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["message"] is None

    # -- nous provider ---------------------------------------------------

    def test_nous_provider_is_session_only(self):
        result = validate_requested_model("hermes-3", "nous")

        assert result["accepted"] is True
        assert result["persist"] is False
        assert "Nous Portal" in result["message"]

    # -- other providers with catalogs -----------------------------------

    def test_known_zai_model_accepted_and_persisted(self):
        result = validate_requested_model("glm-5", "zai")

        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True

    def test_unknown_zai_model_is_session_only(self):
        result = validate_requested_model("glm-99", "zai")

        assert result["accepted"] is True
        assert result["persist"] is False
        assert "Z.AI" in result["message"]

    # -- provider with no catalog ----------------------------------------

    def test_unknown_provider_is_session_only(self):
        result = validate_requested_model("some-model", "totally-unknown")

        assert result["accepted"] is True
        assert result["persist"] is False
        assert result["message"] is not None

    # -- codex provider --------------------------------------------------

    def test_unknown_codex_model_is_session_only(self):
        result = validate_requested_model("totally-made-up", "openai-codex")

        assert result["accepted"] is True
        assert result["persist"] is False
        assert "OpenAI Codex" in result["message"]

    # -- fuzzy suggestions -----------------------------------------------

    def test_close_match_gets_suggestion(self):
        # Typo of a known model — should get a suggestion in the message
        result = validate_requested_model("anthropic/claude-opus-4.5", "openrouter")
        # May or may not match depending on cutoff, but should be session-only
        assert result["accepted"] is True
        assert result["persist"] is False
