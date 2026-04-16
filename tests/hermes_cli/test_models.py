"""Tests for the hermes_cli models module."""

from hermes_cli.models import OPENROUTER_MODELS, menu_labels, model_ids, detect_provider_for_model


class TestModelIds:
    def test_returns_non_empty_list(self):
        ids = model_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_ids_match_models_list(self):
        ids = model_ids()
        expected = [mid for mid, _ in OPENROUTER_MODELS]
        assert ids == expected

    def test_all_ids_contain_provider_slash(self):
        """Model IDs should follow the provider/model format."""
        for mid in model_ids():
            assert "/" in mid, f"Model ID '{mid}' missing provider/ prefix"

    def test_no_duplicate_ids(self):
        ids = model_ids()
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"


class TestMenuLabels:
    def test_same_length_as_model_ids(self):
        assert len(menu_labels()) == len(model_ids())

    def test_first_label_marked_recommended(self):
        labels = menu_labels()
        assert "recommended" in labels[0].lower()

    def test_each_label_contains_its_model_id(self):
        for label, mid in zip(menu_labels(), model_ids()):
            assert mid in label, f"Label '{label}' doesn't contain model ID '{mid}'"

    def test_non_recommended_labels_have_no_tag(self):
        """Only the first model should have (recommended)."""
        labels = menu_labels()
        for label in labels[1:]:
            assert "recommended" not in label.lower(), f"Unexpected 'recommended' in '{label}'"


class TestOpenRouterModels:
    def test_structure_is_list_of_tuples(self):
        for entry in OPENROUTER_MODELS:
            assert isinstance(entry, tuple) and len(entry) == 2
            mid, desc = entry
            assert isinstance(mid, str) and len(mid) > 0
            assert isinstance(desc, str)

    def test_at_least_5_models(self):
        """Sanity check that the models list hasn't been accidentally truncated."""
        assert len(OPENROUTER_MODELS) >= 5


class TestFindOpenrouterSlug:
    def test_exact_match(self):
        from hermes_cli.models import _find_openrouter_slug
        assert _find_openrouter_slug("anthropic/claude-opus-4.6") == "anthropic/claude-opus-4.6"

    def test_bare_name_match(self):
        from hermes_cli.models import _find_openrouter_slug
        result = _find_openrouter_slug("claude-opus-4.6")
        assert result == "anthropic/claude-opus-4.6"

    def test_case_insensitive(self):
        from hermes_cli.models import _find_openrouter_slug
        result = _find_openrouter_slug("Anthropic/Claude-Opus-4.6")
        assert result is not None

    def test_unknown_returns_none(self):
        from hermes_cli.models import _find_openrouter_slug
        assert _find_openrouter_slug("totally-fake-model-xyz") is None


class TestDetectProviderForModel:
    def test_anthropic_model_detected(self):
        """claude-opus-4-6 should resolve to anthropic provider."""
        result = detect_provider_for_model("claude-opus-4-6", "openai-codex")
        assert result is not None
        assert result[0] == "anthropic"

    def test_deepseek_model_detected(self):
        """deepseek-chat should resolve to deepseek provider."""
        result = detect_provider_for_model("deepseek-chat", "openai-codex")
        assert result is not None
        # Provider is deepseek (direct) or openrouter (fallback) depending on creds
        assert result[0] in ("deepseek", "openrouter")

    def test_current_provider_model_returns_none(self):
        """Models belonging to the current provider should not trigger a switch."""
        assert detect_provider_for_model("gpt-5.3-codex", "openai-codex") is None

    def test_openrouter_slug_match(self):
        """Models in the OpenRouter catalog should be found."""
        result = detect_provider_for_model("anthropic/claude-opus-4.6", "openai-codex")
        assert result is not None
        assert result[0] == "openrouter"
        assert result[1] == "anthropic/claude-opus-4.6"

    def test_bare_name_gets_openrouter_slug(self, monkeypatch):
        for env_var in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_TOKEN",
            "CLAUDE_CODE_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
        ):
            monkeypatch.delenv(env_var, raising=False)
        """Bare model names should get mapped to full OpenRouter slugs."""
        result = detect_provider_for_model("claude-opus-4.6", "openai-codex")
        assert result is not None
        # Should find it on OpenRouter with full slug
        assert result[1] == "anthropic/claude-opus-4.6"

    def test_unknown_model_returns_none(self):
        """Completely unknown model names should return None."""
        assert detect_provider_for_model("nonexistent-model-xyz", "openai-codex") is None

    def test_aggregator_not_suggested(self):
        """nous/openrouter should never be auto-suggested as target provider."""
        result = detect_provider_for_model("claude-opus-4-6", "openai-codex")
        assert result is not None
        assert result[0] not in ("nous",)  # nous has claude models but shouldn't be suggested
