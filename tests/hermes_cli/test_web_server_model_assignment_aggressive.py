"""Aggressive failure-mode battery for the C9 model-assignment helpers.

Covers the failure matrix from the R1 consensus test plan (T5/T6) for
``_normalize_main_model_assignment`` and ``_apply_main_model_assignment``:
empty/invalid provider+model strings, the stray-vendor-prefix fallback, named
custom-provider preservation, and the endpoint-secret/base_url lifecycle
(same-provider re-pick vs provider switch).

Assertions are grounded in behavior verified against the pre-extraction
functions at the pinned base (aec331899e).
"""

from unittest.mock import patch

from hermes_cli.web_server_model_assignment import (
    _apply_main_model_assignment,
    _normalize_main_model_assignment,
)


def _norm(provider, model, config):
    """Call _normalize with load_config stubbed to ``config``."""
    with patch(
        "hermes_cli.web_server_model_assignment.load_config", return_value=config
    ):
        return _normalize_main_model_assignment(provider, model)


# ---------------------------------------------------------------------------
# T5 — _normalize_main_model_assignment failure matrix
# ---------------------------------------------------------------------------


class TestEmptyAndInvalidInputs:
    def test_empty_provider_and_model_are_preserved(self):
        assert _norm("", "", {}) == ("", "")

    def test_whitespace_only_inputs_are_stripped_to_empty(self):
        assert _norm("  ", "\t ", {}) == ("", "")

    def test_empty_provider_with_slash_model_does_not_trip_vendor_fallback(self):
        # canonical of "" is "openrouter" (KNOWN) -> no fallback reassignment.
        assert _norm("", "foo/bar", {}) == ("", "foo/bar")

    def test_none_provider_is_tolerated(self):
        assert _norm(None, None, {}) == ("", "")


class TestStrayVendorPrefixFallback:
    def test_unknown_vendor_with_slash_model_falls_back_to_openrouter(self):
        assert _norm("unconfigured-vendor", "foo/bar", {}) == ("openrouter", "foo/bar")

    def test_current_aggregator_provider_is_preserved(self):
        config = {"model": {"provider": "openrouter"}}
        assert _norm("unconfigured-vendor", "foo/bar", config) == (
            "openrouter",
            "foo/bar",
        )

    def test_current_non_aggregator_does_not_hijack_fallback(self):
        # deepseek is a native provider, not an aggregator: the fallback must
        # NOT reuse it as the routing provider for a vendor-prefixed slug.
        config = {"model": {"provider": "deepseek"}}
        assert _norm("unconfigured-vendor", "foo/bar", config) == (
            "openrouter",
            "foo/bar",
        )

    def test_known_vendor_without_slash_model_is_untouched(self):
        # "/" not in model -> the analytics-vendor branch never fires.
        assert _norm("anthropic", "claude-opus-4-6", {}) == ("anthropic", "claude-opus-4-6")


class TestNamedCustomProviderProtection:
    def test_unresolved_named_custom_slug_with_slash_model_is_preserved(self):
        assert _norm("custom:litellm", "ollama/glm-5.2", {}) == (
            "custom:litellm",
            "ollama/glm-5.2",
        )

    def test_bare_custom_bucket_is_preserved(self):
        assert _norm("custom", "my-model", {}) == ("custom", "my-model")

    def test_custom_provider_name_canonicalizes_to_durable_slug(self):
        config = {
            "custom_providers": [
                {"name": "US Azure", "base_url": "http://localhost:18025/v1"}
            ]
        }
        assert _norm("US Azure", "vendor/model-a", config) == (
            "custom:us-azure",
            "vendor/model-a",
        )
        assert _norm("custom:us-azure", "vendor/model-a", config) == (
            "custom:us-azure",
            "vendor/model-a",
        )


class TestKnownProviderModelNormalization:
    def test_known_native_provider_normalizes_vendor_prefixed_model(self):
        assert _norm("anthropic", "anthropic/claude-opus-4.6", {}) == (
            "anthropic",
            "claude-opus-4-6",
        )

    def test_user_declared_provider_keeps_bare_slug(self):
        config = {
            "providers": {"commandcode": {"base_url": "http://localhost:55990/v1"}}
        }
        assert _norm("commandcode", "vendor/model-a", config) == (
            "commandcode",
            "vendor/model-a",
        )
        # Slash-bearing model id under a declared provider is preserved too.
        assert _norm("commandcode", "commandcode/model-x", config) == (
            "commandcode",
            "commandcode/model-x",
        )


class TestLoadConfigFailure:
    def test_load_config_raising_falls_back_to_empty_config(self):
        with patch(
            "hermes_cli.web_server_model_assignment.load_config",
            side_effect=RuntimeError("config exploded"),
        ):
            assert _normalize_main_model_assignment(
                "unconfigured-vendor", "foo/bar"
            ) == ("openrouter", "foo/bar")

    def test_load_config_raising_with_empty_inputs(self):
        with patch(
            "hermes_cli.web_server_model_assignment.load_config",
            side_effect=RuntimeError("config exploded"),
        ):
            assert _normalize_main_model_assignment("", "") == ("", "")


# ---------------------------------------------------------------------------
# T6 — _apply_main_model_assignment failure matrix
# ---------------------------------------------------------------------------


class TestBaseUrlLifecycle:
    def test_same_provider_repick_preserves_base_url_and_returns_same_dict(self):
        cfg = {"provider": "openrouter", "default": "old", "base_url": "http://x"}
        result = _apply_main_model_assignment(cfg, "openrouter", "new")
        assert result is cfg
        assert result["base_url"] == "http://x"
        assert result["default"] == "new"

    def test_provider_switch_clears_stale_base_url(self):
        cfg = {"provider": "openrouter", "default": "old", "base_url": "http://x"}
        result = _apply_main_model_assignment(cfg, "anthropic", "m")
        assert result["base_url"] == ""

    def test_explicit_base_url_is_always_persisted(self):
        cfg = {"provider": "openrouter"}
        result = _apply_main_model_assignment(
            cfg, "openrouter", "m", base_url="  http://new/v1  "
        )
        assert result["base_url"] == "http://new/v1"

    def test_explicit_base_url_survives_provider_switch(self):
        cfg = {"provider": "openrouter", "base_url": "http://old"}
        result = _apply_main_model_assignment(
            cfg, "anthropic", "m", base_url="http://new"
        )
        assert result["base_url"] == "http://new"

    def test_provider_comparison_is_case_insensitive(self):
        cfg = {"provider": "OPENROUTER", "base_url": "http://x"}
        result = _apply_main_model_assignment(cfg, "openrouter", "m")
        assert result["base_url"] == "http://x"


class TestEndpointSecretLifecycle:
    def test_explicit_api_key_persisted_and_api_alias_popped(self):
        cfg = {"provider": "openrouter", "api": "old-secret"}
        result = _apply_main_model_assignment(cfg, "openrouter", "m", api_key=" k ")
        assert result["api_key"] == "k"
        assert "api" not in result

    def test_provider_switch_clears_stale_api_key_and_api_mode(self):
        cfg = {"provider": "openrouter", "api_key": "secret", "api_mode": "env"}
        result = _apply_main_model_assignment(cfg, "anthropic", "m")
        assert "api_key" not in result
        assert "api_mode" not in result

    def test_provider_switch_clears_legacy_api_alias(self):
        cfg = {"provider": "openrouter", "api": "old-secret", "api_mode": "env"}
        result = _apply_main_model_assignment(cfg, "anthropic", "m")
        assert "api" not in result
        assert "api_mode" not in result

    def test_same_provider_repick_preserves_api_key(self):
        cfg = {"provider": "openrouter", "api_key": "secret"}
        result = _apply_main_model_assignment(cfg, "openrouter", "m")
        assert result["api_key"] == "secret"

    def test_switch_with_explicit_base_url_still_clears_stale_key(self):
        cfg = {"provider": "openrouter", "api_key": "secret", "api_mode": "env"}
        result = _apply_main_model_assignment(
            cfg, "anthropic", "m", base_url="http://new"
        )
        assert "api_key" not in result
        assert "api_mode" not in result
        assert result["base_url"] == "http://new"


class TestContextLengthAndInputCoercion:
    def test_context_length_always_dropped(self):
        cfg = {"provider": "openrouter", "context_length": 128000}
        result = _apply_main_model_assignment(cfg, "openrouter", "m")
        assert "context_length" not in result

    def test_none_input_coerced_to_fresh_dict(self):
        result = _apply_main_model_assignment(None, "openrouter", "m")
        assert isinstance(result, dict)
        assert result["provider"] == "openrouter"
        assert result["default"] == "m"

    def test_non_dict_input_coerced_to_fresh_dict(self):
        result = _apply_main_model_assignment("not-a-dict", "anthropic", "m")
        assert isinstance(result, dict)
        assert result["provider"] == "anthropic"

    def test_dict_input_returns_same_object(self):
        cfg = {"provider": "openrouter", "default": "old"}
        result = _apply_main_model_assignment(cfg, "openrouter", "m")
        assert result is cfg
