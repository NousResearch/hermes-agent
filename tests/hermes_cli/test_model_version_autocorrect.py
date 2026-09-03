"""Regression tests for issue #101975: version-variant auto-correction.

When a requested model is not in the provider's catalog, the typo
auto-corrector (get_close_matches, cutoff=0.9) can silently substitute a
*different model version* — e.g. `google/gemini-3.8-flash` →
`google/gemini-3.6-flash` — because version digits are string-similar.
The user explicitly asked for a specific version and gets routed to another
model with different pricing, context window, and capabilities.

The fix: a close match that differs from the request ONLY in the values of
version digits (same digit-stripped skeleton, same digit-run lengths) is
treated as a plausible uncataloged/newer version, not a typo. It must NOT be
silently substituted — each validation path's existing unknown-model
handling (soft-accept with a note, or reject with suggestions) applies
instead. True typos (missing dash `gpt5.3-codex`, doubled digit `gpt-5.44`)
change non-digit characters or digit-run lengths and still auto-correct.
"""

from unittest.mock import patch

from hermes_cli.models import validate_requested_model


# -- helper-free fixtures ------------------------------------------------------


class TestVersionVariantAutoCorrection:
    """Version variants must never be silently substituted (#101975)."""

    def test_issue_replay_vertex_uncataloged_version_not_substituted(self):
        """The issue's exact scenario: vertex catalog has gemini-3.6-flash,
        the user requests a newer uncataloged 3.9 flash. The static-catalog
        fallback (gateway path, /models unreachable) must accept the request
        as-is with an unverified note instead of correcting to 3.6."""
        with patch("hermes_cli.models.fetch_api_models", return_value=None):
            result = validate_requested_model(
                "google/gemini-3.9-flash", "vertex"
            )
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result.get("corrected_model") is None
        # The requested name must survive untouched...
        assert "google/gemini-3.9-flash" in result["message"]
        # ...and the older sibling must only appear as a suggestion.
        assert "google/gemini-3.6-flash" in result["message"]
        assert "Auto-corrected" not in result["message"]

    def test_live_listing_version_variant_rejected_with_suggestions(self):
        """Generic live path: a version variant of a listed model must be
        rejected with 'Similar models' guidance, not accepted as the
        sibling."""
        api_models = ["anthropic/claude-sonnet-4.5"]
        with patch("hermes_cli.models.fetch_api_models", return_value=api_models), \
             patch("hermes_cli.models._model_in_provider_catalog", return_value=False):
            result = validate_requested_model(
                "anthropic/claude-sonnet-4.6", "openrouter"
            )
        assert result["accepted"] is False
        assert result.get("corrected_model") is None
        assert "not found" in result["message"]
        assert "anthropic/claude-sonnet-4.5" in result["message"]

    def test_custom_endpoint_version_variant_accepted_as_is(self):
        """Custom-endpoint path: a version variant is soft-accepted with a
        note (hidden/aliased models are common), never substituted."""
        api_models = ["mistral/mistral-large-2"]
        probe_payload = {
            "models": api_models,
            "probed_url": "http://localhost:8000/v1/models",
            "resolved_base_url": "http://localhost:8000/v1",
            "suggested_base_url": None,
            "used_fallback": False,
        }
        with patch("hermes_cli.models.probe_api_models", return_value=probe_payload):
            result = validate_requested_model(
                "mistral/mistral-large-3",
                "custom",
                base_url="http://localhost:8000/v1",
            )
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result.get("corrected_model") is None
        assert "mistral/mistral-large-3" in result["message"]
        assert "mistral/mistral-large-2" in result["message"]
        assert "Auto-corrected" not in result["message"]

    def test_codex_catalog_version_variant_not_corrected(self):
        """openai-codex catalog path: requesting an uncataloged newer codex
        version must not be corrected to an older one."""
        codex_models = ["gpt-5.3-codex", "gpt-5.2-codex"]
        with patch("hermes_cli.models.provider_model_ids", return_value=codex_models):
            result = validate_requested_model("gpt-5.4-codex", "openai-codex")
        assert result["accepted"] is True
        assert result.get("corrected_model") is None
        assert "Auto-corrected" not in result["message"]
        assert "gpt-5.4-codex" in result["message"]

    def test_minimax_catalog_version_variant_not_corrected(self):
        """MiniMax static-catalog path (no /models endpoint): a version
        variant is accepted with a note + suggestion, not corrected."""
        with patch("hermes_cli.models.fetch_api_models", return_value=None):
            result = validate_requested_model("MiniMax-M2.9", "minimax")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result.get("corrected_model") is None
        assert "MiniMax-M2.9" in result["message"]
        assert "MiniMax-M2.7" in result["message"]

    def test_anthropic_native_version_variant_not_corrected(self):
        """Native Anthropic /v1/models path: a version variant falls through
        to the early-access soft-accept with suggestions."""
        with patch(
            "hermes_cli.models._fetch_anthropic_models",
            return_value=["claude-opus-4.6"],
        ):
            result = validate_requested_model("claude-opus-4.5", "anthropic")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result.get("corrected_model") is None
        assert "claude-opus-4.5" in result["message"]
        assert "claude-opus-4.6" in result["message"]
        assert "Auto-corrected" not in result["message"]

    def test_anthropic_messages_version_variant_not_corrected(self):
        """Anthropic Messages API transport path: version variants fall to
        the unverified soft-accept, not substitution."""
        api_models = ["claude-sonnet-4.5"]
        with patch("hermes_cli.models.fetch_api_models", return_value=api_models):
            result = validate_requested_model(
                "claude-sonnet-4.6",
                "groq",
                base_url="https://proxy.example.com",
                api_mode="anthropic_messages",
            )
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result.get("corrected_model") is None

    def test_letter_typo_still_auto_corrects(self):
        """A true typo (non-digit characters differ) must still correct."""
        api_models = ["anthropic/claude-opus-4.6"]
        with patch("hermes_cli.models.fetch_api_models", return_value=api_models), \
             patch("hermes_cli.models._model_in_provider_catalog", return_value=False):
            result = validate_requested_model(
                "anthropic/claude-opuss-4.6", "openrouter"
            )
        assert result["accepted"] is True
        assert result.get("corrected_model") == "anthropic/claude-opus-4.6"
        assert "Auto-corrected" in result["message"]

    def test_doubled_digit_typo_still_auto_corrects(self):
        """A doubled-digit typo changes digit-run lengths (5.44 vs 5.4) and
        must still correct."""
        api_models = ["openai/gpt-5.4"]
        with patch("hermes_cli.models.fetch_api_models", return_value=api_models), \
             patch("hermes_cli.models._model_in_provider_catalog", return_value=False):
            result = validate_requested_model(
                "openai/gpt-5.44", "openrouter"
            )
        assert result["accepted"] is True
        assert result.get("corrected_model") == "openai/gpt-5.4"


class TestSwitchModelKeepsRequestedVersion:
    """Consumer contract: model_switch must not rewrite the user's model
    when validation returns no corrected_model (#101975)."""

    def test_switch_model_keeps_uncataloged_version(self, monkeypatch):
        from hermes_cli.model_switch import switch_model

        monkeypatch.setattr(
            "hermes_cli.models.validate_requested_model",
            lambda *a, **k: {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    "Note: `mistral/mistral-large-3` was not found in this "
                    "custom endpoint's model listing."
                ),
            },
        )
        monkeypatch.setattr(
            "hermes_cli.model_switch.get_model_info", lambda *a, **k: None
        )
        monkeypatch.setattr(
            "hermes_cli.model_switch.get_model_capabilities",
            lambda *a, **k: None,
        )

        result = switch_model(
            raw_input="mistral/mistral-large-3",
            current_provider="custom",
            current_model="mistral/mistral-large-2",
            current_base_url="https://proxy.example.com/v1",
            current_api_key="sk-test",
            explicit_provider="custom",
            user_providers={},
            custom_providers=[],
        )

        assert result.success is True
        assert result.new_model == "mistral/mistral-large-3"
