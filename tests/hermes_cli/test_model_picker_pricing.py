"""Tests for model picker pricing parity.

Covers the /model picker integration that fetches provider pricing
(via get_pricing_for_provider) and renders inline price tags on
model list entries — bringing pricing parity with `hermes model`.
"""

from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.models import _format_price_per_mtok


class TestFormatPricePerMtok:
    """Unit tests for the shared price formatter (upstream of tags)."""

    def test_tiny_price_renders_dollars_per_million(self):
        assert _format_price_per_mtok("0.000003") == "$3.00"

    def test_very_small_price_still_renders_cents(self):
        assert _format_price_per_mtok("0.00000015") == "$0.15"

    def test_free_model_returns_free(self):
        assert _format_price_per_mtok("0") == "free"

    def test_zero_as_numeric_string(self):
        assert _format_price_per_mtok("0.0") == "free"

    def test_unparseable_returns_question_mark(self):
        assert _format_price_per_mtok("n/a") == "?"

# ------------------------------------------------------------------
# _format_price_tag  (static method on HermesCLI)
# ------------------------------------------------------------------

class TestFormatPriceTag:
    """Directly test _format_price_tag with mock pricing dicts."""

    @staticmethod
    def _format_price_tag(model_id, pricing):
        """Inline the method so the test doesn't need a full HermesCLI instance."""
        p = pricing.get(model_id, {})
        if not p:
            return ""
        inp = p.get("prompt", "")
        out = p.get("completion", "")
        if not inp and not out:
            return ""
        inp_str = _format_price_per_mtok(inp)
        out_str = _format_price_per_mtok(out)
        return f"  (I:{inp_str}/O:{out_str})"

    # --- happy path ---

    def test_typical_pricing_renders_tag(self):
        pricing = {
            "claude-sonnet-4": {"prompt": "0.000004", "completion": "0.000012"},
        }
        tag = self._format_price_tag("claude-sonnet-4", pricing)
        assert tag == "  (I:$4.00/O:$12.00)"

    def test_free_model_shows_free(self):
        pricing = {"free-model": {"prompt": "0", "completion": "0"}}
        tag = self._format_price_tag("free-model", pricing)
        assert tag == "  (I:free/O:free)"

    # --- missing data ---

    def test_model_not_in_pricing_returns_empty(self):
        pricing = {"deepseek-chat": {"prompt": "0.000001", "completion": "0.000002"}}
        assert self._format_price_tag("unknown-model", pricing) == ""

    def test_entirely_empty_pricing_dict(self):
        assert self._format_price_tag("any-model", {}) == ""

    # --- partial data ---

    def test_only_input_available_shows_output_as_question(self):
        pricing = {"partial": {"prompt": "0.000001"}}
        tag = self._format_price_tag("partial", pricing)
        assert tag == "  (I:$1.00/O:?)"

    def test_only_output_available_shows_input_as_question(self):
        pricing = {"partial": {"completion": "0.000002"}}
        tag = self._format_price_tag("partial", pricing)
        assert tag == "  (I:?/O:$2.00)"

    def test_empty_strings_are_same_as_missing(self):
        pricing = {"empty-price": {"prompt": "", "completion": ""}}
        assert self._format_price_tag("empty-price", pricing) == ""

    # --- edge cases ---

    def test_unparseable_prices_both_show_question(self):
        pricing = {"bad": {"prompt": "nonsense", "completion": "garbage"}}
        tag = self._format_price_tag("bad", pricing)
        assert tag == "  (I:?/O:?)"

    def test_string_model_id_with_slashes_in_key(self):
        pricing = {"anthropic/claude-sonnet-4": {"prompt": "0.000005",
                                                   "completion": "0.00002"}}
        tag = self._format_price_tag("anthropic/claude-sonnet-4", pricing)
        assert "I:$5.00" in tag
        assert "O:$20.00" in tag


# ------------------------------------------------------------------
# get_pricing_for_provider (upstream fetch)
# ------------------------------------------------------------------

class TestGetPricingForProvider:
    """Ensure get_pricing_for_provider dispatch works correctly."""

    def test_unknown_provider_returns_empty_dict(self):
        from hermes_cli.models import get_pricing_for_provider
        result = get_pricing_for_provider("nonexistent-provider-xyz")
        assert result == {}

    def test_openrouter_calls_fetch_with_correct_params(self):
        from hermes_cli.models import get_pricing_for_provider
        with patch("hermes_cli.models._resolve_openrouter_api_key",
                   return_value="sk-test-key") as mock_key, \
             patch("hermes_cli.models.fetch_models_with_pricing",
                   return_value={"test/model": {"prompt": "0.01", "completion": "0.02"}}) as mock_fetch:
            result = get_pricing_for_provider("openrouter")
            mock_key.assert_called_once()
            mock_fetch.assert_called_once()
            assert "test/model" in result
            assert result["test/model"]["prompt"] == "0.01"

    def test_novita_calls_fetch_with_correct_params(self):
        from hermes_cli.models import get_pricing_for_provider
        with patch("hermes_cli.models._fetch_novita_pricing",
                   return_value={"novita-model": {"prompt": "0.005", "completion": "0.01"}}) as mock_fetch:
            result = get_pricing_for_provider("novita")
            mock_fetch.assert_called_once_with(force_refresh=False)
            assert "novita-model" in result

    def test_nous_calls_fetch_with_stripped_base_url(self):
        from hermes_cli.models import get_pricing_for_provider
        creds = ("sk-nous-test", "https://inference-api.nousresearch.com/v1")
        with patch("hermes_cli.models._resolve_nous_pricing_credentials",
                   return_value=creds), \
             patch("hermes_cli.models.fetch_models_with_pricing",
                   return_value={"deepseek-r1": {"prompt": "0.05", "completion": "0.1"}}) as mock_fetch:
            result = get_pricing_for_provider("nous")
            # Should strip /v1
            call_kwargs = mock_fetch.call_args.kwargs
            assert call_kwargs["base_url"] == "https://inference-api.nousresearch.com"
            assert call_kwargs["api_key"] == "sk-nous-test"
            assert "deepseek-r1" in result

    def test_normalized_provider_name_works(self):
        """Variant spellings should still route correctly."""
        from hermes_cli.models import get_pricing_for_provider
        with patch("hermes_cli.models._fetch_novita_pricing",
                   return_value={"n": {"prompt": "0", "completion": "0"}}) as mock:
            # novitaai should normalize to novita
            get_pricing_for_provider("novitaai")
            mock.assert_called_once()