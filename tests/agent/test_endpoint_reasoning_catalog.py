"""Tests for endpoint_model_supports_reasoning() in agent/model_metadata.py.

Covers the supported_parameters extraction added to fetch_endpoint_model_metadata
and the model-lookup logic in endpoint_model_supports_reasoning.

Lookup is exact-match only: no substring fallback, no single-entry shortcut.
A model not listed by its exact name returns None.
"""

from unittest.mock import patch

from agent.model_metadata import endpoint_model_supports_reasoning


def _make_metadata(model_id, supported_parameters=None, context_length=8192):
    """Build a minimal metadata dict as fetch_endpoint_model_metadata returns."""
    entry = {"context_length": context_length}
    if supported_parameters is not None:
        entry["supported_parameters"] = supported_parameters
    return {model_id: entry}


# ============================================================
# endpoint_model_supports_reasoning
# ============================================================


class TestEndpointModelSupportsReasoning:

    def test_returns_true_when_reasoning_in_supported_parameters(self):
        """Exact model match with reasoning in supported_parameters -> True."""
        metadata = _make_metadata("claude-sonnet", ["temperature", "reasoning", "top_p"])
        with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
            result = endpoint_model_supports_reasoning(
                "claude-sonnet", "http://127.0.0.1:8977/v1"
            )
        assert result is True

    def test_returns_false_when_reasoning_absent_from_supported_parameters(self):
        """Model listed but reasoning not in supported_parameters -> False."""
        metadata = _make_metadata("gpt-4o", ["temperature", "top_p", "seed"])
        with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
            result = endpoint_model_supports_reasoning(
                "gpt-4o", "http://127.0.0.1:8977/v1"
            )
        assert result is False

    def test_returns_none_when_catalog_empty(self):
        """Empty catalog -> None."""
        with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}):
            result = endpoint_model_supports_reasoning(
                "sonnet", "http://127.0.0.1:8977/v1"
            )
        assert result is None

    def test_returns_none_when_model_not_listed_by_exact_name(self):
        """Catalog present but model name is not an exact key -> None.
        No fuzzy/substring/shortcut matching."""
        metadata = {
            "claude-sonnet-4-6": {"context_length": 200000, "supported_parameters": ["reasoning"]},
            "claude-haiku-4": {"context_length": 48000, "supported_parameters": ["reasoning"]},
        }
        with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
            result = endpoint_model_supports_reasoning(
                "gpt-4o", "http://127.0.0.1:8977/v1"
            )
        assert result is None

    def test_substring_of_catalog_key_does_not_match(self):
        """A model name that is a substring of a listed key still returns None.
        Callers must use the exact same name the endpoint lists."""
        metadata = _make_metadata("claude-sonnet-4-6", ["reasoning"])
        with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
            # "sonnet" is a substring of "claude-sonnet-4-6" but is not the same key
            result = endpoint_model_supports_reasoning(
                "sonnet", "http://127.0.0.1:8977/v1"
            )
        assert result is None

    def test_single_entry_catalog_requires_exact_name(self):
        """Even a one-model catalog must not match a request for a different model name."""
        metadata = _make_metadata("claude-sonnet-4-6", ["reasoning"])
        with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
            result = endpoint_model_supports_reasoning(
                "sonnet", "http://127.0.0.1:8977/v1"
            )
        assert result is None

    def test_returns_none_when_supported_parameters_missing(self):
        """Model found by exact name but entry has no supported_parameters key -> None."""
        metadata = {"claude-sonnet": {"context_length": 200000}}
        with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
            result = endpoint_model_supports_reasoning(
                "claude-sonnet", "http://127.0.0.1:8977/v1"
            )
        assert result is None

    def test_returns_none_when_supported_parameters_not_a_list(self):
        """supported_parameters present but not a list -> None (malformed response)."""
        metadata = {"claude-sonnet": {"context_length": 8192, "supported_parameters": "reasoning"}}
        with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
            result = endpoint_model_supports_reasoning(
                "claude-sonnet", "http://127.0.0.1:8977/v1"
            )
        assert result is None


# ============================================================
# supported_parameters extraction via the public lookup function
# (exercises the extraction indirectly through pre-populated metadata)
# ============================================================


class TestSupportedParametersExtractionViaLookup:
    """Verify the supported_parameters field is respected in the model lookup.
    These tests inject already-populated metadata (bypassing the HTTP fetch)
    to isolate the extraction/lookup logic from network calls.
    """

    def test_reasoning_detected_from_pre_populated_metadata(self):
        """When the cache already has supported_parameters populated with
        reasoning, endpoint_model_supports_reasoning returns True."""
        metadata = {
            "claude-sonnet": {
                "context_length": 200000,
                "supported_parameters": ["temperature", "reasoning", "top_p"],
            }
        }
        with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
            assert endpoint_model_supports_reasoning(
                "claude-sonnet", "http://127.0.0.1:8977/v1"
            ) is True

    def test_reasoning_absent_means_false(self):
        """supported_parameters present but without reasoning -> False."""
        metadata = {
            "gpt-4o": {
                "context_length": 128000,
                "supported_parameters": ["temperature", "top_p", "seed"],
            }
        }
        with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
            assert endpoint_model_supports_reasoning(
                "gpt-4o", "http://127.0.0.1:8977/v1"
            ) is False

    def test_non_list_supported_parameters_returns_none(self):
        """supported_parameters present but not a list -> None (malformed)."""
        metadata = {
            "bad-model": {
                "context_length": 8192,
                "supported_parameters": "reasoning",  # string, not list
            }
        }
        with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
            assert endpoint_model_supports_reasoning(
                "bad-model", "http://127.0.0.1:8977/v1"
            ) is None
