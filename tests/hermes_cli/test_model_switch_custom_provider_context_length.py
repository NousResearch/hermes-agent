"""Regression tests for /model switch ignoring custom_providers per-model context_length.

Bug (#15779, April 2026): when a gateway session switched to a named custom
provider via ``/model``, Hermes ignored ``custom_providers[].models.<model>.
context_length`` and fell back to 128 K.  Two paths were broken:

1. *Display* — ``resolve_display_context_length()`` called
   ``get_model_context_length()`` without ``config_context_length``, so the
   custom_providers value never reached the resolver.
2. *Runtime* — ``run_agent.switch_model()`` passed the startup
   ``_config_context_length`` (set for the original model) to
   ``get_model_context_length()`` instead of looking up the new target.

Fix: the new ``_lookup_custom_provider_context_length(model, base_url)``
helper in ``hermes_cli.model_switch`` reads ``custom_providers`` config and is
called by both paths.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.model_switch import (
    _lookup_custom_provider_context_length,
    resolve_display_context_length,
)

# ---------------------------------------------------------------------------
# Shared fixture: a custom_providers list with a per-model context_length
# ---------------------------------------------------------------------------

_FAKE_PROVIDERS = [
    {
        "name": "my-custom-endpoint",
        "base_url": "https://example.invalid/v1",
        "api_key": "sk-test",
        "models": {
            "gpt-5.5": {"context_length": 1_050_000},
            "gpt-4o": {"context_length": 128_000},
        },
    },
    {
        "name": "another-endpoint",
        "base_url": "https://other.invalid/v1",
        "api_key": "sk-other",
        "models": {},
    },
]


def _patch_providers(providers=_FAKE_PROVIDERS):
    return patch(
        "hermes_cli.config.get_compatible_custom_providers",
        return_value=providers,
    )


# ---------------------------------------------------------------------------
# _lookup_custom_provider_context_length
# ---------------------------------------------------------------------------

class TestLookupCustomProviderContextLength:
    def test_returns_configured_context_length(self):
        """Positive: matching (model, base_url) pair returns configured value."""
        with _patch_providers():
            result = _lookup_custom_provider_context_length(
                "gpt-5.5", "https://example.invalid/v1"
            )
        assert result == 1_050_000

    def test_trailing_slash_ignored_on_base_url(self):
        """base_url trailing slash must not prevent a match."""
        with _patch_providers():
            result = _lookup_custom_provider_context_length(
                "gpt-5.5", "https://example.invalid/v1/"
            )
        assert result == 1_050_000

    def test_model_not_in_provider_returns_none(self):
        """Negative: model name not listed under the matching provider → None."""
        with _patch_providers():
            result = _lookup_custom_provider_context_length(
                "unknown-model", "https://example.invalid/v1"
            )
        assert result is None

    def test_base_url_mismatch_returns_none(self):
        """Negative: base_url does not match any provider → None."""
        with _patch_providers():
            result = _lookup_custom_provider_context_length(
                "gpt-5.5", "https://totally-different.invalid/v1"
            )
        assert result is None

    def test_empty_base_url_returns_none(self):
        """Edge: empty base_url → None without querying config."""
        with _patch_providers():
            result = _lookup_custom_provider_context_length("gpt-5.5", "")
        assert result is None

    def test_non_integer_context_length_returns_none(self):
        """Edge: non-int context_length in config (e.g. '1M') → None."""
        bad_providers = [
            {
                "base_url": "https://bad.invalid/v1",
                "models": {"model-x": {"context_length": "1M"}},
            }
        ]
        with _patch_providers(bad_providers):
            result = _lookup_custom_provider_context_length(
                "model-x", "https://bad.invalid/v1"
            )
        assert result is None

    def test_config_load_error_returns_none(self):
        """Edge: config loading raises → None (never propagates exception)."""
        with patch(
            "hermes_cli.config.get_compatible_custom_providers",
            side_effect=RuntimeError("disk error"),
        ):
            result = _lookup_custom_provider_context_length(
                "gpt-5.5", "https://example.invalid/v1"
            )
        assert result is None

    def test_empty_providers_list_returns_none(self):
        """Edge: custom_providers list is empty → None."""
        with _patch_providers([]):
            result = _lookup_custom_provider_context_length(
                "gpt-5.5", "https://example.invalid/v1"
            )
        assert result is None

    def test_provider_without_models_key_returns_none(self):
        """Edge: matching provider entry has no 'models' key → None."""
        providers = [{"base_url": "https://example.invalid/v1", "api_key": "sk"}]
        with _patch_providers(providers):
            result = _lookup_custom_provider_context_length(
                "gpt-5.5", "https://example.invalid/v1"
            )
        assert result is None

    def test_second_matching_model_in_same_provider(self):
        """Positive: a second model in the same provider entry resolves correctly."""
        with _patch_providers():
            result = _lookup_custom_provider_context_length(
                "gpt-4o", "https://example.invalid/v1"
            )
        assert result == 128_000


# ---------------------------------------------------------------------------
# resolve_display_context_length — custom_providers integration
# ---------------------------------------------------------------------------

class TestResolveDisplayContextLengthCustomProvider:
    def test_custom_provider_context_length_used_in_display(self):
        """Display must show custom_providers context_length, not 128K default."""
        with (
            _patch_providers(),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=1_050_000,
            ) as mock_gmcl,
        ):
            ctx = resolve_display_context_length(
                "gpt-5.5",
                "custom",
                base_url="https://example.invalid/v1",
                api_key="sk-test",
            )
        assert ctx == 1_050_000
        # Verify config_context_length was forwarded to the resolver
        _call_kwargs = mock_gmcl.call_args
        assert _call_kwargs.kwargs.get("config_context_length") == 1_050_000

    def test_no_custom_provider_entry_falls_through_to_resolver(self):
        """When no custom_providers match, resolver still runs without override."""
        with (
            _patch_providers([]),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=32_768,
            ) as mock_gmcl,
        ):
            ctx = resolve_display_context_length(
                "unknown-model",
                "some-provider",
                base_url="https://no-match.invalid/v1",
            )
        assert ctx == 32_768
        _call_kwargs = mock_gmcl.call_args
        assert _call_kwargs.kwargs.get("config_context_length") is None

    def test_custom_provider_value_wins_over_model_info(self):
        """custom_providers context_length must override ModelInfo.context_window."""
        fake_mi = SimpleNamespace(context_window=128_000)
        with (
            _patch_providers(),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=1_050_000,
            ),
        ):
            ctx = resolve_display_context_length(
                "gpt-5.5",
                "custom",
                base_url="https://example.invalid/v1",
                model_info=fake_mi,
            )
        assert ctx == 1_050_000
