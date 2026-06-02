"""Regression tests for custom_providers per-model context_length resolution.

Covers the fix for #15779 — mid-session /model switch to a named custom
provider must honor ``custom_providers[].models.<id>.context_length`` the
same way startup already does.
"""
from __future__ import annotations

from unittest.mock import patch

from hermes_cli.config import get_custom_provider_context_length


class TestGetCustomProviderContextLength:
    def test_returns_override_for_matching_entry(self):
        custom = [
            {
                "name": "my-endpoint",
                "base_url": "https://example.invalid/v1",
                "models": {"gpt-5.5": {"context_length": 1_050_000}},
            }
        ]
        assert (
            get_custom_provider_context_length(
                "gpt-5.5", "https://example.invalid/v1", custom
            )
            == 1_050_000
        )

    def test_trailing_slash_insensitive(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1/",
                "models": {"m": {"context_length": 500_000}},
            }
        ]
        # config has trailing slash, runtime doesn't — must match
        assert (
            get_custom_provider_context_length(
                "m", "https://example.invalid/v1", custom
            )
            == 500_000
        )
        # and the reverse
        custom2 = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {"m": {"context_length": 500_000}},
            }
        ]
        assert (
            get_custom_provider_context_length(
                "m", "https://example.invalid/v1/", custom2
            )
            == 500_000
        )

    def test_returns_none_when_url_does_not_match(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {"m": {"context_length": 400_000}},
            }
        ]
        assert (
            get_custom_provider_context_length(
                "m", "https://other.invalid/v1", custom
            )
            is None
        )

    def test_returns_none_when_model_does_not_match(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {"gpt-5.5": {"context_length": 400_000}},
            }
        ]
        assert (
            get_custom_provider_context_length(
                "different-model", "https://example.invalid/v1", custom
            )
            is None
        )

    def test_returns_none_for_string_value(self):
        """'256K' string is not a valid int — skip silently.

        (The inline startup path still emits a user-visible warning; the
        helper itself returns None so downstream fallbacks can run.)
        """
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {"m": {"context_length": "256K"}},
            }
        ]
        assert (
            get_custom_provider_context_length(
                "m", "https://example.invalid/v1", custom
            )
            is None
        )

    def test_returns_none_for_zero_or_negative(self):
        for bad in (0, -1, -100):
            custom = [
                {
                    "base_url": "https://example.invalid/v1",
                    "models": {"m": {"context_length": bad}},
                }
            ]
            assert (
                get_custom_provider_context_length(
                    "m", "https://example.invalid/v1", custom
                )
                is None
            ), f"value {bad!r} should be rejected"

    def test_empty_inputs_return_none(self):
        assert get_custom_provider_context_length("", "http://x", [{"base_url": "http://x", "models": {"": {"context_length": 1}}}]) is None
        assert get_custom_provider_context_length("m", "", [{"base_url": "", "models": {"m": {"context_length": 1}}}]) is None
        assert get_custom_provider_context_length("m", "http://x", None) is None
        assert get_custom_provider_context_length("m", "http://x", []) is None

    def test_ignores_non_dict_entries(self):
        """Malformed entries must not crash the lookup."""
        custom = [
            "not a dict",
            None,
            {"base_url": "https://example.invalid/v1", "models": "not a dict"},
            {"base_url": "https://example.invalid/v1", "models": {"m": "not a dict"}},
            {
                "base_url": "https://example.invalid/v1",
                "models": {"m": {"context_length": 400_000}},
            },
        ]
        assert (
            get_custom_provider_context_length(
                "m", "https://example.invalid/v1", custom
            )
            == 400_000
        )


class TestGetModelContextLengthHonorsOverride:
    """agent.model_metadata.get_model_context_length must honor the
    custom_providers override at step 0b — before any probe, cache hit,
    or models.dev lookup can override it.
    """

    def _mock_all_probes(self):
        """Context manager that disables every downstream resolution step."""
        from agent import model_metadata as _mm
        return [
            patch.object(_mm, "get_cached_context_length", return_value=None),
            patch.object(_mm, "fetch_endpoint_model_metadata", return_value={}),
            patch.object(_mm, "fetch_model_metadata", return_value={}),
            patch.object(_mm, "is_local_endpoint", return_value=False),
            patch.object(_mm, "_is_known_provider_base_url", return_value=False),
        ]

    def test_custom_providers_override_wins_over_default_fallback(self):
        from agent.model_metadata import get_model_context_length
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {"gpt-5.5": {"context_length": 1_050_000}},
            }
        ]
        patches = self._mock_all_probes()
        for p in patches:
            p.start()
        try:
            ctx = get_model_context_length(
                "gpt-5.5",
                base_url="https://example.invalid/v1",
                provider="custom",
                custom_providers=custom,
            )
        finally:
            for p in patches:
                p.stop()
        assert ctx == 1_050_000

    def test_explicit_config_context_length_still_wins(self):
        """Top-level model.context_length (step 0) outranks custom_providers (step 0b).

        Users who set both should see the top-level value — that's the
        documented precedence and matches the long-standing step-0 behavior.
        """
        from agent.model_metadata import get_model_context_length
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {"m": {"context_length": 1_050_000}},
            }
        ]
        ctx = get_model_context_length(
            "m",
            base_url="https://example.invalid/v1",
            provider="custom",
            config_context_length=500_000,  # explicit top-level wins
            custom_providers=custom,
        )
        assert ctx == 500_000

    def test_no_override_falls_through_to_default(self):
        """With custom_providers=None and all probes disabled, resolver
        returns DEFAULT_FALLBACK_CONTEXT (256K after the stepdown bump).
        """
        from agent.model_metadata import get_model_context_length, DEFAULT_FALLBACK_CONTEXT
        patches = self._mock_all_probes()
        for p in patches:
            p.start()
        try:
            ctx = get_model_context_length(
                "unknown-model",
                base_url="https://example.invalid/v1",
                provider="custom",
                custom_providers=None,
            )
        finally:
            for p in patches:
                p.stop()
        assert ctx == DEFAULT_FALLBACK_CONTEXT


class TestContextProbeTiers:
    def test_256k_is_top_tier_and_default(self):
        """The stepdown probe starts at 256K and 256K is the new default."""
        from agent.model_metadata import CONTEXT_PROBE_TIERS, DEFAULT_FALLBACK_CONTEXT

        assert CONTEXT_PROBE_TIERS[0] == 256_000
        assert DEFAULT_FALLBACK_CONTEXT == 256_000
        # Tiers still descend monotonically
        for a, b in zip(CONTEXT_PROBE_TIERS, CONTEXT_PROBE_TIERS[1:]):
            assert a > b, f"tiers must strictly descend, got {a} then {b}"
        # 128K is still a tier (users relying on it probe-down get there)
        assert 128_000 in CONTEXT_PROBE_TIERS
