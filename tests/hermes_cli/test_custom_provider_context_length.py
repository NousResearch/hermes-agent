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

    # ------------------------------------------------------------------
    # Slug-normalisation fallback (LM Studio "publisher/slug" runtime ids)
    # ------------------------------------------------------------------

    def test_slug_fallback_prefixed_runtime_matches_bare_config_key(self):
        """Runtime model is 'lmstudio/phi-4'; config key is bare 'phi-4'.

        This is the primary LM Studio scenario: probe_lmstudio_models returns
        'publisher/slug' keys; users typically configure only the bare slug.
        """
        custom = [
            {
                "base_url": "http://localhost:1234/v1",
                "models": {"phi-4": {"context_length": 131_072}},
            }
        ]
        assert (
            get_custom_provider_context_length(
                "lmstudio/phi-4", "http://localhost:1234/v1", custom
            )
            == 131_072
        )

    def test_exact_match_wins_over_slug_fallback(self):
        """When the runtime id has a '/' AND appears as a literal key, exact
        match must be returned without consulting the slug fallback.
        """
        custom = [
            {
                "base_url": "http://localhost:1234/v1",
                "models": {
                    "pub/phi-4": {"context_length": 99_999},  # exact key
                    "phi-4": {"context_length": 131_072},      # slug-only key
                },
            }
        ]
        # Exact match wins — returns the exact-key value, not the slug value.
        assert (
            get_custom_provider_context_length(
                "pub/phi-4", "http://localhost:1234/v1", custom
            )
            == 99_999
        )

    def test_slug_fallback_and_exact_match_precedence_with_prefixed_keys(self):
        """Slug fallback for a lone bare key; exact match wins over slug.

        Part 1: runtime 'acme/m' with only bare-slug 'm' configured resolves
        via the slug fallback.  Part 2: when two prefixed keys ('acme/m',
        'bigco/m') exist, each prefixed runtime id hits its own exact key and
        the slug path is never taken — so the result is deterministic and does
        NOT depend on dict insertion order.

        Note on a genuine collision (two *different* prefixed keys, runtime id
        matches neither exactly, both share a bare slug): the function returns
        the first matching entry in dict-iteration order.  That case is
        intentionally not asserted here — it is order-dependent by nature and
        only arises from a self-contradictory config (two publishers, same
        slug, same endpoint).  Asserting on it would bake a brittle
        ordering expectation into the suite.
        """
        # Only one bare slug key — no ambiguity.
        custom_single = [
            {
                "base_url": "http://localhost:1234/v1",
                "models": {"m": {"context_length": 8_192}},
            }
        ]
        assert (
            get_custom_provider_context_length(
                "acme/m", "http://localhost:1234/v1", custom_single
            )
            == 8_192
        )

        # Two prefixed keys, same bare slug 'm' — exact match on 'acme/m'
        # returns the correct entry; 'bigco/m' is not reached.
        custom_two_prefixed = [
            {
                "base_url": "http://localhost:1234/v1",
                "models": {
                    "acme/m": {"context_length": 8_192},
                    "bigco/m": {"context_length": 32_768},
                },
            }
        ]
        assert (
            get_custom_provider_context_length(
                "acme/m", "http://localhost:1234/v1", custom_two_prefixed
            )
            == 8_192  # exact hit, not slug path
        )
        assert (
            get_custom_provider_context_length(
                "bigco/m", "http://localhost:1234/v1", custom_two_prefixed
            )
            == 32_768  # exact hit
        )

    def test_reverse_direction_not_handled(self):
        """Bare runtime id 'm' does NOT match a config key 'pub/m'.

        The fix is one-directional: runtime has '/', config key is bare slug.
        The reverse (runtime bare, config key prefixed) is not a real scenario
        because LM Studio always returns prefixed ids from its native API and
        normalize_model_for_provider does not strip the publisher prefix for
        the lmstudio provider.  Implementing the reverse would risk spurious
        cross-publisher collisions and is explicitly out of scope.
        """
        custom = [
            {
                "base_url": "http://localhost:1234/v1",
                "models": {"pub/m": {"context_length": 8_192}},
            }
        ]
        assert (
            get_custom_provider_context_length(
                "m", "http://localhost:1234/v1", custom
            )
            is None
        )

    def test_case_sensitive_slug_fallback(self):
        """Slug fallback is case-sensitive, matching _model_id_matches precedent.

        Config key 'Nemotron' does not match runtime 'nvidia/nemotron'.
        Users must match case exactly, same as everywhere else in Hermes config.
        """
        custom = [
            {
                "base_url": "http://localhost:1234/v1",
                "models": {"Nemotron": {"context_length": 131_072}},
            }
        ]
        # Wrong case — must miss.
        assert (
            get_custom_provider_context_length(
                "nvidia/nemotron", "http://localhost:1234/v1", custom
            )
            is None
        )
        # Correct case — must hit.
        custom_correct = [
            {
                "base_url": "http://localhost:1234/v1",
                "models": {"nemotron": {"context_length": 131_072}},
            }
        ]
        assert (
            get_custom_provider_context_length(
                "nvidia/nemotron", "http://localhost:1234/v1", custom_correct
            )
            == 131_072
        )

    def test_quant_suffix_in_slug_not_supported(self):
        """Runtime id 'pub/model@q4_k_m' — slug fallback yields 'model@q4_k_m',
        not 'model'.  Quant-suffix stripping is explicitly out of scope for this
        fix.  LM Studio's native /api/v1/models returns plain 'publisher/slug'
        keys with no '@' quant suffix (confirmed by probe_lmstudio_models which
        uses raw.get('key') or raw.get('id')).  If a user manually writes a
        quant-suffixed config key, they must match the full runtime id exactly.
        """
        custom = [
            {
                "base_url": "http://localhost:1234/v1",
                "models": {"model": {"context_length": 32_768}},
            }
        ]
        # Falls through to slug 'model@q4_k_m' — does not match bare 'model'.
        assert (
            get_custom_provider_context_length(
                "pub/model@q4_k_m", "http://localhost:1234/v1", custom
            )
            is None
        )

    def test_multi_slash_id_strips_only_last_segment(self):
        """rsplit('/', 1)[1] takes the final segment of a multi-slash id.

        For 'org/team/model' the slug fallback is the last segment 'model'.
        LM Studio ids are single-slash 'publisher/slug', so this is an edge
        case, but pinning it documents the well-defined rsplit behavior.
        """
        custom = [
            {
                "base_url": "http://localhost:1234/v1",
                "models": {"model": {"context_length": 65_536}},
            }
        ]
        assert (
            get_custom_provider_context_length(
                "org/team/model", "http://localhost:1234/v1", custom
            )
            == 65_536
        )

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
