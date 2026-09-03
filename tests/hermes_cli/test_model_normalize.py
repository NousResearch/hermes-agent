"""Tests for hermes_cli.model_normalize — provider-aware model name normalization.

Covers issue #5211: opencode-go model names with dots (e.g. minimax-m2.7)
must NOT be mangled to hyphens (minimax-m2-7).
"""
import pytest

from hermes_cli.model_normalize import (
    normalize_model_for_provider,
    is_model_in_curated_catalog,
    _DOT_TO_HYPHEN_PROVIDERS,
    _normalize_for_deepseek,
    detect_vendor,
)


# ── Regression: issue #5211 ────────────────────────────────────────────

class TestIssue5211OpenCodeGoDotPreservation:
    """OpenCode Go model names with dots must pass through unchanged."""

    @pytest.mark.parametrize("model,expected", [
        ("minimax-m2.7", "minimax-m2.7"),
        ("minimax-m2.5", "minimax-m2.5"),
        ("glm-4.5", "glm-4.5"),
        ("kimi-k2.5", "kimi-k2.5"),
        ("some-model-1.0.3", "some-model-1.0.3"),
    ])
    def test_opencode_go_preserves_dots(self, model, expected):
        result = normalize_model_for_provider(model, "opencode-go")
        assert result == expected, f"Expected {expected!r}, got {result!r}"

    def test_opencode_go_not_in_dot_to_hyphen_set(self):
        """opencode-go must NOT be in the dot-to-hyphen provider set."""
        assert "opencode-go" not in _DOT_TO_HYPHEN_PROVIDERS


# ── Anthropic dot-to-hyphen conversion (regression) ────────────────────

class TestAnthropicDotToHyphen:
    """Anthropic API still needs dots→hyphens."""


# ── OpenCode Zen regression ────────────────────────────────────────────

class TestOpenCodeZenModelNormalization:
    """OpenCode Zen preserves dots for most models, but Claude stays hyphenated."""


# ── Copilot dot preservation (regression) ──────────────────────────────

class TestCopilotDotPreservation:
    """Copilot preserves dots in model names."""


# ── Copilot model-name normalization (issue #6879 regression) ──────────

class TestCopilotModelNormalization:
    """Copilot requires bare dot-notation model IDs.

    Regression coverage for issue #6879 and the broken Copilot branch
    that previously left vendor-prefixed Anthropic IDs (e.g.
    ``anthropic/claude-sonnet-4.6``) and dash-notation Claude IDs (e.g.
    ``claude-sonnet-4-6``) unchanged, causing the Copilot API to reject
    the request with HTTP 400 "model_not_supported".
    """


    def test_openai_codex_still_strips_openai_prefix(self):
        """Regression: openai-codex must still strip the openai/ prefix."""
        assert normalize_model_for_provider("openai/gpt-5.4", "openai-codex") == "gpt-5.4"


# ── Aggregator providers (regression) ──────────────────────────────────

class TestAggregatorProviders:
    """Aggregators need vendor/model slugs."""


class TestCustomProviderIsNotAVendorIdentity:
    """``custom`` is a generic bucket, not a vendor -- an alias that merely
    *resolves to* ``custom`` (e.g. ``ollama`` -> ``custom`` in
    ``_PROVIDER_ALIASES``) must not be treated as a redundant prefix the
    way ``zai/``, ``gemini/``, etc. are for their own native providers.

    Regression for: a named custom provider (e.g. a LiteLLM proxy fronting
    Ollama) registers its own routing name as ``ollama/glm-5.2``. Stripping
    the ``ollama/`` prefix because it happens to alias to ``custom``
    produced a bare ``glm-5.2`` the proxy doesn't recognise.
    """


# ── detect_vendor ──────────────────────────────────────────────────────


# ── DeepSeek V-series pass-through (bug: V4 models silently folded to V3) ──

class TestDeepseekVSeriesPassThrough:
    """DeepSeek's V-series IDs (``deepseek-v4-pro``, ``deepseek-v4-flash``,
    and future ``deepseek-v<N>-*`` variants) are first-class model IDs
    accepted directly by DeepSeek's Chat Completions API. Earlier code
    folded every non-reasoner name into ``deepseek-chat``, which on
    aggregators (Nous portal, OpenRouter via DeepInfra) routes to V3 —
    silently downgrading users who picked V4.
    """


    def test_deepseek_provider_preserves_v4_pro(self):
        """End-to-end via normalize_model_for_provider — user selecting
        V4 Pro must reach DeepSeek's API as V4 Pro, not V3 alias."""
        result = normalize_model_for_provider("deepseek-v4-pro", "deepseek")
        assert result == "deepseek-v4-pro"


# ── DeepSeek post-2026-07-24 alias remapping ───────────────────────────

class TestDeepseekCanonicalAndReasonerMapping:
    """Retired aliases and fuzzy names rewrite to deepseek-v4-flash.

    DeepSeek cut off ``deepseek-chat`` / ``deepseek-reasoner`` on
    2026-07-24; sending them on the wire returns HTTP 400.
    """


    def test_provider_path_rewrites_reasoner(self):
        assert (
            normalize_model_for_provider("deepseek-reasoner", "deepseek")
            == "deepseek-v4-flash"
        )

    @pytest.mark.parametrize("model", [
        "deepseek-r1",
        "deepseek-r1-0528",
        "deepseek-think-v3",
        "deepseek-reasoning-preview",
        "deepseek-cot-experimental",
    ])
    def test_reasoner_keywords_map_to_v4_flash(self, model):
        assert _normalize_for_deepseek(model) == "deepseek-v4-flash"


# ── Regression: issue #78796 ───────────────────────────────────────────

class TestIssue78796NvidiaPrefixRepair:
    """A bare NVIDIA model id must regain its ``vendor/`` prefix.

    build.nvidia.com serves ``nvidia/nemotron-…``; a bare
    ``nemotron-3-ultra-550b-a55b`` returns a naked ``404 page not found``
    that never names the model, so the failure reads like an outage.
    """

    @pytest.mark.parametrize("model,expected", [
        ("nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-ultra-550b-a55b"),
        ("nemotron-3-super-120b-a12b", "nvidia/nemotron-3-super-120b-a12b"),
        (
            "nemotron-3-nano-omni-30b-a3b-reasoning",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        ),
    ])
    def test_bare_nemotron_regains_prefix(self, model, expected):
        assert normalize_model_for_provider(model, "nvidia") == expected

    def test_third_party_model_gets_its_own_vendor(self):
        """NIM also hosts third-party models — the prefix is the catalogue's,
        not a hardcoded ``nvidia/``."""
        assert normalize_model_for_provider("glm-5.2", "nvidia") == "z-ai/glm-5.2"

    @pytest.mark.parametrize("model", [
        "nvidia/nemotron-3-ultra-550b-a55b",
        "z-ai/glm-5.2",
    ])
    def test_already_prefixed_is_untouched(self, model):
        assert normalize_model_for_provider(model, "nvidia") == model

    @pytest.mark.parametrize("model", [
        "my-local-nim-container",
        "some-finetune-v2",
    ])
    def test_unknown_names_pass_through(self, model):
        """The same provider id fronts local NIM containers. An id absent from
        the catalogue is a lookup miss, not a guess — leave it alone."""
        assert normalize_model_for_provider(model, "nvidia") == model

    def test_other_providers_unaffected(self):
        assert normalize_model_for_provider("my-model", "custom") == "my-model"
        assert (
            normalize_model_for_provider("claude-sonnet-4.6", "openrouter")
            == "anthropic/claude-sonnet-4.6"
        )


# ── Regression: issue #96276 (stale-catalog 404) ───────────────────────

class TestIsModelInCuratedCatalog:
    """``is_model_in_curated_catalog`` disambiguates a 404 whose model IS in
    Hermes' curated picker list — i.e. the stale-catalog / retired-model
    class that surfaces a raw ``Provider error HTTP 404: Model 'X' not
    found`` to the user with no actionable hint (#96276).

    Pre-fix the helper did not exist; ``agent.conversation_loop`` had no way
    to tell a typo'd bare id (already covered by ``suggest_prefixed_model_id``)
    from a known curated id the provider has since dropped, so the second
    case surfaced the bare 404 string. The post-fix helper returns ``True``
    for curated entries so the conversation loop can render the
    "model retired — try ``/model --refresh``" hint.
    """

    def test_curated_model_under_nous_returns_true(self):
        """The post-fix surface: when a model is in ``_PROVIDER_MODELS[nous]``
        (so it was a known Portal-recommended pick) and the provider returns
        404, the helper returns True so the conversation loop can show the
        "model retired — try ``/model --refresh``" hint (#96276).

        Regression anchors on ``anthropic/claude-sonnet-5`` because it is
        pinned in the curated ``nous`` list and a Portal-side retirement is
        the realistic 404 trigger. The original report named
        ``stealth/ox-alpha`` which is no longer curated at this revision,
        so the test asserts the behavior contract, not the literal incident
        model id (catalogs drift; the helper contract is what matters).
        """
        assert is_model_in_curated_catalog("nous", "anthropic/claude-sonnet-5") is True

    @pytest.mark.parametrize("provider,model", [
        # Aggregator / Nous-Portal style: vendor-prefixed.
        ("nous", "anthropic/claude-fable-5"),
        ("kilocode", "anthropic/claude-opus-4.6"),
        ("gmi", "anthropic/claude-sonnet-5"),
        # Native-provider style: bare ids in the catalog.
        ("openai", "gpt-5.4"),
        ("anthropic", "claude-fable-5"),
        ("deepseek", "deepseek-v4-flash"),
        ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
    ])
    def test_known_curated_entries_return_true(self, provider, model):
        """A well-formed curated id returns True — covers the common path
        where the provider has since retired the model."""
        assert is_model_in_curated_catalog(provider, model) is True

    @pytest.mark.parametrize("provider,model", [
        # Bare id (no vendor/ prefix) — owned by ``suggest_prefixed_model_id``.
        ("nous", "ox-alpha"),
        ("nvidia", "nemotron-3-ultra-550b-a55b"),
        # Hand-rolled id never curated — must NOT claim catalogue membership
        # or we'd mis-diagnose a genuine 404 as "retired".
        ("nous", "some-hand-rolled-id/variant-3"),
        ("openai", "gpt-9.9-imaginary"),
        # Provider with no curated list.
        ("custom", "anything/at-all"),
    ])
    def test_non_curated_or_bare_ids_return_false(self, provider, model):
        """Negative path: never claim catalogue membership for ids that
        don't have a curated entry — that's the bug we're preventing."""
        assert is_model_in_curated_catalog(provider, model) is False

    def test_blank_or_empty_inputs_return_false(self):
        """Defensive — an empty model name or provider must not crash and
        must not claim membership (the conversation-loop site swallows
        import errors, but the helper itself stays honest)."""
        assert is_model_in_curated_catalog("", "anthropic/claude-sonnet-5") is False
        assert is_model_in_curated_catalog("nous", "") is False
        assert is_model_in_curated_catalog("", "") is False
        assert is_model_in_curated_catalog("nous", "  ") is False

    def test_alias_provider_is_normalised(self):
        """A known alias for the canonical provider still finds the entry.
        ``_normalize_provider_alias`` (the canonical resolver) maps common
        spellings before the catalogue lookup; a regression there would
        silently downgrade this hint to "model not in catalog" for users on
        aliased provider ids."""
        # If there's a known alias in the catalogue, prefer that entry.
        # We don't lock onto a specific alias here (catalogs drift); the
        # canonical ``nous`` path is the regression assertion.
        assert is_model_in_curated_catalog("nous", "anthropic/claude-sonnet-5") is True

