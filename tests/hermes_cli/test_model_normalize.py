"""Tests for hermes_cli.model_normalize — provider-aware model name normalization.

Covers issue #5211: opencode-go model names with dots (e.g. minimax-m2.7)
must NOT be mangled to hyphens (minimax-m2-7).
"""
import pytest

from hermes_cli.model_normalize import (
    normalize_model_for_provider,
    _DOT_TO_HYPHEN_PROVIDERS,
    _DEEPSEEK_CANONICAL_MODELS,
    _DEEPSEEK_V_SERIES_RE,
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
    """Retired aliases and fuzzy names rewrite to the CURRENT flash model.

    DeepSeek cut off ``deepseek-chat`` / ``deepseek-reasoner`` on
    2026-07-24; sending them on the wire returns HTTP 400. The rewrite target is
    asserted as a *contract* — "a DeepSeek-accepted id, and the same one the canonical
    flash id resolves to" — rather than a frozen literal, so this class does not need
    editing every time DeepSeek renames the flash model (it was pinned to the now-retired
    ``deepseek-v4-flash`` and had to be updated when ``deepseek-flash`` became canonical).
    """

    @staticmethod
    def _assert_deepseek_accepted(model_id: str) -> None:
        assert (
            model_id in _DEEPSEEK_CANONICAL_MODELS or _DEEPSEEK_V_SERIES_RE.match(model_id)
        ), f"{model_id!r} is not an id DeepSeek will accept"

    def test_provider_path_rewrites_reasoner(self):
        rewritten = normalize_model_for_provider("deepseek-reasoner", "deepseek")
        self._assert_deepseek_accepted(rewritten)
        assert rewritten == normalize_model_for_provider("deepseek-flash", "deepseek")

    @pytest.mark.parametrize("model", [
        "deepseek-r1",
        "deepseek-r1-0528",
        "deepseek-think-v3",
        "deepseek-reasoning-preview",
        "deepseek-cot-experimental",
    ])
    def test_reasoner_keywords_map_to_the_live_flash_model(self, model):
        rewritten = _normalize_for_deepseek(model)
        self._assert_deepseek_accepted(rewritten)
        assert rewritten == _normalize_for_deepseek("deepseek-flash")


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


# ── Regression: DeepSeek-V4.1-Flash canonical id (2026-09-10) ──────────

class TestDeepSeekFlashCanonicalId:
    """``deepseek-flash`` is the canonical id of the CURRENT flash model (V4.1-Flash).

    Regression: the id has no digit directly after ``-v``, so it never matched
    ``_DEEPSEEK_V_SERIES_RE`` and was absent from ``_DEEPSEEK_CANONICAL_MODELS``.
    It therefore fell through to the legacy fallback and was silently rewritten to the
    retired ``deepseek-v4-flash``, which DeepSeek keeps only as a TEMPORARY compat alias.
    """

    @pytest.mark.parametrize("model", [
        "deepseek-flash",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ])
    def test_accepted_ids_pass_through_unchanged(self, model):
        assert _normalize_for_deepseek(model) == model
        assert normalize_model_for_provider(model, "deepseek") == model

    def test_vendor_prefixed_canonical_id_survives(self):
        assert normalize_model_for_provider("deepseek/deepseek-flash", "deepseek") == "deepseek-flash"

    def test_matching_is_case_insensitive(self):
        assert _normalize_for_deepseek("DeepSeek-Flash") == "deepseek-flash"
        assert normalize_model_for_provider("deepseek-flash", "DEEPSEEK") == "deepseek-flash"

    def test_retired_aliases_resolve_to_the_live_flash_model(self):
        """deepseek-chat / deepseek-reasoner are aliases of the flash model's modes, so they
        must land on the same target as the canonical id rather than a legacy literal."""
        assert (
            _normalize_for_deepseek("deepseek-chat")
            == _normalize_for_deepseek("deepseek-reasoner")
            == _normalize_for_deepseek("deepseek-flash")
        )

    def test_unknown_input_falls_back_to_the_live_flash_model(self):
        assert _normalize_for_deepseek("some-retired-model") == _normalize_for_deepseek("deepseek-flash")

    def test_future_v_series_still_passes_through_without_release(self):
        """The explicit canonical list must not regress the generic V-series escape hatch."""
        assert _normalize_for_deepseek("deepseek-v5-flash") == "deepseek-v5-flash"

