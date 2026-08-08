"""Tests for hermes_cli.model_normalize — provider-aware model name normalization.

Covers issue #5211: opencode-go model names with dots (e.g. minimax-m2.7)
must NOT be mangled to hyphens (minimax-m2-7).
"""
import pytest

from hermes_cli.model_normalize import (
    normalize_model_for_provider,
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


# ── Regression: issue #64787 — ``provider:model`` colon prefixes ───────

# (provider, bare model, expected normalized result).  One row per provider
# branch that repairs a redundant ``provider/`` prefix, so the colon form is
# pinned everywhere the slash form already worked.
_COLON_PREFIX_CASES = [
    # _MATCHING_PREFIX_STRIP_PROVIDERS
    ("xai", "grok-5", "grok-5"),
    ("gemini", "gemini-3-pro", "gemini-3-pro"),
    ("zai", "glm-5.1", "glm-5.1"),
    ("minimax", "minimax-m2.7", "minimax-m2.7"),
    ("minimax-oauth", "minimax-m2.7", "minimax-m2.7"),
    ("minimax-cn", "minimax-m2.7", "minimax-m2.7"),
    ("kimi-coding", "kimi-k2.5", "kimi-k2.5"),
    ("kimi-coding-cn", "kimi-k2.5", "kimi-k2.5"),
    ("alibaba", "qwen3-max", "qwen3-max"),
    ("qwen-oauth", "qwen3-max", "qwen3-max"),
    ("arcee", "trinity-2", "trinity-2"),
    ("ollama-cloud", "qwen3", "qwen3"),
    ("xiaomi", "mimo-v2.5-pro", "mimo-v2.5-pro"),
    # _DOT_TO_HYPHEN_PROVIDERS
    ("anthropic", "claude-opus-5", "claude-opus-5"),
    # _STRIP_VENDOR_ONLY_PROVIDERS (+ the Copilot delegation ahead of it)
    ("openai-codex", "gpt-5.6-sol", "gpt-5.6-sol"),
    ("copilot", "gpt-5.4", "gpt-5.4"),
    ("copilot-acp", "gpt-5.4", "gpt-5.4"),
    # flat-namespace resellers
    ("opencode-zen", "glm-5.1", "glm-5.1"),
    ("opencode-go", "minimax-m2.7", "minimax-m2.7"),
    # DeepSeek canonicalisation
    ("deepseek", "deepseek-v4-pro", "deepseek-v4-pro"),
    # _CATALOGUE_PREFIX_REPAIR_PROVIDERS.  The odd one out: here ``nvidia/``
    # is the *canonical* prefix the API serves, not a redundant one, so both
    # spellings settle on the catalogue entry rather than on a bare id.
    (
        "nvidia",
        "nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-ultra-550b-a55b",
    ),
]


class TestIssue64787ColonProviderPrefix:
    """``provider:model`` must normalize exactly like ``provider/model``.

    ``hermes chat -m openai-codex:gpt-5.6-sol`` stores the flag verbatim
    (``cli.py`` -- only a ``moa:`` prefix is special-cased) and hands it
    straight to ``normalize_model_for_provider``.  Because
    ``_strip_matching_provider_prefix`` split on ``/`` only, the colon form
    survived normalization and the provider API received the redundant
    prefix.  Same reachable inputs: the Desktop model switch
    (``hermes_cli/web_server.py``), ``/model`` (``hermes_cli/model_switch.py``)
    and ``model.default`` in ``config.yaml``.
    """

    @pytest.mark.parametrize("provider,model,expected", _COLON_PREFIX_CASES)
    def test_colon_prefix_is_stripped(self, provider, model, expected):
        assert normalize_model_for_provider(f"{provider}:{model}", provider) == expected

    @pytest.mark.parametrize("provider,model,expected", _COLON_PREFIX_CASES)
    def test_slash_prefix_still_stripped(self, provider, model, expected):
        """Invariant control: the pre-existing slash behaviour is unchanged."""
        assert normalize_model_for_provider(f"{provider}/{model}", provider) == expected

    def test_deepseek_colon_prefix_no_longer_downgrades_v_series(self):
        """``deepseek:deepseek-v4-pro`` used to silently resolve to Flash.

        ``_DEEPSEEK_V_SERIES_RE`` is anchored at ``^deepseek-v<digit>``, so a
        surviving ``deepseek:`` prefix defeated the match and the unrecognised
        remainder fell through to ``_normalize_for_deepseek``'s catch-all --
        routing a user who explicitly picked V4 Pro to the cheaper V4 Flash
        with no error.  (Before the 2026-07-24 alias retirement the same path
        landed on ``deepseek-chat``/V3; the catch-all moved, the downgrade did
        not.)
        """
        assert normalize_model_for_provider("deepseek:deepseek-v4-pro", "deepseek") == "deepseek-v4-pro"
        # Control: the reasoner-keyword path is unaffected by prefix stripping.
        assert normalize_model_for_provider("deepseek:deepseek-r1", "deepseek") == "deepseek-v4-flash"

    def test_nvidia_colon_prefix_reaches_the_catalogue_repair(self):
        """``nvidia:<bare>`` used to reach the API with the prefix attached.

        ``_repair_prefix_from_catalogue`` short-circuits on ``/`` and then
        compares the *whole* name against each catalogue entry's post-slash
        suffix.  A colon-prefixed id has no slash, so it cleared the
        short-circuit and then matched nothing — the one dispatch branch that
        never stripped a matching prefix first, so ``nvidia:nemotron-…``
        reached build.nvidia.com verbatim and drew the same content-free
        ``404 page not found`` that #78796 set out to eliminate.
        """
        assert (
            normalize_model_for_provider("nvidia:nemotron-3-super-120b-a12b", "nvidia")
            == "nvidia/nemotron-3-super-120b-a12b"
        )
        # Alias-aware, like every other branch: ``nim`` resolves to ``nvidia``.
        assert (
            normalize_model_for_provider("nim:nemotron-3-super-120b-a12b", "nvidia")
            == "nvidia/nemotron-3-super-120b-a12b"
        )
        # The repaired prefix is the catalogue's, not a hardcoded ``nvidia/``.
        assert normalize_model_for_provider("nvidia:glm-5.2", "nvidia") == "z-ai/glm-5.2"

    @pytest.mark.parametrize("model", [
        # Slash form: the catalogue short-circuit is the only thing keeping a
        # self-hosted id addressable, so stripping must NOT reach it.
        "nvidia/my-local-nim-container",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "z-ai/glm-5.2",
        # Bare unknown name: a lookup miss stays a miss.
        "my-local-nim-container",
        # A colon that is not a provider prefix is left for the catalogue to
        # match verbatim, exactly as before.
        "nemotron-3-ultra-550b-a55b:beta",
    ])
    def test_nvidia_non_colon_prefixed_forms_are_untouched(self, model):
        """The #78796 pass-through guarantees survive the colon strip."""
        assert normalize_model_for_provider(model, "nvidia") == model

    def test_nvidia_colon_prefixed_unknown_name_stays_unrepaired(self):
        """Strip the redundant prefix, but never invent a catalogue entry.

        ``nvidia:my-local-nim`` names a self-hosted container behind the NVIDIA
        provider id; the prefix is redundant, the model is not in the
        catalogue, so the result is the bare id the local NIM actually serves.
        """
        assert (
            normalize_model_for_provider("nvidia:my-local-nim", "nvidia")
            == "my-local-nim"
        )

    def test_non_matching_colon_prefix_is_preserved(self):
        """A colon that is not a matching provider prefix must survive.

        ``llama3:8b`` is an Ollama tag, not a ``provider:model`` pair, so the
        alias match guard must leave it alone.
        """
        assert normalize_model_for_provider("llama3:8b", "ollama-cloud") == "llama3:8b"
        assert normalize_model_for_provider("kimi-k2.5:free", "opencode-zen") == "kimi-k2.5:free"
        assert normalize_model_for_provider("openai:gpt-5.4", "xai") == "openai:gpt-5.4"

    def test_custom_provider_colon_slug_is_preserved(self):
        """``custom:<name>`` is a durable provider identity, not a prefix."""
        assert normalize_model_for_provider("custom:my-model", "custom") == "custom:my-model"
        # The slash form keeps its pre-existing meaning.
        assert normalize_model_for_provider("custom/my-model", "custom") == "my-model"

    def test_variant_suffix_after_slash_prefix_is_untouched(self):
        """First-separator split: ``/`` wins when it comes before ``:``."""
        assert (
            normalize_model_for_provider("anthropic/claude-3.5-sonnet:beta", "openai-codex")
            == "anthropic/claude-3.5-sonnet:beta"
        )

    def test_openai_vendor_prefix_on_codex_still_stripped(self):
        """Regression guard for the existing openai-codex vendor carve-out."""
        assert normalize_model_for_provider("openai/gpt-5.4", "openai-codex") == "gpt-5.4"

    @pytest.mark.parametrize("model,expected", [
        ("openai-codex:gpt-5.6-sol", ("openai-codex", ":", "gpt-5.6-sol")),
        ("anthropic/claude-3.5-sonnet:beta", ("anthropic", "/", "claude-3.5-sonnet:beta")),
        ("openai-codex:anthropic/claude-opus-5", ("openai-codex", ":", "anthropic/claude-opus-5")),
        ("gpt-5.4", ("gpt-5.4", "", "")),
    ])
    def test_split_provider_prefix_takes_first_separator(self, model, expected):
        from hermes_cli.model_normalize import _split_provider_prefix

        assert _split_provider_prefix(model) == expected
