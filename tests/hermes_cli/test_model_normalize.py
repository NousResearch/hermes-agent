"""Tests for hermes_cli.model_normalize — provider-aware model name normalization.

Covers issue #5211: opencode-go model names with dots (e.g. minimax-m2.7)
must NOT be mangled to hyphens (minimax-m2-7).
"""
import pytest

from hermes_cli.model_normalize import (
    normalize_model_for_provider,
    _normalize_for_deepseek,
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


# ── Anthropic dot-to-hyphen conversion (regression) ────────────────────


# ── OpenCode Zen regression ────────────────────────────────────────────


# ── Copilot dot preservation (regression) ──────────────────────────────


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

    def test_deepseek_provider_preserves_versionless_flash_id(self):
        """``deepseek-flash`` must reach DeepSeek's API unchanged.

        DeepSeek's 2026-09 Flash refresh dropped the ``v<N>`` marker from the
        public id: ``GET /v1/models`` reports ``deepseek-flash`` and the API
        accepts it directly (verified live — it answers 200, and the older
        ``deepseek-v4-flash`` is aliased onto it).  Folding it onto
        ``deepseek-v4-flash`` meant the id users picked never reached the wire
        and the config stored a different model than the picker advertised.
        """
        assert (
            normalize_model_for_provider("deepseek-flash", "deepseek")
            == "deepseek-flash"
        )


# ── DeepSeek post-2026-07-24 alias remapping ───────────────────────────

class TestDeepseekRetiredAliasesAndCustomSlugs:
    """Only the two retired aliases are rewritten; every other id is the user's and reaches the
    wire as typed (a shape allow-list swallowed the vendor's own ``deepseek-flash``, #107206)."""

    def test_provider_path_rewrites_reasoner(self):
        assert (
            normalize_model_for_provider("deepseek-reasoner", "deepseek")
            == "deepseek-flash"
        )

    @pytest.mark.parametrize("model", [
        "deepseek-v4.1-flash",
        "deepseek-v4-flash-0731",
        "deepseek-r1",
        "deepseek-next-preview",
        "my-fine-tune",
    ])
    def test_unknown_ids_pass_through_untouched(self, model):
        assert _normalize_for_deepseek(model) == model


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


class TestColonProviderPrefixIsStrippedLikeSlash:
    """Issue #64787: ``-m openai-codex:gpt-5.6-sol`` (Hermes's own ``provider:model`` switch syntax)
    reached the Codex wire with the prefix attached and got HTTP 400. A matching ``provider:`` prefix
    must normalize exactly like ``provider/``; a later colon (Ollama tags) is never a separator."""

    def test_agent_init_strips_colon_prefix_before_the_wire(self, tmp_path, monkeypatch):
        """Production path: ``AIAgent(model="openai-codex:gpt-5.6-sol", provider="openai-codex")`` —
        the form that survives ``-m provider:model --provider X``, programmatic construction and
        gateway config — must leave ``agent.model`` without the prefix (agent/agent_init.py)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
        from run_agent import AIAgent

        agent = AIAgent(
            model="openai-codex:gpt-5.6-sol", provider="openai-codex", api_key="sk-dummy",
            base_url="https://chatgpt.com/backend-api/codex", quiet_mode=True,
            skip_context_files=True, skip_memory=True, platform="cli",
        )
        assert agent.model == "gpt-5.6-sol"

    @pytest.mark.parametrize("model,provider,expected", [
        ("openai:gpt-5.4", "openai-codex", "gpt-5.4"),
        ("zai:glm-5.1", "zai", "glm-5.1"),
        ("custom:qwen3:8b", "custom", "qwen3:8b"),
    ])
    def test_matching_colon_prefix_stripped(self, model, provider, expected):
        assert normalize_model_for_provider(model, provider) == expected

    @pytest.mark.parametrize("model,provider", [
        ("qwen3:8b", "custom"),            # bare Ollama tag: first colon is not a provider prefix
        ("anthropic:claude-x", "openai-codex"),  # non-matching prefix passes through untouched
    ])
    def test_non_matching_colon_untouched(self, model, provider):
        assert normalize_model_for_provider(model, provider) == model


# ── Named custom providers: foreign vendor/ prefix dropped by catalogue lookup ──
#
# #93980 / #93983 fix the ollama-cloud case with a static provider set. A *named* custom
# provider cannot live in a static set — it is user-defined — so the repair there has to be a
# lookup against the provider's own declared ``models:`` catalogue. That makes it the mirror of
# ``_repair_prefix_from_catalogue``, which restores a *dropped* prefix for providers that need
# one; both rewrite only an id the provider itself declares.

_QWEN_CLOUD_MODELS_YAML = """\
    models:
      deepseek-v4.1-flash: {}
      qwen3.8-max: {}
      glm-5.3: {}
"""

# The two shapes ``get_compatible_custom_providers`` merges: the legacy list and the v12+ dict.
# Written as literal YAML ("yaml" is not importable under scripts/run_tests.sh).
_NAMED_CUSTOM_LEGACY = """\
custom_providers:
  - name: qwencloud-individual
    base_url: https://token-plan.example.com/compatible-mode/v1
    key_env: QWENCLOUD_INDIVIDUAL_API_KEY
    api_mode: chat_completions
    models_discovered: true
""" + _QWEN_CLOUD_MODELS_YAML

_NAMED_CUSTOM_V12 = """\
providers:
  qwencloud-individual:
    name: qwencloud-individual
    api: https://token-plan.example.com/compatible-mode/v1
    key_env: QWENCLOUD_INDIVIDUAL_API_KEY
    transport: chat_completions
""" + _QWEN_CLOUD_MODELS_YAML

# A provider whose catalogue declares the *prefixed* form — the LiteLLM-style routing prefix that
# makes bare ``custom`` pass-through correct, and that the lookup must therefore preserve.
_RELAY_DECLARES_PREFIXED = """\
custom_providers:
  - name: relay
    base_url: https://relay.example.com/v1
    key_env: RELAY_KEY
    models:
      ollama/glm-5.2: {}
"""


def _home_with_custom_provider(tmp_path, monkeypatch, config_yaml):
    """Bind *config_yaml* as the active profile's config.yaml under HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(config_yaml, encoding="utf-8")
    return tmp_path


class TestNamedCustomProviderForeignVendorPrefix:
    """``custom:<name>`` sheds an aggregator-style ``vendor/`` prefix when — and only when — its
    own catalogue declares the bare id, so switching between an aggregator and a flat-namespace
    custom endpoint stops 404ing on every turn (#93980)."""

    @pytest.mark.parametrize(
        "config_yaml",
        [_NAMED_CUSTOM_LEGACY, _NAMED_CUSTOM_V12],
        ids=["legacy-custom_providers", "providers-dict"],
    )
    def test_drops_the_prefix_end_to_end(self, tmp_path, monkeypatch, config_yaml):
        """The repair on the real resolution chain: config.yaml → custom provider entry → AIAgent
        init normalization (agent/agent_init.py), the path a session actually takes."""
        _home_with_custom_provider(tmp_path, monkeypatch, config_yaml)
        from run_agent import AIAgent

        agent = AIAgent(
            model="deepseek/deepseek-v4.1-flash",
            provider="custom:qwencloud-individual",
            api_key="sk-dummy",
            base_url="https://token-plan.example.com/compatible-mode/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="cli",
        )
        assert agent.model == "deepseek-v4.1-flash"

    @pytest.mark.parametrize(
        "config_yaml,model,provider,expected",
        [
            # The declared bare id is already right where it is.
            (_NAMED_CUSTOM_LEGACY, "deepseek-v4.1-flash",
             "custom:qwencloud-individual", "deepseek-v4.1-flash"),
            # An id the provider does not declare keeps its spelling: the repair must never invent
            # one, so a genuine endpoint problem keeps its own classification.
            (_NAMED_CUSTOM_LEGACY, "z-ai/glm-5.2",
             "custom:qwencloud-individual", "z-ai/glm-5.2"),
            # A provider that declares the *prefixed* form (a LiteLLM-style routing prefix) keeps
            # it: an explicitly declared full id outranks stripping to its bare tail.
            (_RELAY_DECLARES_PREFIXED, "ollama/glm-5.2", "custom:relay", "ollama/glm-5.2"),
            # The catch-all ``custom`` bucket declares no catalogue — #93983's pass-through.
            (_NAMED_CUSTOM_LEGACY, "ollama/glm-5.2", "custom", "ollama/glm-5.2"),
            # No entry of that name; and nothing to strip on either empty side.
            (_NAMED_CUSTOM_LEGACY, "deepseek/deepseek-v4.1-flash",
             "custom:not-configured", "deepseek/deepseek-v4.1-flash"),
            (_NAMED_CUSTOM_LEGACY, "/deepseek-v4.1-flash",
             "custom:qwencloud-individual", "/deepseek-v4.1-flash"),
            (_NAMED_CUSTOM_LEGACY, "deepseek/", "custom:qwencloud-individual", "deepseek/"),
            # And the case that moves, through the same chokepoint, in both config shapes.
            (_NAMED_CUSTOM_LEGACY, "deepseek/deepseek-v4.1-flash",
             "custom:qwencloud-individual", "deepseek-v4.1-flash"),
            (_NAMED_CUSTOM_V12, "deepseek/deepseek-v4.1-flash",
             "custom:qwencloud-individual", "deepseek-v4.1-flash"),
        ],
        ids=[
            "declared-bare-id-kept",
            "undeclared-id-kept",
            "declared-prefixed-id-wins",
            "custom-bucket-untouched",
            "unconfigured-provider-untouched",
            "empty-prefix-untouched",
            "empty-remainder-untouched",
            "repair-legacy-custom_providers",
            "repair-providers-dict",
        ],
    )
    def test_repairs_exactly_what_the_catalogue_declares(
        self, tmp_path, monkeypatch, config_yaml, model, provider, expected
    ):
        """A lookup, never a shape rule: the bare id comes back only when the provider's own
        catalogue declares it and not the prefixed form."""
        _home_with_custom_provider(tmp_path, monkeypatch, config_yaml)
        assert normalize_model_for_provider(model, provider) == expected
