"""Tests for provider-aware `/model` validation in hermes_cli.models."""

import pytest
from unittest.mock import MagicMock, patch

from hermes_cli.models import (
    azure_foundry_model_api_mode,
    copilot_model_api_mode,
    fetch_github_model_catalog,
    curated_models_for_provider,
    fetch_api_models,
    fetch_lmstudio_models,
    github_model_reasoning_efforts,
    normalize_copilot_model_id,
    normalize_opencode_model_id,
    normalize_provider,
    opencode_model_api_mode,
    parse_model_input,
    probe_api_models,
    provider_label,
    provider_model_ids,
    validate_requested_model,
)


# -- helpers -----------------------------------------------------------------

FAKE_API_MODELS = [
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.4-pro",
    "openai/gpt-5.4",
    "google/gemini-3-pro-preview",
]


def _validate(model, provider="openrouter", api_models=FAKE_API_MODELS, **kw):
    """Shortcut: call validate_requested_model with mocked API."""
    probe_payload = {
        "models": api_models,
        "probed_url": "http://localhost:11434/v1/models",
        "resolved_base_url": kw.get("base_url", "") or "http://localhost:11434/v1",
        "suggested_base_url": None,
        "used_fallback": False,
    }
    with patch("hermes_cli.models.fetch_api_models", return_value=api_models), \
         patch("hermes_cli.models.probe_api_models", return_value=probe_payload):
        return validate_requested_model(model, provider, **kw)


# -- parse_model_input -------------------------------------------------------

class TestParseModelInput:
    def test_plain_model_keeps_current_provider(self):
        provider, model = parse_model_input("anthropic/claude-sonnet-4.5", "openrouter")
        assert provider == "openrouter"
        assert model == "anthropic/claude-sonnet-4.5"


# -- curated_models_for_provider ---------------------------------------------

class TestCuratedModelsForProvider:
    def test_openrouter_returns_curated_list(self):
        with patch(
            "hermes_cli.models.fetch_openrouter_models",
            return_value=[
                ("anthropic/claude-opus-4.6", "recommended"),
                ("qwen/qwen3.6-plus", ""),
            ],
        ):
            models = curated_models_for_provider("openrouter")
        assert len(models) > 0
        assert any("claude" in m[0] for m in models)

    def test_unknown_provider_returns_empty(self):
        assert curated_models_for_provider("totally-unknown") == []


# -- normalize_provider ------------------------------------------------------

class TestNormalizeProvider:

    def test_known_aliases(self):
        assert normalize_provider("glm") == "zai"
        assert normalize_provider("kimi") == "kimi-coding"
        assert normalize_provider("moonshot") == "kimi-coding"
        assert normalize_provider("step") == "stepfun"
        assert normalize_provider("github-copilot") == "copilot"


class TestProviderLabel:
    def test_known_labels_and_auto(self):
        assert provider_label("anthropic") == "Anthropic"
        assert provider_label("kimi") == "Kimi / Kimi Coding Plan"
        assert provider_label("stepfun") == "StepFun Step Plan"
        assert provider_label("copilot") == "GitHub Copilot"
        assert provider_label("copilot-acp") == "GitHub Copilot ACP"
        assert provider_label("auto") == "Auto"


# -- provider_model_ids ------------------------------------------------------

class TestProviderModelIds:


    def test_stepfun_prefers_live_catalog(self):
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "***", "base_url": "https://api.stepfun.com/step_plan/v1"},
        ), patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["step-3.5-flash", "step-3-agent-lite"],
        ):
            assert provider_model_ids("stepfun") == ["step-3.5-flash", "step-3-agent-lite"]


    def test_anthropic_provider_uses_configured_base_url_for_live_catalog(self):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data": [{"id": "enterprise-claude"}]}'

        with patch(
            "hermes_cli.config.load_config",
            return_value={
                "model": {
                    "provider": "anthropic",
                    "base_url": "http://localhost:6655/anthropic/v1",
                    "api_key": "proxy-key",
                }
            },
        ), patch(
            "hermes_cli.models._urlopen_model_catalog_request",
            return_value=_Resp(),
        ) as mock_urlopen:
            assert provider_model_ids("anthropic") == ["enterprise-claude"]

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:6655/anthropic/v1/models"
        assert req.get_header("X-api-key") == "proxy-key"

    def test_custom_provider_passes_anthropic_mode_for_versioned_proxy_catalog(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={
                "model": {
                    "provider": "custom",
                    "base_url": "http://localhost:6655/anthropic/v1",
                    "api_key": "proxy-key",
                }
            },
        ), patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["enterprise-claude"],
        ) as mock_fetch:
            assert provider_model_ids("custom") == ["enterprise-claude"]

        mock_fetch.assert_called_once_with(
            "proxy-key",
            "http://localhost:6655/anthropic/v1",
            api_mode="anthropic_messages",
        )


# -- fetch_api_models --------------------------------------------------------

class TestFetchApiModels:
    def test_returns_none_when_no_base_url(self):
        assert fetch_api_models("key", None) is None


    def test_probe_api_models_tries_v1_fallback(self):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data": [{"id": "local-model"}]}'

        calls = []

        def _fake_urlopen(req, timeout=5.0):
            calls.append(req.full_url)
            if req.full_url.endswith("/v1/models"):
                return _Resp()
            raise Exception("404")

        with patch("hermes_cli.models._urlopen_model_catalog_request", side_effect=_fake_urlopen):
            probe = probe_api_models("key", "http://localhost:8000")

        assert calls == ["http://localhost:8000/models", "http://localhost:8000/v1/models"]
        assert probe["models"] == ["local-model"]
        assert probe["resolved_base_url"] == "http://localhost:8000/v1"
        assert probe["used_fallback"] is True

    def test_probe_api_models_uses_copilot_catalog(self):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data": [{"id": "gpt-5.4", "model_picker_enabled": true, "supported_endpoints": ["/responses"], "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}}}, {"id": "claude-sonnet-4.6", "model_picker_enabled": true, "supported_endpoints": ["/chat/completions"], "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}}}, {"id": "text-embedding-3-small", "model_picker_enabled": true, "capabilities": {"type": "embedding"}}]}'

        with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=_Resp()) as mock_urlopen:
            probe = probe_api_models("gh-token", "https://api.githubcopilot.com")

        assert mock_urlopen.call_args[0][0].full_url == "https://api.githubcopilot.com/models"
        assert probe["models"] == ["gpt-5.4", "claude-sonnet-4.6"]
        assert probe["resolved_base_url"] == "https://api.githubcopilot.com"
        assert probe["used_fallback"] is False


class TestGithubReasoningEfforts:
    def test_gpt5_supports_minimal_to_high(self):
        catalog = [{
            "id": "gpt-5.4",
            "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}},
            "supported_endpoints": ["/responses"],
        }]
        assert github_model_reasoning_efforts("gpt-5.4", catalog=catalog) == [
            "low",
            "medium",
            "high",
        ]


class TestCopilotNormalization:

    def test_copilot_api_mode_gpt5_uses_responses(self):
        """GPT-5+ models should use Responses API (matching opencode)."""
        assert copilot_model_api_mode("gpt-5.4") == "codex_responses"
        assert copilot_model_api_mode("gpt-5.4-mini") == "codex_responses"
        assert copilot_model_api_mode("gpt-5.3-codex") == "codex_responses"
        assert copilot_model_api_mode("gpt-5.2-codex") == "codex_responses"
        assert copilot_model_api_mode("gpt-5.2") == "codex_responses"





    def test_opencode_go_api_modes_match_docs(self):
        assert opencode_model_api_mode("opencode-go", "glm-5.1") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "opencode-go/glm-5.1") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "glm-5") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "opencode-go/glm-5") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "kimi-k2.5") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "opencode-go/kimi-k2.5") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "minimax-m2.5") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-go", "opencode-go/minimax-m2.5") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-go", "qwen3.7-max") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-go", "opencode-go/qwen3.7-max") == "anthropic_messages"
        # All Qwen models on Go route via /v1/messages (Go endpoint table).
        assert opencode_model_api_mode("opencode-go", "qwen3.7-plus") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-go", "qwen3.6-plus") == "anthropic_messages"
        # DeepSeek / MiMo on Go are OpenAI-compatible chat completions.
        assert opencode_model_api_mode("opencode-go", "deepseek-v4-pro") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "deepseek-v4-flash") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "mimo-v2.5") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "kimi-k2.7-code") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "glm-5.2") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "minimax-m3") == "anthropic_messages"
        # GPT models on Go are Responses-only (Go endpoint table).
        assert opencode_model_api_mode("opencode-go", "gpt-5.6-luna") == "codex_responses"
        assert opencode_model_api_mode("opencode-go", "opencode-go/gpt-5.6-luna") == "codex_responses"
        # Muse Spark on Go is Responses-only. chat/completions returns HTTP 503.
        assert opencode_model_api_mode("opencode-go", "muse-spark-1.2-contributor") == "codex_responses"
        assert opencode_model_api_mode("opencode-go", "opencode-go/muse-spark-1.2-contributor") == "codex_responses"
        assert opencode_model_api_mode("opencode-go", "muse-spark-1.2") == "codex_responses"
        # Zen serves the standard Muse Spark variant on /v1/responses too.
        assert opencode_model_api_mode("opencode-zen", "muse-spark-1.2") == "codex_responses"
        assert opencode_model_api_mode("opencode-zen", "opencode-zen/muse-spark-1.2") == "codex_responses"
        # Grok models route via /v1/responses on both Zen and Go
        # (Zen/Go endpoint tables).
        assert opencode_model_api_mode("opencode-go", "grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("opencode-go", "opencode-go/grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("opencode-zen", "grok-4.6") == "codex_responses"
        assert opencode_model_api_mode("opencode-zen", "grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("opencode-zen", "grok-build-0.1") == "codex_responses"
        # Ox Alpha (x-preview-f-free) on Zen is OpenAI-compatible
        # chat/completions per the Zen endpoint table.
        assert opencode_model_api_mode("opencode-zen", "x-preview-f-free") == "chat_completions"
        assert opencode_model_api_mode("opencode-zen", "opencode-zen/x-preview-f-free") == "chat_completions"
        # Other free-tier Zen models are chat/completions too.
        assert opencode_model_api_mode("opencode-zen", "hy3-free") == "chat_completions"
        assert opencode_model_api_mode("opencode-zen", "nemotron-3.5-lightning-free") == "chat_completions"
        # Hy3 on Go is chat/completions (Go endpoint table).
        assert opencode_model_api_mode("opencode-go", "hy3") == "chat_completions"
        # New Go models keep their family routing: GLM chat/completions,
        # Qwen anthropic_messages.
        assert opencode_model_api_mode("opencode-go", "glm-5.3") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "glm-5.3-flash") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "qwen3.8-max") == "anthropic_messages"
        # Custom opencode-go-* providers route according to opencode-go rules
        # (family-prefix providers, issue #85589).
        assert opencode_model_api_mode("opencode-go-bridge", "grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("opencode-go-bridge", "opencode-go-bridge/grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("opencode-go-bridge", "minimax-m2.5") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-go-bridge", "deepseek-v4-flash") == "chat_completions"
        # Case-insensitive provider ID handling (e.g. OpenCode-Go-Bridge).
        assert opencode_model_api_mode("OpenCode-Go-Bridge", "grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("OpenCode-Go-Bridge", "minimax-m2.5") == "anthropic_messages"
        # Custom opencode-zen-* providers route according to opencode-zen rules.
        assert opencode_model_api_mode("opencode-zen-custom", "claude-3-5-sonnet") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-zen-custom", "gpt-5") == "codex_responses"
        assert opencode_model_api_mode("opencode-zen-custom", "grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("OpenCode-Zen-Custom", "claude-3-7-sonnet") == "anthropic_messages"


class TestNormalizeOpencodeBaseUrl:
    """Symmetric /v1 normalization for OpenCode Zen / Go base URLs.

    Regression for the 'only minimax works on opencode-go' bug: switching into
    an anthropic-routed model strips /v1 from the base URL and that stripped
    URL gets persisted to model.base_url; every later chat_completions model
    (glm, deepseek, kimi) then POSTed to https://opencode.ai/zen/go/chat/completions
    — a 404 (the marketing site).  The normalizer must heal a stripped URL.
    """

    def test_strips_v1_for_anthropic_messages(self):
        from hermes_cli.models import normalize_opencode_base_url
        assert normalize_opencode_base_url(
            "opencode-go", "anthropic_messages", "https://opencode.ai/zen/go/v1"
        ) == "https://opencode.ai/zen/go"
        assert normalize_opencode_base_url(
            "opencode-zen", "anthropic_messages", "https://opencode.ai/zen/v1/"
        ) == "https://opencode.ai/zen"


    def test_non_opencode_provider_untouched(self):
        from hermes_cli.models import normalize_opencode_base_url
        assert normalize_opencode_base_url(
            "openrouter", "chat_completions", "https://openrouter.ai/api"
        ) == "https://openrouter.ai/api"


class TestAzureFoundryModelApiMode:
    """Azure Foundry deploys GPT-5.x / codex / o-series as Responses-API-only.

    Azure returns ``400 "The requested operation is unsupported."`` when
    /chat/completions is called against these deployments.  Verified in the
    wild by a user debug bundle on 2026-04-26: gpt-5.3-codex failed with
    that exact payload while gpt-4o-pure worked on the same endpoint.
    """

    def test_gpt5_family_uses_responses(self):
        assert azure_foundry_model_api_mode("gpt-5") == "codex_responses"
        assert azure_foundry_model_api_mode("gpt-5.3") == "codex_responses"
        assert azure_foundry_model_api_mode("gpt-5.4") == "codex_responses"
        assert azure_foundry_model_api_mode("gpt-5-codex") == "codex_responses"
        assert azure_foundry_model_api_mode("gpt-5.3-codex") == "codex_responses"
        # gpt-5-mini exceptions are Copilot-specific; Azure deploys the whole
        # gpt-5 family on Responses API uniformly.
        assert azure_foundry_model_api_mode("gpt-5-mini") == "codex_responses"

    def test_codex_family_uses_responses(self):
        assert azure_foundry_model_api_mode("codex") == "codex_responses"
        assert azure_foundry_model_api_mode("codex-mini") == "codex_responses"


    def test_gpt4_family_returns_none(self):
        """GPT-4, GPT-4o, etc. speak chat completions on Azure."""
        assert azure_foundry_model_api_mode("gpt-4") is None
        assert azure_foundry_model_api_mode("gpt-4o") is None
        assert azure_foundry_model_api_mode("gpt-4o-pure") is None
        assert azure_foundry_model_api_mode("gpt-4o-mini") is None
        assert azure_foundry_model_api_mode("gpt-4-turbo") is None
        assert azure_foundry_model_api_mode("gpt-4.1") is None
        assert azure_foundry_model_api_mode("gpt-3.5-turbo") is None


# -- validate — format checks -----------------------------------------------

class TestValidateFormatChecks:
    def test_empty_model_rejected(self):
        result = _validate("")
        assert result["accepted"] is False
        assert "empty" in result["message"]


    def test_no_slash_model_still_probes_api(self):
        result = _validate("gpt-5.4", api_models=["gpt-5.4", "gpt-5.4-pro"])
        assert result["accepted"] is True
        assert result["persist"] is True

    def test_no_slash_model_rejected_if_not_in_api(self):
        result = _validate("gpt-5.4", api_models=["openai/gpt-5.4"])
        assert result["accepted"] is False
        assert result["persist"] is False
        assert "not found" in result["message"]


# -- validate — API found ----------------------------------------------------


# -- validate — API not found ------------------------------------------------

class TestValidateApiNotFound:

    def test_warning_includes_suggestions(self):
        result = _validate("anthropic/claude-opus-4.5")
        assert result["accepted"] is True
        # Close match auto-corrects; less similar inputs show suggestions
        assert "Auto-corrected" in result["message"] or "Similar models" in result["message"]


# -- validate — API unreachable — soft-accept via catalog or warning --------

class TestValidateApiFallback:
    """When /models is unreachable, the validator must accept the model (with
    a warning) rather than reject it outright — otherwise provider switches
    fail in the gateway for any provider whose /models endpoint is down or
    doesn't exist (e.g. opencode-go returns 404 HTML).

    Two paths:
      1. Provider has a curated catalog (``_PROVIDER_MODELS`` / live fetch):
         validate against it (recognized=True for known models,
         recognized=False with 'Note:' for unknown).
      2. Provider has no catalog: accept with a generic 'Note:' warning.

    In both cases ``accepted`` and ``persist`` must be True so the gateway can
    write the ``_session_model_overrides`` entry.
    """






    def test_fetch_lmstudio_models_filters_embedding_type(self):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_resp.read.return_value = (
            b'{"models":['
            b'{"key":"publisher/chat-model","id":"publisher/chat-model","type":"llm"},'
            b'{"key":"publisher/embed-model","id":"publisher/embed-model","type":"embedding"}'
            b']}'
        )

        with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=mock_resp):
            models = fetch_lmstudio_models(base_url="http://localhost:1234/v1")

        assert models == ["publisher/chat-model"]



    def test_validate_lmstudio_distinguishes_auth_failure(self):
        import urllib.error

        http_error = urllib.error.HTTPError(
            url="http://localhost:1234/api/v1/models",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

        with patch("hermes_cli.models._urlopen_model_catalog_request", side_effect=http_error):
            result = validate_requested_model(
                "publisher/chat-model",
                "lmstudio",
                base_url="http://localhost:1234/v1",
            )

        assert result["accepted"] is False
        assert "401" in result["message"]
        assert "LM_API_KEY" in result["message"]



# -- validate — Codex auto-correction ------------------------------------------

class TestValidateCodexAutoCorrection:
    """Auto-correction for typos on openai-codex provider."""

    def test_missing_dash_auto_corrects(self):
        """gpt5.3-codex (missing dash) auto-corrects to gpt-5.3-codex."""
        codex_models = ["gpt-5.4-mini", "gpt-5.4", "gpt-5.3-codex",
                        "gpt-5.2-codex", "gpt-5.1-codex-max"]
        with patch("hermes_cli.models.provider_model_ids", return_value=codex_models):
            result = validate_requested_model("gpt5.3-codex", "openai-codex")
        assert result["accepted"] is True
        assert result["recognized"] is True
        assert result["corrected_model"] == "gpt-5.3-codex"
        assert "Auto-corrected" in result["message"]

    def test_exact_match_no_correction(self):
        """Exact model name does not trigger auto-correction."""
        codex_models = ["gpt-5.4-mini", "gpt-5.4", "gpt-5.3-codex"]
        with patch("hermes_cli.models.provider_model_ids", return_value=codex_models):
            result = validate_requested_model("gpt-5.3-codex", "openai-codex")
        assert result["accepted"] is True
        assert result["recognized"] is True
        assert result.get("corrected_model") is None
        assert result["message"] is None


class TestValidateCodex900kVariants:
    """`-900k` is a Hermes picker convention: valid variants come from the
    catalog; ineligible aliases are hard-rejected BEFORE the hidden-slug
    soft-accept (#92797 review)."""

    _CATALOG = ["gpt-5.6-sol", "gpt-5.6-sol-900k", "gpt-5.5", "gpt-5.4-mini"]

    def test_catalog_listed_variant_accepted(self):
        with patch("hermes_cli.models.provider_model_ids", return_value=self._CATALOG):
            result = validate_requested_model("gpt-5.6-sol-900k", "openai-codex")
        assert result["accepted"] is True
        assert result["recognized"] is True

    @pytest.mark.parametrize("alias", ["gpt-5.5-900k", "gpt-5.4-mini-900k", "gpt-5.6-sol-pro-900k"])
    def test_ineligible_900k_alias_rejected_not_soft_accepted(self, alias):
        with patch("hermes_cli.models.provider_model_ids", return_value=self._CATALOG):
            result = validate_requested_model(alias, "openai-codex")
        assert result["accepted"] is False
        assert result["persist"] is False
        assert "272K" in result["message"]

    def test_valid_variant_missing_from_catalog_still_accepted(self):
        """A verified variant not yet in the (possibly stale) catalog is
        accepted via the eligibility predicate, not the soft-accept."""
        with patch("hermes_cli.models.provider_model_ids", return_value=["gpt-5.6-sol"]):
            result = validate_requested_model("gpt-5.6-sol-900k", "openai-codex")
        assert result["accepted"] is True


# -- probe_api_models — Cloudflare UA mitigation --------------------------------

class TestProbeApiModelsUserAgent:
    """Probing custom /v1/models must send a Hermes User-Agent.

    Some custom Claude proxies (e.g. ``packyapi.com``) sit behind Cloudflare with
    Browser Integrity Check enabled. The default ``Python-urllib/3.x`` signature
    is rejected with HTTP 403 ``error code: 1010``, which ``probe_api_models``
    swallowed into ``{"models": None}``, surfacing to users as a misleading
    "Could not reach the ... API to validate ..." error — even though the
    endpoint is reachable and the listing exists.
    """

    def _make_mock_response(self, body: bytes):
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=body)
        return mock_resp

    def test_probe_sends_hermes_user_agent(self):
        from unittest.mock import patch

        body = b'{"data":[{"id":"claude-opus-4.7"}]}'
        with patch(
            "hermes_cli.models._urlopen_model_catalog_request",
            return_value=self._make_mock_response(body),
        ) as mock_urlopen:
            result = probe_api_models("sk-test", "https://example.com/v1")

        assert result["models"] == ["claude-opus-4.7"]
        # The urlopen call receives a Request object as its first positional arg
        req = mock_urlopen.call_args[0][0]
        ua = req.get_header("User-agent")  # urllib title-cases header names
        assert ua, "probe_api_models must send a User-Agent header"
        assert ua.startswith("hermes-cli/"), (
            f"User-Agent must advertise hermes-cli, got {ua!r}"
        )
        # Must not fall back to urllib's default — that's what Cloudflare 1010 blocks.
        assert not ua.startswith("Python-urllib")

    def test_probe_user_agent_sent_without_api_key(self):
        """UA must be present even for endpoints that don't need auth."""
        from unittest.mock import patch

        body = b'{"data":[]}'
        with patch(
            "hermes_cli.models._urlopen_model_catalog_request",
            return_value=self._make_mock_response(body),
        ) as mock_urlopen:
            probe_api_models(None, "https://example.com/v1")

        req = mock_urlopen.call_args[0][0]
        ua = req.get_header("User-agent")
        assert ua and ua.startswith("hermes-cli/")
        # No Authorization was set, but UA must still be present.
        assert req.get_header("Authorization") is None




# -- validate — OpenRouter routing-variant suffixes (:nitro / :floor / ...) ----

class TestValidateOpenRouterVariantSuffixes:
    """OpenRouter's `:nitro`, `:floor`, `:exacto`, `:online` are request-time
    routing modifiers, not catalog models — /models lists only the base id.
    Validation must accept `base:variant` when `base` is listed, preserve the
    suffixed id (no auto-correct stripping the routing opt-in), and still
    reject variants on unknown bases and unknown suffixes."""

    _LISTING = [
        "~x-ai/grok-latest",
        "x-ai/grok-4.6",
        "deepseek/deepseek-v4-flash",
        "thinkingmachines/inkling:free",
    ]

    def _validate(self, model):
        return _validate(model, "openrouter", api_models=self._LISTING)

    @pytest.mark.parametrize("suffix", ["nitro", "floor", "exacto", "online"])
    def test_variant_on_listed_base_accepted_unmodified(self, suffix):
        result = self._validate(f"~x-ai/grok-latest:{suffix}")
        assert result["accepted"] is True
        assert result["recognized"] is True
        assert result.get("corrected_model") is None
        assert result["message"] is None

    def test_variant_not_fuzzy_corrected_to_base(self):
        """The old failure mode: get_close_matches would 'fix' model:nitro
        to the bare base id and silently drop the routing behavior."""
        result = self._validate("x-ai/grok-4.6:nitro")
        assert result["accepted"] is True
        assert result.get("corrected_model") is None

    def test_variant_on_unknown_base_rejected(self):
        result = self._validate("x-ai/notreal-model:nitro")
        assert result["accepted"] is False

    def test_unknown_suffix_keeps_old_behavior(self):
        result = self._validate("x-ai/grok-4.6:bogus")
        assert result["accepted"] is False

    def test_free_sku_still_direct_matched(self):
        """`:free` SKUs ARE catalog entries; direct membership handles them."""
        result = self._validate("thinkingmachines/inkling:free")
        assert result["accepted"] is True
        assert result.get("corrected_model") is None

    def test_variant_uppercase_suffix_accepted(self):
        result = self._validate("x-ai/grok-4.6:NITRO")
        assert result["accepted"] is True
        assert result.get("corrected_model") is None

    def test_non_openrouter_provider_unaffected(self):
        """The variant carve-out is OpenRouter-only; other providers keep
        their existing behavior for colon-suffixed names."""
        result = _validate(
            "x-ai/grok-4.6:nitro",
            "groq",
            api_models=["x-ai/grok-4.6"],
        )
        assert result.get("corrected_model") != "x-ai/grok-4.6:nitro"

    def test_static_catalog_fallback_accepts_variant(self):
        """Gateway path: /models unreachable → static catalog validates the
        base id and preserves the suffix."""
        with patch("hermes_cli.models.fetch_api_models", return_value=None), \
             patch(
                 "hermes_cli.models.provider_model_ids",
                 return_value=["x-ai/grok-4.6", "anthropic/claude-opus-4.6"],
             ):
            result = validate_requested_model(
                "x-ai/grok-4.6:floor",
                "openrouter",
                base_url="https://openrouter.ai/api/v1",
            )
        assert result["accepted"] is True
        assert result["recognized"] is True
        assert result.get("corrected_model") is None


class TestValidateRequestedModelNousPortalRecommendations:
    """Regression tests for issue #71312: the Nous Telegram picker (and any
    other messaging-platform /model validation, since they all share
    validate_requested_model()) rejected models that are live Nous Portal
    recommendations (/api/nous/recommended-models) but not yet in the
    hardcoded curated catalog -- even though `hermes chat` already accepts
    these via union_with_portal_free/paid_recommendations() at model-list
    build time. The per-message validation path now checks the same Portal
    feed as a fallback tier before rejecting, so Telegram/CLI agree.
    """

    PORTAL_PAYLOAD = {
        "freeRecommendedModels": [
            {"modelName": "inclusionai/ling-3.0-flash:free"},
        ],
        "paidRecommendedModels": [
            {"modelName": "inclusionai/ling-3.0-pro"},
        ],
    }

    def _validate_nous(self, model, api_models=None, portal_payload=None, portal_raises=False):
        api_models = api_models if api_models is not None else ["inclusionai/ling-2.6-flash"]
        probe_payload = {
            "models": api_models,
            "probed_url": "https://portal.nousresearch.com/v1/models",
            "resolved_base_url": "https://portal.nousresearch.com/v1",
            "suggested_base_url": None,
            "used_fallback": False,
        }

        def _fetch_portal(*a, **kw):
            if portal_raises:
                raise RuntimeError("portal unreachable")
            return portal_payload if portal_payload is not None else self.PORTAL_PAYLOAD

        with patch("hermes_cli.models.fetch_api_models", return_value=api_models), \
             patch("hermes_cli.models.probe_api_models", return_value=probe_payload), \
             patch("hermes_cli.models.fetch_nous_recommended_models", side_effect=_fetch_portal), \
             patch("hermes_cli.models._resolve_nous_portal_url", return_value="https://portal.nousresearch.com"), \
             patch("hermes_cli.models._model_in_provider_catalog", return_value=False):
            return validate_requested_model(model, "nous")

    def test_free_portal_recommendation_accepted(self):
        """The exact scenario from #71312: a free-tier Portal recommendation
        missing from the curated catalog and the live /v1/models listing
        must be accepted, not rejected."""
        result = self._validate_nous("inclusionai/ling-3.0-flash:free")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert "Portal recommendation" in (result["message"] or "")

    def test_paid_portal_recommendation_accepted(self):
        result = self._validate_nous("inclusionai/ling-3.0-pro")
        assert result["accepted"] is True

    def test_model_absent_from_portal_and_catalog_still_rejected(self):
        """A model that's genuinely nowhere (not live, not curated, not a
        Portal recommendation) must still be rejected -- this fallback
        tier must not make validation permissive for everything."""
        result = self._validate_nous("totally-made-up-model-xyz")
        assert result["accepted"] is False
        assert result["recognized"] is False

    def test_portal_fetch_failure_falls_through_to_rejection_not_crash(self):
        """A network/parse failure fetching the Portal feed must not crash
        validation -- it degrades to the existing rejection path."""
        result = self._validate_nous(
            "inclusionai/ling-3.0-flash:free", portal_raises=True
        )
        assert result["accepted"] is False  # fails closed, doesn't crash

    def test_non_string_model_name_entries_ignored(self):
        """Malformed Portal entries (non-string / empty modelName) must be
        skipped via _extract_model_name -- never stringified into garbage
        matches (e.g. an int modelName 5 must not accept a model named "5")."""
        payload = {
            "freeRecommendedModels": [
                {"modelName": 5},
                {"modelName": ""},
                {"modelName": None},
                "not-a-dict",
                {"modelName": "inclusionai/ling-3.0-flash:free"},
            ],
            "paidRecommendedModels": [],
        }
        assert self._validate_nous("5", portal_payload=payload)["accepted"] is False
        result = self._validate_nous(
            "inclusionai/ling-3.0-flash:free", portal_payload=payload
        )
        assert result["accepted"] is True

    def test_non_nous_provider_does_not_consult_portal_feed(self):
        """This fallback tier is Nous-specific; a non-Nous provider must
        not have its rejection changed by (or trigger a call to) the Nous
        Portal feed."""
        probe_payload = {
            "models": ["some/other-model"],
            "probed_url": "https://api.example.com/v1/models",
            "resolved_base_url": "https://api.example.com/v1",
            "suggested_base_url": None,
            "used_fallback": False,
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["some/other-model"]), \
             patch("hermes_cli.models.probe_api_models", return_value=probe_payload), \
             patch("hermes_cli.models.fetch_nous_recommended_models") as mock_portal, \
             patch("hermes_cli.models._model_in_provider_catalog", return_value=False):
            result = validate_requested_model("inclusionai/ling-3.0-flash:free", "openrouter")
        mock_portal.assert_not_called()
        assert result["accepted"] is False

    def test_curated_catalog_hit_short_circuits_before_portal_check(self):
        """When the curated-catalog fallback already accepts the model, the
        Portal feed should not need to be consulted at all (cheaper, and
        avoids an unnecessary network call on the common path)."""
        api_models = ["inclusionai/ling-2.6-flash"]
        probe_payload = {
            "models": api_models, "probed_url": "x", "resolved_base_url": "x",
            "suggested_base_url": None, "used_fallback": False,
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=api_models), \
             patch("hermes_cli.models.probe_api_models", return_value=probe_payload), \
             patch("hermes_cli.models._model_in_provider_catalog", return_value=True), \
             patch("hermes_cli.models.fetch_nous_recommended_models") as mock_portal:
            result = validate_requested_model("inclusionai/ling-2.6-flash", "nous")
        mock_portal.assert_not_called()
        assert result["accepted"] is True
