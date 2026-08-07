"""Tests for provider-aware `/model` validation in hermes_cli.models."""

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

    def test_probe_api_models_parses_bare_array_response(self):
        """Together AI returns ``[{"id": ...}, ...]`` (no ``data`` envelope).

        Regression: the parser used to call ``data.get("data", [])`` on the
        payload, raising ``AttributeError`` on a list, which the broad
        ``except Exception`` swallowed — reporting the endpoint unreachable.
        """
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'[{"id": "moonshotai/Kimi-K3"}, {"id": "deepseek-ai/DeepSeek-V4-Pro"}]'

        with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=_Resp()):
            probe = probe_api_models("sk-test", "https://api.together.xyz/v1")

        assert probe["models"] == ["moonshotai/Kimi-K3", "deepseek-ai/DeepSeek-V4-Pro"]
        assert probe["probed_url"] == "https://api.together.xyz/v1/models"
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


