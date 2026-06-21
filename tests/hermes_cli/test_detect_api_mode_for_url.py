"""Tests for hermes_cli.runtime_provider._detect_api_mode_for_url.

The helper maps base URLs to api_modes for three cases:
  * api.openai.com  → codex_responses
  * api.x.ai        → codex_responses
  * */anthropic     → anthropic_messages (third-party gateways like MiniMax,
                                          Zhipu GLM, LiteLLM proxies)

Consolidating the /anthropic detection in this helper (instead of three
inline ``endswith`` checks spread across _resolve_runtime_from_pool_entry,
the explicit-provider path, and the api-key-provider path) means every
future update to the detection logic lives in one place.
"""

from __future__ import annotations

from hermes_cli.models import openai_model_api_mode
from hermes_cli.runtime_provider import _detect_api_mode_for_url


class TestCodexResponsesDetection:
    def test_openai_api_returns_codex_responses(self):
        # No model supplied → historical Responses-API default is preserved.
        assert _detect_api_mode_for_url("https://api.openai.com/v1") == "codex_responses"

    def test_xai_api_returns_codex_responses(self):
        assert _detect_api_mode_for_url("https://api.x.ai/v1") == "codex_responses"

    def test_openrouter_is_not_codex_responses(self):
        # api.openai.com check must exclude openrouter (which routes to openai-hosted models).
        assert _detect_api_mode_for_url("https://openrouter.ai/api/v1") is None

    def test_openai_host_suffix_does_not_match(self):
        assert _detect_api_mode_for_url("https://api.openai.com.example/v1") is None

    def test_openai_path_segment_does_not_match(self):
        assert _detect_api_mode_for_url("https://proxy.example.test/api.openai.com/v1") is None

    def test_xai_host_suffix_does_not_match(self):
        assert _detect_api_mode_for_url("https://api.x.ai.example/v1") is None


class TestAnthropicMessagesDetection:
    """Third-party gateways that speak the Anthropic protocol under /anthropic."""

    def test_minimax_anthropic_endpoint(self):
        assert _detect_api_mode_for_url("https://api.minimax.io/anthropic") == "anthropic_messages"

    def test_minimax_cn_anthropic_endpoint(self):
        assert _detect_api_mode_for_url("https://api.minimaxi.com/anthropic") == "anthropic_messages"

    def test_dashscope_anthropic_endpoint(self):
        assert (
            _detect_api_mode_for_url("https://dashscope.aliyuncs.com/api/v2/apps/anthropic")
            == "anthropic_messages"
        )

    def test_trailing_slash_tolerated(self):
        assert _detect_api_mode_for_url("https://api.minimax.io/anthropic/") == "anthropic_messages"

    def test_versioned_anthropic_base_url_tolerated(self):
        assert _detect_api_mode_for_url("https://proxy.example.com/anthropic/v1") == "anthropic_messages"

    def test_uppercase_path_tolerated(self):
        assert _detect_api_mode_for_url("https://API.MINIMAX.IO/Anthropic") == "anthropic_messages"

    def test_anthropic_endpoint_subpath_does_not_match(self):
        # The helper requires ``/anthropic`` as the path SUFFIX, not anywhere.
        # Protects against false positives on e.g. /anthropic/v1/models.
        assert _detect_api_mode_for_url("https://api.example.com/anthropic/v1/models") is None


class TestOpenAIModelAwareDetection:
    """api.openai.com serves reasoning families on the Responses API and chat
    families on Chat Completions — the helper must pick per-model when a model
    is known (regression: GPT-4.1 / GPT-4o 400'd on the forced Responses API).
    """

    def test_chat_model_uses_chat_completions(self):
        assert _detect_api_mode_for_url("https://api.openai.com/v1", "gpt-4.1") == "chat_completions"

    def test_gpt_4o_uses_chat_completions(self):
        assert _detect_api_mode_for_url("https://api.openai.com/v1", "gpt-4o") == "chat_completions"

    def test_reasoning_model_uses_codex_responses(self):
        assert _detect_api_mode_for_url("https://api.openai.com/v1", "gpt-5.5") == "codex_responses"

    def test_o_series_uses_codex_responses(self):
        assert _detect_api_mode_for_url("https://api.openai.com/v1", "o4-mini") == "codex_responses"

    def test_unknown_model_preserves_responses_default(self):
        assert _detect_api_mode_for_url("https://api.openai.com/v1", "") == "codex_responses"


class TestOpenAIModelApiMode:
    """Unit coverage for the shared hermes_cli.models.openai_model_api_mode helper."""

    def test_gpt_chat_families_are_chat_completions(self):
        for model in ("gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"):
            assert openai_model_api_mode(model) == "chat_completions", model

    def test_reasoning_families_are_codex_responses(self):
        for model in ("gpt-5", "gpt-5.5", "gpt-5.5-pro", "o1", "o1-preview", "o3-mini", "o4-mini", "codex-mini"):
            assert openai_model_api_mode(model) == "codex_responses", model

    def test_vendor_prefix_is_stripped(self):
        assert openai_model_api_mode("openai/gpt-4.1") == "chat_completions"
        assert openai_model_api_mode("openai/gpt-5.5") == "codex_responses"

    def test_unknown_or_empty_returns_none(self):
        assert openai_model_api_mode("") is None
        assert openai_model_api_mode(None) is None


class TestDefaultCase:
    def test_generic_url_returns_none(self):
        assert _detect_api_mode_for_url("https://api.together.xyz/v1") is None

    def test_empty_string_returns_none(self):
        assert _detect_api_mode_for_url("") is None

    def test_none_returns_none(self):
        assert _detect_api_mode_for_url(None) is None

    def test_localhost_returns_none(self):
        assert _detect_api_mode_for_url("http://localhost:11434/v1") is None
