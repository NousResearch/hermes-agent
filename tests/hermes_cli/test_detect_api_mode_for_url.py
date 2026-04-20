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

from hermes_cli.runtime_provider import _detect_api_mode_for_url


class TestCodexResponsesDetection:
    def test_openai_api_returns_codex_responses(self):
        assert _detect_api_mode_for_url("https://api.openai.com/v1") == "codex_responses"

    def test_xai_api_returns_codex_responses(self):
        assert _detect_api_mode_for_url("https://api.x.ai/v1") == "codex_responses"

    def test_openrouter_is_not_codex_responses(self):
        # api.openai.com check must exclude openrouter (which routes to openai-hosted models).
        assert _detect_api_mode_for_url("https://openrouter.ai/api/v1") is None


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

    def test_uppercase_path_tolerated(self):
        assert _detect_api_mode_for_url("https://API.MINIMAX.IO/Anthropic") == "anthropic_messages"

    def test_anthropic_in_middle_of_path_does_not_match(self):
        # The helper requires ``/anthropic`` as the path SUFFIX, not anywhere.
        # Protects against false positives on e.g. /anthropic/v1/models.
        assert _detect_api_mode_for_url("https://api.example.com/anthropic/v1") is None


class TestDefaultCase:
    def test_generic_url_returns_none(self):
        assert _detect_api_mode_for_url("https://api.together.xyz/v1") is None

    def test_empty_string_returns_none(self):
        assert _detect_api_mode_for_url("") is None

    def test_none_returns_none(self):
        assert _detect_api_mode_for_url(None) is None

    def test_localhost_returns_none(self):
        assert _detect_api_mode_for_url("http://localhost:11434/v1") is None
