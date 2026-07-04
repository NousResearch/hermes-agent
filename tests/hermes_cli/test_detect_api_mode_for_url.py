"""Tests for hermes_cli.runtime_provider._detect_api_mode_for_url.

The helper maps base URLs to api_modes for four cases:
  * api.openai.com    → codex_responses
  * api.x.ai          → codex_responses
  * api.anthropic.com → anthropic_messages (Pro/Max OAuth is only billed
                                            against /v1/messages; the
                                            chat_completions shim counts
                                            against a separate empty
                                            "extra usage" pool, see #32243)
  * */anthropic       → anthropic_messages (third-party gateways like MiniMax,
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

    def test_openai_host_suffix_does_not_match(self):
        assert _detect_api_mode_for_url("https://api.openai.com.example/v1") is None

    def test_openai_path_segment_does_not_match(self):
        assert _detect_api_mode_for_url("https://proxy.example.test/api.openai.com/v1") is None

    def test_xai_host_suffix_does_not_match(self):
        assert _detect_api_mode_for_url("https://api.x.ai.example/v1") is None


class TestDirectAnthropicHost:
    """Native api.anthropic.com → /v1/messages. Pinned for issue #32243.

    The Anthropic OpenAI-compat ``/chat/completions`` shim on the same
    host bills against a separate "extra usage" pool that Pro/Max OAuth
    subscriptions don't fund, so a fresh OAuth credential 400s with
    "out of extra usage" the moment a request lands there. The detector
    must keep ``api.anthropic.com`` on the native Messages API.
    """

    def test_bare_host(self):
        assert _detect_api_mode_for_url("https://api.anthropic.com") == "anthropic_messages"

    def test_with_trailing_slash(self):
        assert _detect_api_mode_for_url("https://api.anthropic.com/") == "anthropic_messages"

    def test_with_v1_suffix(self):
        # The Anthropic SDK appends /v1/messages itself but the user's
        # config may persist the /v1 form — must still resolve.
        assert _detect_api_mode_for_url("https://api.anthropic.com/v1") == "anthropic_messages"

    def test_uppercase_host_tolerated(self):
        assert _detect_api_mode_for_url("https://API.ANTHROPIC.COM/v1") == "anthropic_messages"

    def test_lookalike_subdomain_does_not_match(self):
        # ``api.anthropic.com.attacker.test`` is an attacker-controlled
        # host; the registrable label is ``attacker``, not Anthropic.
        # Must NOT be routed to anthropic_messages — leaking an
        # Anthropic OAuth token there is the worst case.
        assert (
            _detect_api_mode_for_url("https://api.anthropic.com.attacker.test/v1")
            is None
        )

    def test_anthropic_path_segment_does_not_match(self):
        # A reverse proxy under an unrelated host whose path *contains*
        # ``api.anthropic.com`` should not be classified as native.
        assert (
            _detect_api_mode_for_url("https://proxy.example.test/api.anthropic.com/v1")
            is None
        )


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


class TestDefaultCase:
    def test_generic_url_returns_none(self):
        assert _detect_api_mode_for_url("https://api.together.xyz/v1") is None

    def test_empty_string_returns_none(self):
        assert _detect_api_mode_for_url("") is None

    def test_none_returns_none(self):
        assert _detect_api_mode_for_url(None) is None

    def test_localhost_returns_none(self):
        assert _detect_api_mode_for_url("http://localhost:11434/v1") is None
