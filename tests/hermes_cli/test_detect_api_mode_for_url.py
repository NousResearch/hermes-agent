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


class TestRuntimeAndLeafDetectorsAgree:
    """The resolver (``runtime_provider``) and the leaf (``provider_resolution``)
    must select the SAME api_mode for every URL — the Epic previously shipped
    divergent copies (cpf-zkw.10). This asserts behavioral agreement across a
    representative URL set (incl. the query/fragment-spoof and lookalike cases)
    so the two can never drift apart unnoticed, regardless of how they're wired.
    """

    URLS = [
        "https://api.openai.com/v1",
        "https://api.x.ai/v1",
        "https://api.minimax.io/anthropic",
        "https://api.kimi.com/coding",
        "https://gw.example.com/anthropic?api-version=1",
        "https://api.example.com/v1?next=/anthropic",
        "https://api.example.com/v1#/anthropic",
        "https://api.openai.com.attacker.test/v1",
        "http://localhost:11434/v1",
        "https://api.together.xyz/v1",
        "",
    ]

    def test_detect_api_mode_agrees_for_every_url(self):
        from hermes_cli import provider_resolution, runtime_provider

        for url in self.URLS:
            assert (
                runtime_provider._detect_api_mode_for_url(url)
                == provider_resolution._detect_api_mode_for_url(url)
            ), f"detectors disagree on {url!r}"

    def test_parse_api_mode_agrees_for_every_value(self):
        from hermes_cli import provider_resolution, runtime_provider

        for raw in ("chat_completions", "anthropic_messages", "codex_responses",
                    "codex_app_server", "  Anthropic_Messages ", "bogus", "", None):
            assert (
                runtime_provider._parse_api_mode(raw)
                == provider_resolution._parse_api_mode(raw)
            ), f"parse disagrees on {raw!r}"

    def test_codex_app_server_is_a_valid_mode_in_both(self):
        from hermes_cli import provider_resolution, runtime_provider

        assert "codex_app_server" in runtime_provider._VALID_API_MODES
        assert "codex_app_server" in provider_resolution.VALID_API_MODES


class TestQueryAndFragmentCannotSpoof:
    """Detection matches on the URL PATH only, so a query/fragment value can't
    masquerade as a protocol suffix. (The previous full-URL copy matched
    ``?x=/anthropic`` and ``#/anthropic`` as ``anthropic_messages``.)
    """

    def test_query_string_anthropic_does_not_match(self):
        assert _detect_api_mode_for_url("https://api.example.com/v1?next=/anthropic") is None

    def test_fragment_anthropic_does_not_match(self):
        assert _detect_api_mode_for_url("https://api.example.com/v1#/anthropic") is None

    def test_query_string_coding_on_kimi_does_not_match(self):
        assert _detect_api_mode_for_url("https://api.kimi.com/v1?redirect=/coding") is None

    def test_legit_anthropic_gateway_with_query_still_matches(self):
        # A real /anthropic gateway carrying a query string is correctly
        # detected (the full-URL copy used to MISS this → 404).
        assert (
            _detect_api_mode_for_url("https://gw.example.com/anthropic?api-version=1")
            == "anthropic_messages"
        )
