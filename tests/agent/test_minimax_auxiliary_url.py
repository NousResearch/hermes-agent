"""Tests for MiniMax auxiliary client URL normalization.

MiniMax and MiniMax-CN set inference_base_url to the /anthropic path.
The auxiliary client uses the OpenAI SDK, which needs /v1 instead.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.auxiliary_client import _to_openai_base_url, _url_has_versioned_path


class TestUrlHasVersionedPath:
    def test_empty_url(self):
        assert _url_has_versioned_path("") is False

    def test_no_path(self):
        assert _url_has_versioned_path("https://example.com") is False

    def test_root_path(self):
        assert _url_has_versioned_path("https://example.com/") is False

    def test_v1_path(self):
        assert _url_has_versioned_path("https://example.com/v1") is True

    def test_v1_trailing_slash(self):
        assert _url_has_versioned_path("https://example.com/v1/") is True

    def test_api_v1_path(self):
        assert _url_has_versioned_path("https://example.com/api/v1") is True

    def test_v1beta_path(self):
        assert _url_has_versioned_path("https://example.com/v1beta") is True

    def test_v2023_path(self):
        assert _url_has_versioned_path("https://example.com/v2023-01-01") is True

    def test_v2_path(self):
        assert _url_has_versioned_path("https://example.com/v2") is True

    def test_anthropic_path(self):
        """/anthropic is not a versioned path — contains no digits after 'v'."""
        assert _url_has_versioned_path("https://api.minimax.io/anthropic") is False

    def test_custom_non_versioned_path(self):
        """A custom path like /proxy/openai is not versioned."""
        assert _url_has_versioned_path("https://proxy.example.com/openai") is False

    def test_custom_deep_path(self):
        """Deep non-versioned path should not match."""
        assert _url_has_versioned_path("https://example.com/a/b/c") is False

    def test_ip_with_port_no_path(self):
        assert _url_has_versioned_path("http://192.168.1.1:8000") is False

    def test_ip_with_v1(self):
        assert _url_has_versioned_path("http://192.168.1.1:8000/v1") is True

    def test_localhost_no_path(self):
        assert _url_has_versioned_path("http://localhost:8000") is False

    def test_localhost_v1(self):
        assert _url_has_versioned_path("http://localhost:8000/v1") is True


class TestToOpenaiBaseUrl:
    def test_minimax_global_anthropic_suffix_replaced(self):
        assert _to_openai_base_url("https://api.minimax.io/anthropic") == "https://api.minimax.io/v1"

    def test_minimax_cn_anthropic_suffix_replaced(self):
        assert _to_openai_base_url("https://api.minimaxi.com/anthropic") == "https://api.minimaxi.com/v1"

    def test_trailing_slash_stripped_before_replace(self):
        assert _to_openai_base_url("https://api.minimax.io/anthropic/") == "https://api.minimax.io/v1"

    def test_v1_url_unchanged(self):
        assert _to_openai_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"

    def test_openrouter_url_unchanged(self):
        assert _to_openai_base_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/v1"

    def test_anthropic_domain_v1_appended(self):
        """api.anthropic.com has no versioned path — /v1 is now appended for OpenAI-wire use."""
        assert _to_openai_base_url("https://api.anthropic.com") == "https://api.anthropic.com/v1"

    def test_anthropic_in_subpath_v1_appended(self):
        """URLs with /anthropic as a non-suffix subpath get /v1 appended
        since they don't end with /anthropic and have no versioned segment."""
        assert _to_openai_base_url("https://example.com/anthropic/extra") == "https://example.com/anthropic/extra/v1"

    def test_localhost_v1_appended(self):
        """The canonical fix for #65488: bare localhost gets /v1."""
        assert _to_openai_base_url("http://localhost:8000") == "http://localhost:8000/v1"

    def test_localhost_v1_present_unchanged(self):
        assert _to_openai_base_url("http://localhost:8000/v1") == "http://localhost:8000/v1"

    def test_bare_ip_v1_appended(self):
        assert _to_openai_base_url("http://192.168.1.1:8080") == "http://192.168.1.1:8080/v1"

    def test_bare_ip_with_v1_unchanged(self):
        assert _to_openai_base_url("http://192.168.1.1:8080/v1") == "http://192.168.1.1:8080/v1"

    def test_deepseek_v1_appended(self):
        assert _to_openai_base_url("https://api.deepseek.com") == "https://api.deepseek.com/v1"

    def test_groq_v1_appended(self):
        """Groq uses /openai/v1 — already versioned, so unchanged."""
        assert _to_openai_base_url("https://api.groq.com/openai/v1") == "https://api.groq.com/openai/v1"

    def test_custom_non_standard_path_unchanged(self):
        """A path segment starting with 'v' followed by a digit is treated as versioned."""
        assert _to_openai_base_url("https://proxy.example.com/proxy/v1") == "https://proxy.example.com/proxy/v1"

    def test_empty_string(self):
        assert _to_openai_base_url("") == ""

    def test_none(self):
        assert _to_openai_base_url(None) == ""
