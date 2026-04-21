"""Targeted tests for ``utils.base_url_hostname``.

The helper is used across provider routing, auxiliary client, and setup
wizards to avoid the substring-match false-positive class documented in
tests/agent/test_direct_provider_url_detection.py.
"""

from __future__ import annotations

from utils import base_url_hostname


def test_empty_returns_empty_string():
    assert base_url_hostname("") == ""
    assert base_url_hostname(None) == ""  # type: ignore[arg-type]


def test_plain_host_without_scheme():
    assert base_url_hostname("api.openai.com") == "api.openai.com"
    assert base_url_hostname("api.openai.com/v1") == "api.openai.com"


def test_https_url_extracts_hostname_only():
    assert base_url_hostname("https://api.openai.com/v1") == "api.openai.com"
    assert base_url_hostname("https://api.x.ai/v1") == "api.x.ai"
    assert base_url_hostname("https://api.anthropic.com") == "api.anthropic.com"


def test_hostname_case_insensitive():
    assert base_url_hostname("https://API.OpenAI.com/v1") == "api.openai.com"


def test_trailing_dot_stripped():
    # Fully-qualified hostnames may include a trailing dot.
    assert base_url_hostname("https://api.openai.com./v1") == "api.openai.com"


def test_path_containing_provider_host_is_not_the_hostname():
    # The key regression — proxy paths must never be misread as the host.
    assert base_url_hostname("https://proxy.example.test/api.openai.com/v1") == "proxy.example.test"
    assert base_url_hostname("https://proxy.example.test/api.anthropic.com/v1") == "proxy.example.test"


def test_host_suffix_is_not_the_provider():
    # A hostname that merely ends with the provider domain is not the provider.
    assert base_url_hostname("https://api.openai.com.example/v1") == "api.openai.com.example"
    assert base_url_hostname("https://api.x.ai.example/v1") == "api.x.ai.example"


def test_port_is_ignored():
    assert base_url_hostname("https://api.openai.com:443/v1") == "api.openai.com"


def test_whitespace_stripped():
    assert base_url_hostname("  https://api.openai.com/v1  ") == "api.openai.com"
