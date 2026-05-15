"""Unit tests for the Gemini Web Search Provider."""

import pytest
from plugins.web.gemini.provider import GeminiWebSearchProvider, _reset_client_for_tests


@pytest.fixture(autouse=True)
def reset_gemini_client():
    _reset_client_for_tests()
    yield
    _reset_client_for_tests()


def test_is_available_without_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = GeminiWebSearchProvider()
    assert not provider.is_available()


def test_is_available_with_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = GeminiWebSearchProvider()
    assert provider.is_available()


def test_is_available_with_google_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    provider = GeminiWebSearchProvider()
    assert provider.is_available()


def test_supports_capabilities():
    provider = GeminiWebSearchProvider()
    assert provider.supports_search()
    assert provider.supports_extract()


def test_search_unconfigured(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = GeminiWebSearchProvider()
    result = provider.search("test query")
    assert not result["success"]
    assert "environment variable not set" in result["error"]


def test_extract_unconfigured(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = GeminiWebSearchProvider()
    results = provider.extract(["https://example.com"])
    assert len(results) == 1
    assert "environment variable not set" in results[0]["error"]
