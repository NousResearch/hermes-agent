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

from unittest.mock import MagicMock, patch

def test_search_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = GeminiWebSearchProvider()

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a grounded answer."
    
    mock_chunk = MagicMock()
    mock_chunk.web.uri = "https://example.com/source"
    mock_chunk.web.title = "Example Source"
    
    mock_candidate = MagicMock()
    mock_candidate.grounding_metadata.grounding_chunks = [mock_chunk]
    mock_response.candidates = [mock_candidate]
    
    mock_client.models.generate_content.return_value = mock_response

    with patch("plugins.web.gemini.provider._get_gemini_client", return_value=mock_client):
        result = provider.search("test query")
        
        assert result["success"]
        assert len(result["data"]["web"]) == 2
        
        ans = result["data"]["web"][0]
        assert ans["title"] == "Gemini Grounded Answer"
        assert ans["description"] == "This is a grounded answer."
        assert "test+query" in ans["url"]
        
        src = result["data"]["web"][1]
        assert src["url"] == "https://example.com/source"
        assert src["title"] == "Example Source"
        assert src["description"] == "Supporting source"

def test_extract_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = GeminiWebSearchProvider()

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is extracted content."
    mock_client.models.generate_content.return_value = mock_response

    with patch("plugins.web.gemini.provider._get_gemini_client", return_value=mock_client):
        results = provider.extract(["https://example.com/test"])
        
        assert len(results) == 1
        res = results[0]
        assert res["url"] == "https://example.com/test"
        assert res["content"] == "This is extracted content."
        assert res["title"] == "Gemini Extraction: https://example.com/test"
        assert "error" not in res
