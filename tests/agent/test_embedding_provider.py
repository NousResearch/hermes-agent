"""Tests for agent/embedding_provider.py — Strategy pattern for embedding generation."""

import pytest
from unittest.mock import MagicMock, patch


class TestEmbeddingProviderInterface:
    """The Strategy interface contract."""

    def test_provider_has_generate_embedding_method(self):
        """Every provider must expose generate_embedding(text) -> list[float]."""
        from agent.embedding_provider import EmbeddingProvider

        # Abstract base class defines the contract
        assert hasattr(EmbeddingProvider, "generate_embedding")
        import inspect
        sig = inspect.signature(EmbeddingProvider.generate_embedding)
        params = list(sig.parameters.keys())
        assert "text" in params
        assert sig.return_annotation == "list[float]"

    def test_provider_has_dimensions_property(self):
        """Every provider must report its vector dimensions."""
        from agent.embedding_provider import EmbeddingProvider
        assert hasattr(EmbeddingProvider, "dimensions")


class TestOllamaEmbeddingProvider:
    """Ollama local embedding provider."""

    def test_generate_embedding_returns_correct_dimensions(self):
        """Returns a vector of the expected length for the model."""
        from agent.embedding_provider import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(
            model="nomic-embed-text",
            base_url="http://localhost:11434/v1",
            dimensions=768,
        )
        assert provider.dimensions == 768

    def test_generate_embedding_calls_ollama_api(self):
        """Uses OpenAI-compatible embeddings endpoint on Ollama."""
        from agent.embedding_provider import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(
            model="nomic-embed-text",
            base_url="http://localhost:11434/v1",
            dimensions=768,
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 768)]
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(provider, "_get_client", return_value=mock_client):
            result = provider.generate_embedding("test text")

        assert len(result) == 768
        mock_client.embeddings.create.assert_called_once_with(
            model="nomic-embed-text",
            input="test text",
        )

    def test_generate_embedding_handles_empty_text(self):
        """Empty text returns a zero vector, not an error."""
        from agent.embedding_provider import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(
            model="nomic-embed-text",
            base_url="http://localhost:11434/v1",
            dimensions=768,
        )
        result = provider.generate_embedding("")
        assert len(result) == 768
        assert all(v == 0.0 for v in result)


class TestOpenAIEmbeddingProvider:
    """OpenAI API embedding provider."""

    def test_generate_embedding_returns_correct_dimensions(self):
        """Returns a vector of the expected length for text-embedding-3-small."""
        from agent.embedding_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            model="text-embedding-3-small",
            api_key="sk-test",
            dimensions=1536,
        )
        assert provider.dimensions == 1536

    def test_generate_embedding_calls_openai_api(self):
        """Uses OpenAI embeddings endpoint."""
        from agent.embedding_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            model="text-embedding-3-small",
            api_key="sk-test",
            dimensions=1536,
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(provider, "_get_client", return_value=mock_client):
            result = provider.generate_embedding("test text")

        assert len(result) == 1536
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test text",
        )


class TestResolveEmbeddingProvider:
    """Factory function for selecting the right strategy."""

    def test_resolve_ollama_from_config(self):
        """Returns OllamaEmbeddingProvider when provider=ollama."""
        from agent.embedding_provider import resolve_embedding_provider

        config = {
            "provider": "ollama",
            "model": "nomic-embed-text",
            "base_url": "http://localhost:11434/v1",
            "api_key": "",
            "dimensions": 768,
        }
        provider = resolve_embedding_provider(config)
        from agent.embedding_provider import OllamaEmbeddingProvider
        assert isinstance(provider, OllamaEmbeddingProvider)
        assert provider.dimensions == 768

    def test_resolve_openai_from_config(self):
        """Returns OpenAIEmbeddingProvider when provider=openai."""
        from agent.embedding_provider import resolve_embedding_provider

        config = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "base_url": "",
            "api_key": "sk-test",
            "dimensions": 1536,
        }
        provider = resolve_embedding_provider(config)
        from agent.embedding_provider import OpenAIEmbeddingProvider
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.dimensions == 1536

    def test_resolve_returns_none_for_unknown_provider(self):
        """Returns None when provider type is unrecognized."""
        from agent.embedding_provider import resolve_embedding_provider

        config = {
            "provider": "unknown_backend",
            "model": "some-model",
            "dimensions": 384,
        }
        provider = resolve_embedding_provider(config)
        assert provider is None

    def test_resolve_returns_none_for_empty_config(self):
        """Returns None when config is empty or missing."""
        from agent.embedding_provider import resolve_embedding_provider

        assert resolve_embedding_provider({}) is None
        assert resolve_embedding_provider(None) is None
