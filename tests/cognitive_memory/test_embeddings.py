"""Tests for cognitive_memory.embeddings module."""

import math
from unittest.mock import MagicMock, patch

import pytest

from cognitive_memory.embeddings import (
    LiteLLMEmbedder,
    cosine_similarity,
    get_embedder,
)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert cosine_similarity(a, b) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_similar_vectors_high_score(self):
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.1]
        sim = cosine_similarity(a, b)
        assert sim > 0.99

    def test_different_vectors_low_score(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# LiteLLMEmbedder
# ---------------------------------------------------------------------------


def _make_mock_response(embeddings):
    """Create a mock litellm embedding response."""
    resp = MagicMock()
    resp.data = [{"embedding": emb} for emb in embeddings]
    return resp


class TestLiteLLMEmbedder:
    def test_embed_text_returns_vector(self):
        fake_emb = [0.1, 0.2, 0.3]
        with patch("litellm.embedding", return_value=_make_mock_response([fake_emb])):
            embedder = LiteLLMEmbedder(model="test-model")
            result = embedder.embed_text("hello world")

        assert result == fake_emb

    def test_embed_text_updates_dimensions(self):
        fake_emb = [0.1] * 384
        with patch("litellm.embedding", return_value=_make_mock_response([fake_emb])):
            embedder = LiteLLMEmbedder(model="test-model", dimensions=1536)
            embedder.embed_text("hello")

        assert embedder.dimensions == 384

    def test_embed_batch_returns_list(self):
        embs = [[0.1, 0.2], [0.3, 0.4]]
        with patch("litellm.embedding", return_value=_make_mock_response(embs)):
            embedder = LiteLLMEmbedder(model="test-model")
            result = embedder.embed_batch(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    def test_embed_batch_empty_list(self):
        embedder = LiteLLMEmbedder(model="test-model")
        assert embedder.embed_batch([]) == []

    def test_embed_text_passes_api_key(self):
        fake_emb = [0.1]
        with patch("litellm.embedding", return_value=_make_mock_response([fake_emb])) as mock_emb:
            embedder = LiteLLMEmbedder(model="test-model", api_key="sk-test")
            embedder.embed_text("hello")

        mock_emb.assert_called_once()
        assert mock_emb.call_args.kwargs["api_key"] == "sk-test"

    def test_embed_text_passes_api_base(self):
        fake_emb = [0.1]
        with patch("litellm.embedding", return_value=_make_mock_response([fake_emb])) as mock_emb:
            embedder = LiteLLMEmbedder(model="test-model", api_base="http://localhost:8080")
            embedder.embed_text("hello")

        assert mock_emb.call_args.kwargs["api_base"] == "http://localhost:8080"

    def test_embed_text_raises_on_failure(self):
        with patch("litellm.embedding", side_effect=RuntimeError("API down")):
            embedder = LiteLLMEmbedder(model="test-model")
            with pytest.raises(RuntimeError, match="API down"):
                embedder.embed_text("hello")

    def test_dimensions_property(self):
        embedder = LiteLLMEmbedder(dimensions=768)
        assert embedder.dimensions == 768


# ---------------------------------------------------------------------------
# get_embedder factory
# ---------------------------------------------------------------------------


class TestGetEmbedder:
    def test_default_config(self):
        embedder = get_embedder()
        assert isinstance(embedder, LiteLLMEmbedder)
        assert embedder._model == "text-embedding-3-small"

    def test_custom_model(self):
        config = {"embedding": {"model": "text-embedding-3-large"}}
        embedder = get_embedder(config)
        assert embedder._model == "text-embedding-3-large"

    def test_custom_api_key(self):
        config = {"embedding": {"api_key": "sk-custom"}}
        embedder = get_embedder(config)
        assert embedder._api_key == "sk-custom"

    def test_empty_config(self):
        embedder = get_embedder({})
        assert isinstance(embedder, LiteLLMEmbedder)

    def test_none_config(self):
        embedder = get_embedder(None)
        assert isinstance(embedder, LiteLLMEmbedder)
