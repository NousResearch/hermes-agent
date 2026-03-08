"""Tests for agent/embeddings.py — Embedder protocol, implementations, factory, utilities."""

import math
import pytest
from unittest.mock import MagicMock, patch

from agent.embeddings import (
    Embedder,
    FastEmbedEmbedder,
    OpenAIEmbedder,
    get_embedder,
    cosine_similarity,
    cosine_similarity_matrix,
)


# =========================================================================
# cosine_similarity
# =========================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0)

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
        b = [1.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimensions must match"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_similar_vectors(self):
        a = [1.0, 1.0]
        b = [1.0, 1.1]
        sim = cosine_similarity(a, b)
        assert 0.99 < sim < 1.0


# =========================================================================
# cosine_similarity_matrix
# =========================================================================

class TestCosineSimilarityMatrix:
    def test_diagonal_is_one(self):
        vecs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        matrix = cosine_similarity_matrix(vecs)
        for i in range(len(vecs)):
            assert matrix[i][i] == pytest.approx(1.0)

    def test_symmetry(self):
        vecs = [[1.0, 0.0], [0.5, 0.5]]
        matrix = cosine_similarity_matrix(vecs)
        assert matrix[0][1] == pytest.approx(matrix[1][0])

    def test_orthogonal_off_diagonal(self):
        vecs = [[1.0, 0.0], [0.0, 1.0]]
        matrix = cosine_similarity_matrix(vecs)
        assert matrix[0][1] == pytest.approx(0.0)

    def test_shape(self):
        vecs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        matrix = cosine_similarity_matrix(vecs)
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)


# =========================================================================
# FastEmbedEmbedder
# =========================================================================

class TestFastEmbedEmbedder:
    def test_default_model(self):
        emb = FastEmbedEmbedder()
        assert emb.model_name == FastEmbedEmbedder.DEFAULT_MODEL

    def test_custom_model(self):
        emb = FastEmbedEmbedder(model="custom-model")
        assert emb.model_name == "custom-model"

    def test_dimensions(self):
        emb = FastEmbedEmbedder()
        assert emb.dimensions == 384

    def test_lazy_load(self):
        emb = FastEmbedEmbedder()
        assert emb._model is None

    def test_import_error_if_not_installed(self):
        emb = FastEmbedEmbedder()
        with patch.dict("sys.modules", {"fastembed": None}):
            with pytest.raises(ImportError, match="fastembed is not installed"):
                emb._load()

    def test_embed_text(self):
        emb = FastEmbedEmbedder()
        mock_model = MagicMock()
        # Use a simple object with .tolist() instead of numpy array
        fake_vec = MagicMock()
        fake_vec.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.embed.return_value = iter([fake_vec])
        emb._model = mock_model
        result = emb.embed_text("hello")
        assert result == pytest.approx([0.1, 0.2, 0.3])

    def test_embed_texts(self):
        emb = FastEmbedEmbedder()
        mock_model = MagicMock()
        fake_vec1 = MagicMock()
        fake_vec1.tolist.return_value = [0.1, 0.2]
        fake_vec2 = MagicMock()
        fake_vec2.tolist.return_value = [0.3, 0.4]
        mock_model.embed.return_value = iter([fake_vec1, fake_vec2])
        emb._model = mock_model
        result = emb.embed_texts(["hello", "world"])
        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.3, 0.4])


# =========================================================================
# OpenAIEmbedder
# =========================================================================

class TestOpenAIEmbedder:
    def test_default_model(self):
        emb = OpenAIEmbedder()
        assert emb.model_name == OpenAIEmbedder.DEFAULT_MODEL

    def test_dimensions_known_model(self):
        assert OpenAIEmbedder(model="text-embedding-3-small").dimensions == 1536
        assert OpenAIEmbedder(model="text-embedding-3-large").dimensions == 3072

    def test_dimensions_unknown_model(self):
        assert OpenAIEmbedder(model="unknown-model").dimensions == 1536

    def test_lazy_load(self):
        emb = OpenAIEmbedder()
        assert emb._client is None

    def test_embed_text(self):
        emb = OpenAIEmbedder()
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3])
        ]
        emb._client = mock_client
        result = emb.embed_text("hello")
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            input=["hello"], model=OpenAIEmbedder.DEFAULT_MODEL
        )

    def test_embed_texts(self):
        emb = OpenAIEmbedder()
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        emb._client = mock_client
        result = emb.embed_texts(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]


# =========================================================================
# get_embedder factory
# =========================================================================

class TestGetEmbedder:
    def test_default_returns_fastembed(self):
        emb = get_embedder({})
        assert isinstance(emb, FastEmbedEmbedder)

    def test_local_provider(self):
        emb = get_embedder({"embeddings": {"provider": "local"}})
        assert isinstance(emb, FastEmbedEmbedder)

    def test_local_custom_model(self):
        emb = get_embedder({"embeddings": {"provider": "local", "model": "custom-model"}})
        assert isinstance(emb, FastEmbedEmbedder)
        assert emb.model_name == "custom-model"

    def test_openai_provider(self):
        emb = get_embedder({"embeddings": {"provider": "openai"}})
        assert isinstance(emb, OpenAIEmbedder)

    def test_openai_custom_model(self):
        emb = get_embedder({"embeddings": {"provider": "openai", "model": "text-embedding-3-large"}})
        assert isinstance(emb, OpenAIEmbedder)
        assert emb.model_name == "text-embedding-3-large"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedder({"embeddings": {"provider": "unknown"}})

    def test_embedder_protocol_compliance(self):
        emb = get_embedder({})
        assert isinstance(emb, Embedder)
