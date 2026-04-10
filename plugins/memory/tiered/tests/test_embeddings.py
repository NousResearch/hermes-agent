"""Tests for embeddings.py — Ollama embedding client."""

import unittest
from unittest.mock import MagicMock, patch

import httpx


class TestGenerateEmbedding(unittest.TestCase):
    """Tests for generate_embedding (single text)."""

    @patch("plugins.memory.tiered.embeddings.httpx.post")
    def test_generate_embedding_success(self, mock_post: MagicMock) -> None:
        """Successful 1024-dim response returns list of 1024 floats."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "embeddings": [[0.1] * 1024],
            "model": "qllama/bge-small-en-v1.5",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from plugins.memory.tiered.embeddings import generate_embedding

        result = generate_embedding("hello world")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1024)
        self.assertIsInstance(result[0], float)

    def test_generate_embedding_empty_text(self) -> None:
        """Empty string returns None without making any HTTP call."""
        from plugins.memory.tiered.embeddings import generate_embedding

        result = generate_embedding("")
        self.assertIsNone(result)

    @patch("plugins.memory.tiered.embeddings.httpx.post")
    def test_generate_embedding_connection_error(self, mock_post: MagicMock) -> None:
        """Connection error returns None (FTS-only fallback)."""
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        from plugins.memory.tiered.embeddings import generate_embedding

        result = generate_embedding("hello world")
        self.assertIsNone(result)

    @patch("plugins.memory.tiered.embeddings.httpx.post")
    def test_generate_embedding_timeout(self, mock_post: MagicMock) -> None:
        """Timeout returns None (FTS-only fallback)."""
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        from plugins.memory.tiered.embeddings import generate_embedding

        result = generate_embedding("hello world")
        self.assertIsNone(result)

    @patch("plugins.memory.tiered.embeddings.httpx.post")
    def test_generate_embedding_wrong_dims(self, mock_post: MagicMock) -> None:
        """Response with wrong dimensions (128 instead of 1024) returns None."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "embeddings": [[0.1] * 128],
            "model": "qllama/bge-small-en-v1.5",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from plugins.memory.tiered.embeddings import generate_embedding

        result = generate_embedding("hello world")
        self.assertIsNone(result)

    @patch("plugins.memory.tiered.embeddings.httpx.post")
    def test_generate_embedding_truncation(self, mock_post: MagicMock) -> None:
        """Text > 8000 chars gets truncated before sending to API."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "embeddings": [[0.1] * 1024],
            "model": "qllama/bge-small-en-v1.5",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from plugins.memory.tiered.embeddings import generate_embedding

        long_text = "x" * 10000
        generate_embedding(long_text)

        # Verify the POST call received truncated text (8000 chars)
        call_args = mock_post.call_args
        sent_text = call_args.kwargs.get("json", call_args[1].get("json", {}))["input"]
        self.assertEqual(len(sent_text), 8000)


class TestGenerateEmbeddings(unittest.TestCase):
    """Tests for generate_embeddings (batch)."""

    @patch("plugins.memory.tiered.embeddings.httpx.post")
    def test_generate_embeddings_batch(self, mock_post: MagicMock) -> None:
        """Batch request with 2 texts returns 2 embeddings."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "embeddings": [[0.1] * 1024, [0.2] * 1024],
            "model": "qllama/bge-small-en-v1.5",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from plugins.memory.tiered.embeddings import generate_embeddings

        result = generate_embeddings(["hello", "world"])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1024)
        self.assertEqual(len(result[1]), 1024)

    def test_generate_embeddings_empty_list(self) -> None:
        """Empty list returns empty list without making any HTTP call."""
        from plugins.memory.tiered.embeddings import generate_embeddings

        result = generate_embeddings([])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
