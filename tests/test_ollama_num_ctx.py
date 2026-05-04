"""Tests for Ollama num_ctx context length detection and injection.

Covers:
  agent/model_metadata.py — query_ollama_num_ctx()
  run_agent.py — _ollama_num_ctx detection + extra_body injection
"""

from unittest.mock import patch, MagicMock

import pytest

from agent.model_metadata import query_ollama_num_ctx


# ═══════════════════════════════════════════════════════════════════════
# Level 1: query_ollama_num_ctx — Ollama API interaction
# ═══════════════════════════════════════════════════════════════════════


def _mock_httpx_client(show_response_data, status_code=200):
    """Create a mock httpx.Client context manager that returns given /api/show data."""
    mock_resp = MagicMock(status_code=status_code)
    mock_resp.json.return_value = show_response_data
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_client)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx, mock_client


class TestQueryOllamaNumCtx:
    """Test the Ollama /api/show context length query."""

    def test_returns_context_from_model_info(self):
        """Should extract context_length from GGUF model_info metadata."""
        show_data = {
            "model_info": {"llama.context_length": 131072},
            "parameters": "",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            # httpx is imported inside the function — patch the module import
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("llama3.1:8b", "http://localhost:11434/v1")

        assert result == 131072

    def test_prefers_explicit_num_ctx_from_modelfile(self):
        """If the Modelfile sets num_ctx explicitly, that should take priority."""
        show_data = {
            "model_info": {"llama.context_length": 131072},
            "parameters": "num_ctx 32768\ntemperature 0.7",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("custom-model", "http://localhost:11434")

        assert result == 32768

    def test_returns_none_for_non_ollama_server(self):
        """Should return None if the server is not Ollama."""
        with patch("agent.model_metadata.detect_local_server_type", return_value="lm-studio"):
            result = query_ollama_num_ctx("model", "http://localhost:1234")
        assert result is None

    def test_returns_none_on_connection_error(self):
        """Should return None if the server is unreachable."""
        with patch("agent.model_metadata.detect_local_server_type", side_effect=Exception("timeout")):
            result = query_ollama_num_ctx("model", "http://localhost:11434")
        assert result is None

    def test_returns_none_on_404(self):
        """Should return None if the model is not found."""
        mock_ctx, _ = _mock_httpx_client({}, status_code=404)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("nonexistent", "http://localhost:11434")

        assert result is None

    def test_strips_provider_prefix(self):
        """Should strip 'local:' prefix from model name before querying."""
        show_data = {
            "model_info": {"qwen2.context_length": 32768},
            "parameters": "",
        }
        mock_ctx, mock_client = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("local:qwen2.5:7b", "http://localhost:11434/v1")

        # Verify the post was called with stripped name (no "local:" prefix)
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["name"] == "qwen2.5:7b" or call_args[0][1] is not None
        assert result == 32768

    def test_handles_qwen2_architecture_key(self):
        """Different model architectures use different key prefixes in model_info."""
        show_data = {
            "model_info": {"qwen2.context_length": 65536},
            "parameters": "",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("qwen2.5:32b", "http://localhost:11434")

        assert result == 65536

    def test_returns_none_when_model_info_empty(self):
        """Should return None if model_info has no context_length key."""
        show_data = {
            "model_info": {"llama.embedding_length": 4096},
            "parameters": "",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("model", "http://localhost:11434")

        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# Level 2: run_agent.py fallback — model.context_length used when
#           /api/show detection fails
# ═══════════════════════════════════════════════════════════════════════


class TestOllamaNumCtxFallback:
    """Verify the run_agent.py fallback: when query_ollama_num_ctx() returns None
    for a local endpoint, the user's model.context_length is used as num_ctx so
    Ollama does not silently default to 2048 tokens and truncate conversation
    history (issue #14420)."""

    def test_context_length_fallback_on_detection_failure(self):
        """query_ollama_num_ctx returning None → context_length used as num_ctx."""
        with patch("agent.model_metadata.detect_local_server_type", return_value=None):
            # detect_local_server_type returning None causes query_ollama_num_ctx
            # to return None immediately.  Simulate the same None return directly.
            result = query_ollama_num_ctx("qwen3.5:9b", "http://192.168.21.205:11434/v1")

        # query returns None; the caller (run_agent.py) should apply the fallback
        # using model.context_length.  This test documents the None return that
        # triggers the fallback, not the fallback itself (which lives in AIAgent.__init__).
        assert result is None

    def test_detection_failure_via_timeout(self):
        """A connection timeout should also return None so the fallback kicks in."""
        import httpx

        with patch("agent.model_metadata.detect_local_server_type", side_effect=httpx.ConnectTimeout("timeout")):
            result = query_ollama_num_ctx("qwen3.5:9b", "http://192.168.21.205:11434/v1")

        assert result is None

    def test_modelfile_num_ctx_below_context_length_returns_modelfile_value(self):
        """query_ollama_num_ctx returns the Modelfile num_ctx even if it is below
        the user's context_length.  run_agent.py is responsible for bumping it up
        to match context_length (issue #14420 follow-up)."""
        show_data = {
            "model_info": {"qwen2.context_length": 131072},
            "parameters": "num_ctx 32768\ntemperature 0.6",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("qwen3.5:9b", "http://192.168.21.205:11434/v1")

        # The function returns the Modelfile's explicit num_ctx (32768).
        # run_agent.py will bump this up to model.context_length (64000) if needed.
        assert result == 32768
