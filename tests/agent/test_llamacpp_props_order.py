"""Tests for llama.cpp /props endpoint probe order (bug #13091).

Verifies that detect_local_server_type() and fetch_endpoint_model_metadata()
try /props BEFORE /v1/props when detecting/fetching from llama.cpp servers,
since llama.cpp exposes /props at server root (not under /v1/).
"""

import pytest
from unittest.mock import patch, MagicMock


def _make_fail_response():
    """Return a mock response that indicates the endpoint is not this server type."""
    resp = MagicMock()
    resp.status_code = 404
    return resp


class TestLlamaCppPropsProbeOrder:
    """Test that /props is tried before /v1/props for llama.cpp detection."""

    def test_props_tried_before_v1_props(self):
        """When /props returns 200, /v1/props should NOT be called."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client

            # Sequence of responses for each probe:
            # 1. LM Studio /api/v1/models -> 404
            # 2. Ollama /api/tags -> 404
            # 3. llama.cpp /props -> 200 OK
            # 4. llama.cpp /v1/props -> should NOT be called
            mock_client.get.side_effect = [
                _make_fail_response(),  # LM Studio
                _make_fail_response(),  # Ollama
                MagicMock(status_code=200, text='{"default_generation_settings": {"n_ctx": 4096}}'),  # /props
            ]

            from agent.model_metadata import detect_local_server_type
            result = detect_local_server_type("http://localhost:8080")

            assert result == "llamacpp"
            # Verify /props was tried first (after LM Studio and Ollama probes)
            props_call = mock_client.get.call_args_list[2]
            assert "/props" in props_call[0][0]
            assert "/v1/props" not in props_call[0][0]
            # Verify only 3 calls (no /v1/props fallback needed)
            assert mock_client.get.call_count == 3

    def test_fallback_to_v1_props_when_props_404(self):
        """When /props returns 404, /v1/props should be tried as fallback."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client

            # Sequence of responses:
            # 1. LM Studio /api/v1/models -> 404
            # 2. Ollama /api/tags -> 404
            # 3. llama.cpp /props -> 404
            # 4. llama.cpp /v1/props -> 200 OK
            mock_client.get.side_effect = [
                _make_fail_response(),  # LM Studio
                _make_fail_response(),  # Ollama
                MagicMock(status_code=404),  # /props
                MagicMock(status_code=200, text='{"default_generation_settings": {"n_ctx": 4096}}'),  # /v1/props
            ]

            from agent.model_metadata import detect_local_server_type
            result = detect_local_server_type("http://localhost:8080")

            assert result == "llamacpp"
            # Verify /props was tried first
            props_call = mock_client.get.call_args_list[2]
            assert "/props" in props_call[0][0]
            # Verify /v1/props was called as fallback
            v1_props_call = mock_client.get.call_args_list[3]
            assert "/v1/props" in v1_props_call[0][0]
            assert mock_client.get.call_count == 4

    def test_no_detection_when_both_fail(self):
        """When both /props and /v1/props fail, should not return llamacpp."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client

            mock_client.get.side_effect = [
                _make_fail_response(),  # LM Studio
                _make_fail_response(),  # Ollama
                MagicMock(status_code=404),  # /props
                MagicMock(status_code=404),  # /v1/props
                _make_fail_response(),  # vLLM
            ]

            from agent.model_metadata import detect_local_server_type
            result = detect_local_server_type("http://localhost:8080")

            assert result is None

    def test_props_must_have_correct_content(self):
        """Response must contain 'default_generation_settings' to be recognized."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client

            # /props returns 200 but wrong content
            wrong_response = MagicMock()
            wrong_response.status_code = 200
            wrong_response.text = '{"error": "something else"}'

            mock_client.get.side_effect = [
                _make_fail_response(),  # LM Studio
                _make_fail_response(),  # Ollama
                wrong_response,  # /props
                wrong_response,  # /v1/props
            ]

            from agent.model_metadata import detect_local_server_type
            result = detect_local_server_type("http://localhost:8080")

            # Should not detect as llamacpp since content is wrong
            assert result is None

    def test_v1_normalized_to_root_for_props_probe(self):
        """When base_url ends with /v1, the /v1 suffix should be stripped before probing."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client

            mock_client.get.side_effect = [
                _make_fail_response(),  # LM Studio (at /v1/api/v1/models... wait)
                _make_fail_response(),  # Ollama
                MagicMock(status_code=200, text='{"default_generation_settings": {"n_ctx": 4096}}'),  # /props at root
            ]

            from agent.model_metadata import detect_local_server_type
            # URL has /v1 suffix — should be stripped to / before probing
            result = detect_local_server_type("http://localhost:8080/v1")

            assert result == "llamacpp"
            # Should probe /props at root, not /v1/props
            props_call = mock_client.get.call_args_list[2]
            called_url = props_call[0][0]
            assert "/props" in called_url
            # Should NOT have double /v1/v1/props
            assert "/v1/v1" not in called_url
