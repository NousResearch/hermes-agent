import json
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from provider_gateway.discovery import OllamaDiscovery


def test_ollama_discovery_unreachable() -> None:
    """Test behavior when Ollama daemon is unreachable."""
    discovery = OllamaDiscovery(host="http://127.0.0.1:9999")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        models = discovery.discover_local_models()
        assert models == []


def test_ollama_discovery_success() -> None:
    """Test successful discovery of Ollama models with context length resolution."""
    discovery = OllamaDiscovery(host="http://127.0.0.1:11434")

    # Mock response for /api/tags
    mock_tags_response = MagicMock()
    mock_tags_response.__enter__.return_value = mock_tags_response
    mock_tags_response.status = 200
    mock_tags_response.read.return_value = json.dumps({
        "models": [
            {"name": "llama3:8b"},
            {"name": "mistral:latest"},
        ]
    }).encode("utf-8")

    # Mock responses for /api/show
    mock_show_llama3 = MagicMock()
    mock_show_llama3.__enter__.return_value = mock_show_llama3
    mock_show_llama3.status = 200
    mock_show_llama3.read.return_value = json.dumps({
        "parameters": "num_ctx        8192\nstop           <|im_end|>",
        "model_info": {
            "llama.context_length": 8192
        }
    }).encode("utf-8")

    mock_show_mistral = MagicMock()
    mock_show_mistral.__enter__.return_value = mock_show_mistral
    mock_show_mistral.status = 200
    # Test fallback model_info context length parsing
    mock_show_mistral.read.return_value = json.dumps({
        "parameters": "",
        "model_info": {
            "mistral.context_length": 32768
        }
    }).encode("utf-8")

    # urllib.request.urlopen side effect
    def urlopen_side_effect(req, timeout=1.0):
        if isinstance(req, str):
            url = req
            req_data = None
        else:
            url = req.full_url
            req_data = req.data

        if "/api/tags" in url:
            return mock_tags_response
        elif "/api/show" in url:
            if req_data:
                payload = json.loads(req_data.decode("utf-8"))
                if payload.get("name") == "llama3:8b":
                    return mock_show_llama3
                elif payload.get("name") == "mistral:latest":
                    return mock_show_mistral
            return mock_show_llama3
        raise ValueError(f"Unexpected request URL: {url}")

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
        models = discovery.discover_local_models()
        assert len(models) == 2
        
        # Check llama3
        assert models[0]["model"] == "llama3:8b"
        assert models[0]["provider"] == "ollama"
        assert models[0]["num_ctx"] == 8192
        assert models[0]["base_url"] == "http://127.0.0.1:11434/v1"

        # Check mistral
        assert models[1]["model"] == "mistral:latest"
        assert models[1]["provider"] == "ollama"
        assert models[1]["num_ctx"] == 32768
        assert models[1]["base_url"] == "http://127.0.0.1:11434/v1"


def test_ollama_discovery_fallback_ctx() -> None:
    """Test fallback context length logic."""
    discovery = OllamaDiscovery(host="http://127.0.0.1:11434")

    mock_tags_response = MagicMock()
    mock_tags_response.__enter__.return_value = mock_tags_response
    mock_tags_response.status = 200
    mock_tags_response.read.return_value = json.dumps({
        "models": [{"name": "unknown-model:latest"}]
    }).encode("utf-8")

    # Failing API show or no ctx information
    mock_show_response = MagicMock()
    mock_show_response.__enter__.return_value = mock_show_response
    mock_show_response.status = 200
    mock_show_response.read.return_value = json.dumps({
        "parameters": "",
        "model_info": {}
    }).encode("utf-8")

    def urlopen_side_effect(req, timeout=1.0):
        if isinstance(req, str):
            url = req
        else:
            url = req.full_url

        if "/api/tags" in url:
            return mock_tags_response
        elif "/api/show" in url:
            return mock_show_response
        raise ValueError(f"Unexpected request URL: {url}")

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
        models = discovery.discover_local_models()
        assert len(models) == 1
        assert models[0]["model"] == "unknown-model:latest"
        assert models[0]["num_ctx"] == 8192  # should fallback to 8192
