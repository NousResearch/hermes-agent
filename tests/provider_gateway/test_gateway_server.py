import json
import socket
import urllib.request
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from provider_gateway.server import start_gateway_server, stop_gateway_server


def get_free_port() -> int:
    """Find a free port dynamically to prevent port conflict errors during test execution."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def running_server():
    port = get_free_port()
    server = start_gateway_server(port=port, background=True)
    yield port, server
    stop_gateway_server(server)


def test_server_models_endpoint(running_server) -> None:
    port, _ = running_server
    url = f"http://127.0.0.1:{port}/v1/models"
    
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=2.0) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) >= 1
        assert "id" in data["data"][0]


@patch("openai.resources.chat.completions.Completions.create")
def test_server_chat_completions_non_streaming(mock_create, running_server) -> None:
    port, _ = running_server
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    # Setup mock response
    mock_choice = MagicMock()
    mock_choice.message = MagicMock(content="Hello from mock server!")
    mock_choice.message.role = "assistant"
    mock_choice.finish_reason = "stop"

    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "id": "chat-mock-123",
        "object": "chat.completion",
        "created": 1677610602,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from mock server!"
            },
            "finish_reason": "stop"
        }]
    }
    mock_create.return_value = mock_resp

    payload = json.dumps({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=2.0) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["id"] == "chat-mock-123"
        assert data["choices"][0]["message"]["content"] == "Hello from mock server!"


@patch("openai.resources.chat.completions.Completions.create")
def test_server_chat_completions_streaming(mock_create, running_server) -> None:
    port, _ = running_server
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    # Setup mock streaming chunks
    chunk1 = MagicMock()
    chunk1.model_dump.return_value = {
        "choices": [{
            "delta": {"content": "Hello"},
            "finish_reason": None
        }]
    }

    chunk2 = MagicMock()
    chunk2.model_dump.return_value = {
        "choices": [{
            "delta": {"content": " world!"},
            "finish_reason": "stop"
        }]
    }

    mock_create.return_value = [chunk1, chunk2]

    payload = json.dumps({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=2.0) as response:
        assert response.status == 200
        lines = response.read().decode("utf-8").splitlines()
        
        events = [line for line in lines if line.strip()]
        assert len(events) >= 3  # chunk1, chunk2, [DONE]
        
        assert events[0].startswith("data: ")
        assert "Hello" in events[0]
        assert "world!" in events[1]
        assert events[-1] == "data: [DONE]"
