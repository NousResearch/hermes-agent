"""Tests for the Graphnosis memory provider."""

import json
from pathlib import Path

import pytest


class FakeGraphnosisClient:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def connect(self):
        return None

    def close(self):
        return None

    def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        if name == "recall":
            return "Remembered: user prefers TypeScript."
        if name == "remember":
            return "Saved."
        if name == "stats":
            return "engrams: 3"
        return ""


@pytest.fixture
def provider_module():
    from plugins.memory.graphnosis import GraphnosisMemoryProvider

    return GraphnosisMemoryProvider


def test_is_available_when_socket_exists(provider_module, tmp_path):
    sock = tmp_path / "mcp.sock"
    sock.touch()
    provider = provider_module({"socket_path": str(sock), "prefetch_max_tokens": 500})
    assert provider.is_available() is True


def test_is_available_when_socket_missing(provider_module):
    provider = provider_module({"socket_path": "/nonexistent/mcp.sock"})
    assert provider.is_available() is False


def test_handle_recall_tool(provider_module):
    provider = provider_module({"socket_path": "/tmp/mcp.sock", "prefetch_max_tokens": 500})
    fake = FakeGraphnosisClient()
    provider._client = fake
    result = json.loads(provider.handle_tool_call("graphnosis_recall", {"query": "typescript preference"}))
    assert "TypeScript" in result["result"]
    assert fake.calls[0][0] == "recall"


def test_handle_remember_tool(provider_module):
    provider = provider_module({"socket_path": "/tmp/mcp.sock", "prefetch_max_tokens": 500})
    fake = FakeGraphnosisClient()
    provider._client = fake
    result = json.loads(provider.handle_tool_call("graphnosis_remember", {"text": "User prefers zsh"}))
    assert "Saved" in result["result"]
    assert fake.calls[0][0] == "remember"


def test_prefetch_queue(provider_module):
    provider = provider_module({"socket_path": "/tmp/mcp.sock", "prefetch_max_tokens": 500})
    fake = FakeGraphnosisClient()
    provider._client = fake
    provider.queue_prefetch("what are my preferences?")
    if provider._prefetch_thread:
        provider._prefetch_thread.join(timeout=2.0)
    block = provider.prefetch("what are my preferences?")
    assert "Graphnosis Memory" in block or block == ""
