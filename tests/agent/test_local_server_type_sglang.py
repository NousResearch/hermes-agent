"""detect_local_server_type must not report SGLang as Ollama.

SGLang serves ``/api/tags`` for Ollama client compatibility and self-labels every entry
(``details.format == "sglang"``, digest ``sha256:sglang…``). The ollama leg matches on the shared
``"models"`` key, so without a dedicated leg every SGLang endpoint is classified as Ollama and its
context/vision probing goes through ``/api/show`` — a route SGLang answers with HTTP 400, so both
probes silently return None instead of falling through to the generic OpenAI-compatible probes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

SGLANG_TAGS = {
    "models": [
        {
            "name": "/data/models/qwen3.8-27b/twolven-AWQ-MTP",
            "model": "/data/models/qwen3.8-27b/twolven-AWQ-MTP",
            "digest": "sha256:sglang0000000000000000000000000000000000000000000000000000000000",
            "details": {"format": "sglang", "family": "twolven-AWQ-MTP", "parameter_size": "unknown"},
        }
    ]
}

OLLAMA_TAGS = {
    "models": [
        {
            "name": "llama3:latest",
            "model": "llama3:latest",
            "digest": "sha256:6a0746a1ec1a6a9de5e0cb7f0f0c0a0d0c0b0a0e0f0a0b0c0d0e0f0a0b0c0d0e",
            "details": {"format": "gguf", "family": "llama", "parameter_size": "8B"},
        }
    ]
}


@pytest.fixture(autouse=True)
def _isolate_probe_state(monkeypatch):
    """No in-memory verdicts and no disk verdicts leaking in or out of these tests."""
    from agent import model_metadata

    model_metadata._endpoint_probe_path_cache.clear()
    monkeypatch.setattr(model_metadata, "_local_probe_disk_get", lambda *a, **k: None)
    monkeypatch.setattr(model_metadata, "_local_probe_disk_put", lambda *a, **k: None)
    yield
    model_metadata._endpoint_probe_path_cache.clear()


def _client(tags_payload):
    """httpx.Client stand-in: /api/tags answers 200 with *tags_payload*, everything else 404."""
    tags = MagicMock()
    tags.status_code = 200
    tags.json.return_value = tags_payload
    miss = MagicMock()
    miss.status_code = 404

    client = MagicMock()
    client.__enter__ = lambda s: client
    client.__exit__ = MagicMock(return_value=False)
    client.get.side_effect = lambda url, *a, **k: tags if url.endswith("/api/tags") else miss
    return client


def test_sglang_ollama_compat_tags_is_not_reported_as_ollama():
    """INVARIANT: a server that self-labels its /api/tags entries as sglang is 'sglang'."""
    from agent.model_metadata import detect_local_server_type

    with patch("httpx.Client", return_value=_client(SGLANG_TAGS)):
        assert detect_local_server_type("http://127.0.0.1:18300/v1") == "sglang"


def test_real_ollama_tags_still_reported_as_ollama():
    """INVARIANT (no regression): a genuine Ollama /api/tags body stays 'ollama'."""
    from agent.model_metadata import detect_local_server_type

    with patch("httpx.Client", return_value=_client(OLLAMA_TAGS)):
        assert detect_local_server_type("http://127.0.0.1:11434/v1") == "ollama"


def test_ollama_verdict_does_not_reach_sglang():
    """The downstream consequences keyed off the verdict must not fire for SGLang."""
    from agent.model_metadata import _is_ollama_server

    with patch("httpx.Client", return_value=_client(SGLANG_TAGS)):
        assert _is_ollama_server("http://127.0.0.1:18300/v1", "") is False
