"""Tests for local provider stream read timeout auto-detection.

When a local LLM provider is detected (Ollama, llama.cpp, vLLM, etc.),
the httpx stream read timeout should be automatically increased from the
default 60s to HERMES_API_TIMEOUT (1800s) to avoid premature connection
kills during long prefill phases.
"""

import os
import pytest
from unittest.mock import patch

from agent.model_metadata import is_local_endpoint


class TestLocalStreamReadTimeout:
    """Verify stream read timeout auto-detection logic."""

    @pytest.mark.parametrize("base_url", [
        "http://localhost:11434",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:5000",
        "http://192.168.1.100:8000",
        "http://10.0.0.5:1234",
    ])
    def test_local_endpoint_bumps_read_timeout(self, base_url):
        """Local endpoint + default timeout -> bumps to base_timeout."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 1800.0

    def test_user_override_respected_for_local(self):
        """User sets HERMES_STREAM_READ_TIMEOUT -> keep their value even for local."""
        with patch.dict(os.environ, {"HERMES_STREAM_READ_TIMEOUT": "300"}, clear=False):
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            base_url = "http://localhost:11434"
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 300.0

    @pytest.mark.parametrize("base_url", [
        "https://api.openai.com",
        "https://openrouter.ai/api",
        "https://api.anthropic.com",
    ])
    def test_remote_endpoint_keeps_default(self, base_url):
        """Remote endpoint -> keep 120s default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 120.0

    def test_empty_base_url_keeps_default(self):
        """No base_url set -> keep 120s default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            base_url = ""
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 120.0
