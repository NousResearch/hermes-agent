from __future__ import annotations

import pytest

from run_agent import AIAgent


def test_validate_openai_base_url_accepts_normal_http_urls():
    AIAgent._validate_openai_base_url("https://api.example.com/v1")
    AIAgent._validate_openai_base_url("http://127.0.0.1:6153/v1")
    AIAgent._validate_openai_base_url("acp://copilot")
    AIAgent._validate_openai_base_url("")


def test_validate_openai_base_url_rejects_malformed_port_suffix():
    with pytest.raises(RuntimeError, match="Malformed custom endpoint URL"):
        AIAgent._validate_openai_base_url("http://127.0.0.1:6153export")
