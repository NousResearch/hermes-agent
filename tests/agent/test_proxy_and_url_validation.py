"""Tests for malformed proxy env var and base URL validation.

Salvaged from PR #6403 by MestreY0d4-Uninter — validates that the agent
surfaces clear errors instead of cryptic httpx ``Invalid port`` exceptions
when proxy env vars or custom endpoint URLs are malformed.
"""
from __future__ import annotations

import os

import pytest

from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls


# -- proxy env validation ------------------------------------------------








@pytest.mark.parametrize("key", [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
])
def test_proxy_env_rejects_malformed_port(monkeypatch, key):
    monkeypatch.setenv(key, "http://127.0.0.1:6153export")
    with pytest.raises(RuntimeError, match=rf"Malformed proxy environment variable {key}=.*6153export"):
        _validate_proxy_env_urls()


# -- base URL validation -------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://api.example.com/v1",
    "http://127.0.0.1:6153/v1",
    "acp://copilot",
    "",
    None,
])
def test_base_url_accepts_valid(url):
    _validate_base_url(url)  # should not raise


