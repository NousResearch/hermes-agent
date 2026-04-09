from __future__ import annotations

import pytest

from run_agent import AIAgent


@pytest.mark.parametrize(
    ("base_url", "should_raise"),
    [
        ("https://api.example.com/v1", False),
        ("http://127.0.0.1:6153/v1", False),
        ("acp://copilot", False),
        ("", False),
        ("http://127.0.0.1:6153export", True),
    ],
)
def test_validate_openai_base_url(base_url: str, should_raise: bool):
    if should_raise:
        with pytest.raises(RuntimeError, match="Malformed custom endpoint URL"):
            AIAgent._validate_openai_base_url(base_url)
    else:
        AIAgent._validate_openai_base_url(base_url)


def test_validate_proxy_environment_urls_accepts_normal_proxy_values(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:6153")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example.com:8443")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")

    AIAgent._validate_proxy_environment_urls()


@pytest.mark.parametrize("key", ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"])
def test_validate_proxy_environment_urls_rejects_malformed_port_suffix(monkeypatch, key: str):
    monkeypatch.setenv(key, "http://127.0.0.1:6153export")

    with pytest.raises(RuntimeError, match=fr"Malformed proxy environment variable {key}=.*6153export"):
        AIAgent._validate_proxy_environment_urls()
