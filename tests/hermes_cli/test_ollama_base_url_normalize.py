"""Regression tests for #7516: Ollama base_url /v1 normalization.

The OpenAI SDK appends /chat/completions directly to base_url. Ollama's
OpenAI surface lives at /v1/chat/completions, so a bare
http://127.0.0.1:11434 produces a 404. normalize_ollama_base_url appends
/v1 to Ollama-looking URLs with an empty path.
"""

import pytest


class TestLooksLikeOllamaBaseUrl:
    def test_loopback_default_port(self):
        from hermes_cli.auth import looks_like_ollama_base_url
        assert looks_like_ollama_base_url("http://127.0.0.1:11434") is True
        assert looks_like_ollama_base_url("http://localhost:11434") is True

    def test_lan_host_default_port(self):
        from hermes_cli.auth import looks_like_ollama_base_url
        assert looks_like_ollama_base_url("http://192.168.1.10:11434") is True

    def test_ollama_in_hostname_any_port(self):
        from hermes_cli.auth import looks_like_ollama_base_url
        assert looks_like_ollama_base_url("https://ollama.acme.com") is True
        assert looks_like_ollama_base_url("http://my-ollama.box:8080") is True

    def test_non_ollama_url(self):
        from hermes_cli.auth import looks_like_ollama_base_url
        assert looks_like_ollama_base_url("https://api.openai.com") is False
        assert looks_like_ollama_base_url("http://127.0.0.1:8080") is False
        assert looks_like_ollama_base_url("") is False


class TestNormalizeOllamaBaseUrl:
    def test_appends_v1_to_bare_localhost(self):
        from hermes_cli.auth import normalize_ollama_base_url
        assert normalize_ollama_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434/v1"
        assert normalize_ollama_base_url("http://localhost:11434") == "http://localhost:11434/v1"

    def test_preserves_existing_v1(self):
        from hermes_cli.auth import normalize_ollama_base_url
        assert normalize_ollama_base_url("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434/v1"
        assert normalize_ollama_base_url("http://127.0.0.1:11434/v1/") == "http://127.0.0.1:11434/v1"

    def test_ignores_non_ollama(self):
        from hermes_cli.auth import normalize_ollama_base_url
        assert normalize_ollama_base_url("https://api.openai.com") == "https://api.openai.com"
        assert normalize_ollama_base_url("http://127.0.0.1:8422") == "http://127.0.0.1:8422"

    def test_preserves_custom_path(self):
        from hermes_cli.auth import normalize_ollama_base_url
        # An Ollama URL with an explicit non-root path stays untouched
        assert normalize_ollama_base_url("http://ollama.acme.com/proxy") == "http://ollama.acme.com/proxy"

    def test_empty_url(self):
        from hermes_cli.auth import normalize_ollama_base_url
        assert normalize_ollama_base_url("") == ""
        assert normalize_ollama_base_url(None) == ""
