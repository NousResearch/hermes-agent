"""Tests for the canonical endpoint mandate resolver."""

from __future__ import annotations

from providers.routing import endpoint_api_mode


class TestCodexResponsesDetection:
    def test_openai_api_returns_codex_responses(self):
        assert endpoint_api_mode("https://api.openai.com/v1") == "codex_responses"


class TestDirectAnthropicHost:
    def test_lookalike_subdomain_does_not_match(self):
        assert endpoint_api_mode("https://api.anthropic.com.attacker.test/v1") is None


class TestDefaultCase:
    def test_localhost_returns_none(self):
        assert endpoint_api_mode("http://localhost:11434/v1") is None
