"""Tests for Amazon Bedrock Mantle provider wiring."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.bedrock_mantle import (
    discover_mantle_models,
    is_mantle_claude_model,
    mantle_anthropic_base_url,
    mantle_openai_base_url,
    reset_mantle_caches_for_tests,
    resolve_mantle_bearer_token,
    resolve_mantle_region,
)
from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
from hermes_cli.models import CANONICAL_PROVIDERS, normalize_provider
from hermes_cli import runtime_provider as rp
from providers import get_provider_profile


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reset_mantle_caches_for_tests()
    for key in (
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "BEDROCK_MANTLE_REGION",
        "BEDROCK_MANTLE_BASE_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)
    yield
    reset_mantle_caches_for_tests()


class TestMantleHelpers:
    def test_endpoints(self):
        assert mantle_openai_base_url("us-west-2") == (
            "https://bedrock-mantle.us-west-2.api.aws/v1"
        )
        assert mantle_anthropic_base_url("eu-west-1") == (
            "https://bedrock-mantle.eu-west-1.api.aws/anthropic"
        )

    def test_region_priority(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        assert resolve_mantle_region() == "us-west-2"
        assert resolve_mantle_region(config_region="ap-south-1") == "ap-south-1"

    def test_claude_model_detection(self):
        assert is_mantle_claude_model("anthropic.claude-sonnet-5") is True
        assert is_mantle_claude_model("amazon-bedrock-mantle/anthropic.claude-opus-4-7")
        assert is_mantle_claude_model("gpt-oss-120b") is False
        assert is_mantle_claude_model("qwen3-coder-480b-a35b") is False

    def test_explicit_bearer(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "mantle-token-abc")
        assert resolve_mantle_bearer_token("us-east-1") == "mantle-token-abc"

    def test_iam_mint_fallback(self, monkeypatch):
        def fake_provide_token(region=None):
            return f"minted-for-{region}"

        token = resolve_mantle_bearer_token(
            "us-east-1",
            allow_iam_mint=True,
            provider_factory=fake_provide_token,
        )
        assert token == "minted-for-us-east-1"

    def test_discovery_parses_openai_list(self):
        payload = {
            "data": [
                {"id": "gpt-oss-120b"},
                {"id": "qwen3-235b-a22b"},
            ]
        }

        def fake_fetch(url, headers=None, timeout=None):
            assert "/v1/models" in url
            assert headers["Authorization"].startswith("Bearer ")
            return payload

        ids = discover_mantle_models(
            "us-east-1", "tok", fetch_fn=fake_fetch
        )
        assert "gpt-oss-120b" in ids
        assert "anthropic.claude-sonnet-5" in ids  # appended Claude rows


class TestMantleProviderRegistration:
    def test_profile(self):
        p = get_provider_profile("amazon-bedrock-mantle")
        assert p is not None
        assert p.name == "amazon-bedrock-mantle"
        assert "bedrock-mantle" in (p.aliases or ())

    def test_aliases(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "t")
        assert resolve_provider("bedrock-mantle") == "amazon-bedrock-mantle"
        assert resolve_provider("mantle") == "amazon-bedrock-mantle"
        assert normalize_provider("bedrock-mantle") == "amazon-bedrock-mantle"

    def test_canonical_picker(self):
        slugs = {e.slug for e in CANONICAL_PROVIDERS}
        assert "amazon-bedrock-mantle" in slugs

    def test_auth_registry(self):
        assert "amazon-bedrock-mantle" in PROVIDER_REGISTRY


class TestMantleRuntimeResolution:
    def test_openai_path(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bearer-xyz")
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        monkeypatch.setattr(
            rp,
            "_get_model_config",
            lambda: {
                "provider": "amazon-bedrock-mantle",
                "default": "gpt-oss-120b",
            },
        )
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="mantle")
        assert resolved["provider"] == "amazon-bedrock-mantle"
        assert resolved["api_mode"] == "chat_completions"
        assert resolved["base_url"] == "https://bedrock-mantle.us-west-2.api.aws/v1"
        assert resolved["api_key"] == "bearer-xyz"
        assert resolved["region"] == "us-west-2"

    def test_claude_anthropic_path(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bearer-xyz")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setattr(
            rp,
            "_get_model_config",
            lambda: {
                "provider": "amazon-bedrock-mantle",
                "default": "anthropic.claude-sonnet-5",
            },
        )
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(
            requested="amazon-bedrock-mantle",
            target_model="anthropic.claude-sonnet-5",
        )
        assert resolved["provider"] == "amazon-bedrock-mantle"
        assert resolved["api_mode"] == "anthropic_messages"
        assert resolved["base_url"].endswith("/anthropic")
        assert "bedrock-mantle.us-east-1.api.aws" in resolved["base_url"]
        assert resolved["api_key"] == "bearer-xyz"

    def test_missing_creds_raises(self, monkeypatch):
        monkeypatch.setattr(
            rp,
            "_get_model_config",
            lambda: {"provider": "amazon-bedrock-mantle", "default": "gpt-oss-120b"},
        )
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)
        with patch(
            "agent.bedrock_mantle.has_mantle_credentials", return_value=False
        ), patch(
            "agent.bedrock_mantle.resolve_mantle_bearer_token", return_value=None
        ):
            with pytest.raises(Exception) as ei:
                rp.resolve_runtime_provider(requested="amazon-bedrock-mantle")
            msg = str(ei.value).lower()
            assert "mantle" in msg or "bearer" in msg


class TestMantleAnthropicBearer:
    def test_requires_bearer_for_mantle_anthropic_url(self):
        from agent.anthropic_adapter import _requires_bearer_auth

        assert _requires_bearer_auth(
            "https://bedrock-mantle.us-east-1.api.aws/anthropic"
        )
        assert not _requires_bearer_auth(
            "https://bedrock-runtime.us-east-1.amazonaws.com"
        )
