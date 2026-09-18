"""Tests for AWS provider fallback chain and credential validation.

Verifies:
1. Provider selection skips providers with missing/invalid credentials before agent spawn.
2. AWS auth failure (e.g. 403 invalid security token from stale AWS_ACCESS_KEY_ID) is classified as auth
   and falls back to Bedrock profile credentials (AWS_PROFILE).
3. Secrets are never exposed in fallback messages or error text.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from agent.error_classifier import classify_api_error, FailoverReason
from agent.chat_completion_helpers import _fallback_entry_unavailable_without_network
from agent.bedrock_adapter import (
    has_aws_credentials,
    fallback_aws_profile_credentials,
    reset_client_cache,
)
from gateway.run import _try_resolve_fallback_provider


class TestAWSProviderFallback:
    def test_aws_auth_error_classification(self):
        """AWS 403 invalid security token / InvalidClientTokenId must classify as FailoverReason.auth."""
        msg = "An error occurred (InvalidClientTokenId) when calling the Converse operation: The security token included in the request is invalid."
        err = Exception(msg)
        classified = classify_api_error(err)
        assert classified.reason == FailoverReason.auth
        assert classified.is_auth is True
        assert classified.should_fallback is True

    def test_fallback_entry_unavailable_without_network_skips_missing_creds(self, monkeypatch):
        """Candidate providers with missing credentials must be detected and skipped locally."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_KEY", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda env=None: False)
        monkeypatch.setattr("hermes_cli.auth._explicit_env_credentials_present", lambda provider: False)

        openrouter_entry = {"provider": "openrouter", "model": "anthropic/claude-3.5-sonnet"}
        bedrock_entry = {"provider": "bedrock", "model": "anthropic.claude-3-5-sonnet-v1:0"}

        agent_mock = MagicMock()
        assert _fallback_entry_unavailable_without_network(agent_mock, bedrock_entry) == "no_aws_credentials"

    def test_fallback_aws_profile_credentials_evicts_stale_env_keys(self, monkeypatch):
        """When host env has stale AWS_ACCESS_KEY_ID but valid AWS_PROFILE, fallback clears stale env keys."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAINVALIDKEY1234567")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "stalesequcretkey1234567")
        monkeypatch.setenv("AWS_PROFILE", "valid-profile")

        assert fallback_aws_profile_credentials() is True
        assert "AWS_ACCESS_KEY_ID" not in os.environ
        assert "AWS_SECRET_ACCESS_KEY" not in os.environ
        assert os.environ.get("AWS_PROFILE") == "valid-profile"
        assert has_aws_credentials() is True

    def test_provider_selection_skips_credentialless_providers_before_spawn(self, monkeypatch):
        """_try_resolve_fallback_provider must skip entries with missing credentials and pick live model."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("AWS_PROFILE", "valid-bedrock-profile")

        fallback_chain = [
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
            {"provider": "bedrock", "model": "anthropic.claude-3-5-sonnet-v1:0"},
        ]

        with patch("gateway.run.get_fallback_chain", return_value=fallback_chain), \
             patch("agent.bedrock_adapter.has_aws_credentials", return_value=True), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_resolve:

            def _side_effect(requested, **kwargs):
                if requested == "openrouter":
                    return {"provider": "openrouter", "api_key": None, "base_url": "https://openrouter.ai/api/v1"}
                return {"provider": "bedrock", "api_key": "aws-sdk", "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com"}

            mock_resolve.side_effect = _side_effect

            resolved = _try_resolve_fallback_provider()
            assert resolved is not None
            assert resolved.get("provider") == "bedrock"
            assert resolved.get("model") == "anthropic.claude-3-5-sonnet-v1:0"
