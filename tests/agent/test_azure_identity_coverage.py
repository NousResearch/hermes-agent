"""Test coverage for agent/azure_identity_adapter.py — 17 functions had LOW coverage.

Tests the pure helper functions: config parsing, token provider detection,
bearer materialization, and credential cache reset. All Azure SDK calls
are mocked — no real Azure identity is touched.
"""

import pytest
from unittest.mock import MagicMock, patch

from agent.azure_identity_adapter import (
    EntraIdentityConfig,
    is_token_provider,
    materialize_bearer_for_http,
    reset_credential_cache,
)


class TestEntraIdentityConfig:
    def test_defaults(self):
        cfg = EntraIdentityConfig()
        assert cfg is not None


class TestIsTokenProvider:
    def test_callable_returns_true(self):
        assert is_token_provider(lambda: "token") is True

    def test_string_returns_false(self):
        """Strings are API keys, not callables — must return False."""
        assert is_token_provider("static-token") is False

    def test_none_returns_false(self):
        assert is_token_provider(None) is False

    def test_int_returns_false(self):
        assert is_token_provider(42) is False

    def test_dict_returns_false(self):
        assert is_token_provider({"key": "value"}) is False

    def test_class_method_returns_true(self):
        class Provider:
            def get_token(self):
                return "tok"
        assert is_token_provider(Provider().get_token) is True


class TestMaterializeBearer:
    def test_string_passthrough(self):
        """A plain string API key is returned as-is."""
        assert materialize_bearer_for_http("static-key-123") == "static-key-123"

    def test_callable_resolves_to_token(self):
        """A callable provider is invoked and its return value is used."""
        provider = lambda: "dynamic-jwt-token"
        assert materialize_bearer_for_http(provider) == "dynamic-jwt-token"

    def test_callable_returning_none_raises(self):
        """A provider returning None/empty should raise, not silently pass."""
        with pytest.raises(ValueError, match="empty value"):
            materialize_bearer_for_http(lambda: None)


class TestResetCache:
    def test_does_not_raise(self):
        reset_credential_cache()
