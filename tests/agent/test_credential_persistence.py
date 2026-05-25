"""Tests for agent/credential_persistence.py — disk-boundary sanitization helpers."""

import hashlib
import pytest
from agent.credential_persistence import (
    _normalize_key,
    is_borrowed_credential_source,
    _is_secret_payload_key,
    _fingerprint_value,
    _credential_secret_fingerprint,
    sanitize_borrowed_credential_payload,
    _PERSISTABLE_PROVIDER_SOURCES,
    _SAFE_SECRETISH_METADATA_KEYS,
    _SECRET_VALUE_KEYS,
    _SECRET_VALUE_SUFFIXES,
)


# ── _normalize_key ────────────────────────────────────────────────────

class TestNormalizeKey:
    def test_camel_case_to_snake(self):
        assert _normalize_key("accessToken") == "access_token"
        assert _normalize_key("refreshToken") == "refresh_token"
        assert _normalize_key("clientSecret") == "client_secret"

    def test_already_snake_case(self):
        assert _normalize_key("access_token") == "access_token"
        assert _normalize_key("already_snake") == "already_snake"

    def test_dots_to_underscores(self):
        assert _normalize_key("some.key.name") == "some_key_name"

    def test_hyphens_to_underscores(self):
        assert _normalize_key("some-key-name") == "some_key_name"

    def test_lowercases(self):
        assert _normalize_key("ACCESS_TOKEN") == "access_token"

    def test_empty_and_none(self):
        assert _normalize_key("") == ""
        assert _normalize_key(None) == ""  # None -> "" via `key or ""`

    def test_whitespace_trimmed(self):
        assert _normalize_key("  access_token  ") == "access_token"

    def test_mixed_camel_and_dots(self):
        assert _normalize_key("some.camelCase.key") == "some_camel_case_key"


# ── is_borrowed_credential_source ─────────────────────────────────────

class TestIsBorrowedCredentialSource:
    def test_owned_sources_are_not_borrowed(self):
        """All entries in _PERSISTABLE_PROVIDER_SOURCES are owned."""
        for provider, source in _PERSISTABLE_PROVIDER_SOURCES:
            assert is_borrowed_credential_source(source, provider) is False, \
                f"({provider}, {source}) should be owned"

    def test_manual_is_owned(self):
        assert is_borrowed_credential_source("manual") is False
        assert is_borrowed_credential_source("manual:some-label") is False

    def test_empty_source_is_owned(self):
        assert is_borrowed_credential_source("") is False
        assert is_borrowed_credential_source(None) is False

    def test_unknown_source_is_borrowed(self):
        assert is_borrowed_credential_source("env_var") is True
        assert is_borrowed_credential_source("external_vault") is True

    def test_correct_provider_required(self):
        """Same source with wrong provider is still borrowed."""
        # anthropic/hermes_pkce is owned
        assert is_borrowed_credential_source("hermes_pkce", "anthropic") is False
        # But hermes_pkce with wrong provider is borrowed
        assert is_borrowed_credential_source("hermes_pkce", "nous") is True


# ── _is_secret_payload_key ────────────────────────────────────────────

class TestIsSecretPayloadKey:
    def test_known_secret_value_keys(self):
        for key in _SECRET_VALUE_KEYS:
            assert _is_secret_payload_key(key) is True, f"{key} should be secret"

    def test_safe_metadata_keys_are_not_secret(self):
        for key in _SAFE_SECRETISH_METADATA_KEYS:
            assert _is_secret_payload_key(key) is False, f"{key} should be safe"

    def test_suffix_matching(self):
        for suffix in _SECRET_VALUE_SUFFIXES:
            test_key = f"my{suffix}"
            assert _is_secret_payload_key(test_key) is True, f"{test_key} should match suffix {suffix}"

    def test_camel_case_variants_of_suffixes(self):
        """CamelCase keys that normalize to suffix patterns should match."""
        assert _is_secret_payload_key("myApiKey") is True
        assert _is_secret_payload_key("myAccessToken") is True

    def test_regular_keys_are_not_secret(self):
        assert _is_secret_payload_key("username") is False
        assert _is_secret_payload_key("provider") is False
        assert _is_secret_payload_key("model") is False
        assert _is_secret_payload_key("source") is False

    def test_empty_and_none(self):
        assert _is_secret_payload_key("") is False
        assert _is_secret_payload_key(None) is False


# ── _fingerprint_value ────────────────────────────────────────────────

class TestFingerprintValue:
    def test_produces_sha256_prefix(self):
        fp = _fingerprint_value("my-secret-token")
        assert fp is not None
        assert fp.startswith("sha256:")
        assert len(fp) == len("sha256:") + 16  # 16 hex chars

    def test_deterministic(self):
        fp1 = _fingerprint_value("same-value")
        fp2 = _fingerprint_value("same-value")
        assert fp1 == fp2

    def test_different_values_different_fingerprints(self):
        fp1 = _fingerprint_value("value-1")
        fp2 = _fingerprint_value("value-2")
        assert fp1 != fp2

    def test_none_returns_none(self):
        assert _fingerprint_value(None) is None

    def test_empty_string_returns_none(self):
        assert _fingerprint_value("") is None

    def test_unicode_values(self):
        fp = _fingerprint_value("tokén-ünicode")
        assert fp is not None
        assert fp.startswith("sha256:")


# ── _credential_secret_fingerprint ────────────────────────────────────

class TestCredentialSecretFingerprint:
    def test_agent_key_first(self):
        fp = _credential_secret_fingerprint({"agent_key": "sk-abc123"})
        assert fp is not None
        assert fp.startswith("sha256:")

    def test_access_token_second(self):
        fp = _credential_secret_fingerprint({
            "access_token": "at-secret",
            "agent_key": None,
        })
        assert fp is not None

    def test_falls_back_to_key_scanning(self):
        """When the priority keys are missing, scan all keys for secret patterns."""
        fp = _credential_secret_fingerprint({"my_api_key": "custom-secret-key"})
        assert fp is not None
        assert fp.startswith("sha256:")

    def test_keeps_existing_fingerprint(self):
        existing = "sha256:abcdef1234567890"
        fp = _credential_secret_fingerprint({"secret_fingerprint": existing})
        assert fp == existing

    def test_no_secret_returns_none(self):
        fp = _credential_secret_fingerprint({"username": "admin", "provider": "nous"})
        assert fp is None

    def test_empty_payload(self):
        assert _credential_secret_fingerprint({}) is None

    def test_ignores_invalid_existing_fingerprint(self):
        """Non-sha256: fingerprints are ignored."""
        fp = _credential_secret_fingerprint({"secret_fingerprint": "not-a-real-fingerprint"})
        assert fp is None


# ── sanitize_borrowed_credential_payload ──────────────────────────────

class TestSanitizeBorrowedCredentialPayload:
    def test_owned_credential_passes_through_unchanged(self):
        """Owned (e.g., manual) credentials should not be modified."""
        payload = {
            "source": "manual",
            "access_token": "sk-secret-token-value",
            "provider": "openai",
            "username": "user123",
        }
        result = sanitize_borrowed_credential_payload(payload)
        assert result == payload
        assert result["access_token"] == "sk-secret-token-value"

    def test_borrowed_credential_strips_secrets(self):
        """Borrowed credentials must have secret values removed."""
        payload = {
            "source": "env_var",
            "access_token": "sk-secret-borrowed-token",
            "refresh_token": "rt-secret-refresh",
            "provider": "anthropic",
            "model": "claude-sonnet-4",
        }
        result = sanitize_borrowed_credential_payload(payload)
        assert "access_token" not in result
        assert "refresh_token" not in result
        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-sonnet-4"

    def test_borrowed_credential_preserves_metadata(self):
        """Non-secret metadata should survive sanitization."""
        payload = {
            "source": "external_vault",
            "token": "sk-borrowed",
            "scope": "read write",
            "expires_at": "2026-06-01T00:00:00Z",
            "last_status": "active",
            "token_type": "Bearer",
        }
        result = sanitize_borrowed_credential_payload(payload)
        assert "token" not in result
        assert result["scope"] == "read write"
        assert result["expires_at"] == "2026-06-01T00:00:00Z"
        assert result["last_status"] == "active"
        assert result["token_type"] == "Bearer"

    def test_borrowed_adds_fingerprint(self):
        """Sanitized borrowed credentials get a secret_fingerprint."""
        payload = {
            "source": "env_var",
            "access_token": "sk-fingerprint-me",
        }
        result = sanitize_borrowed_credential_payload(payload)
        assert "secret_fingerprint" in result
        assert result["secret_fingerprint"].startswith("sha256:")

    def test_borrowed_without_secrets_no_fingerprint(self):
        """If there are no secrets, no fingerprint is added."""
        payload = {
            "source": "env_var",
            "provider": "openai",
            "model": "gpt-5",
        }
        result = sanitize_borrowed_credential_payload(payload)
        assert "secret_fingerprint" not in result

    def test_empty_payload(self):
        result = sanitize_borrowed_credential_payload({})
        assert result == {}

    def test_owned_nous_oauth_passes_through(self):
        """Nous Portal device_code tokens are owned and should survive."""
        payload = {
            "source": "device_code",
            "access_token": "nous-token-secret",
            "refresh_token": "nous-refresh-secret",
            "expires_at": "2026-06-01T00:00:00Z",
        }
        result = sanitize_borrowed_credential_payload(payload, provider_id="nous")
        assert result["access_token"] == "nous-token-secret"
        assert result["refresh_token"] == "nous-refresh-secret"

    def test_suffix_matched_secrets_stripped(self):
        """Keys matching _SECRET_VALUE_SUFFIXES are stripped from borrowed."""
        payload = {
            "source": "borrowed",
            "my_api_key": "sk-12345",
            "custom_secret": "shh",
            "api_key": "sk-67890",
            "name": "my-key",
        }
        result = sanitize_borrowed_credential_payload(payload)
        assert "my_api_key" not in result
        assert "custom_secret" not in result
        assert "api_key" not in result
        assert result["name"] == "my-key"

    def test_camel_case_secret_keys_stripped(self):
        """CamelCase secret keys normalise and get stripped."""
        payload = {
            "source": "borrowed",
            "myApiKey": "sk-camel-secret",
            "safeField": "keep-me",
        }
        result = sanitize_borrowed_credential_payload(payload)
        assert "myApiKey" not in result
        assert result["safeField"] == "keep-me"

    def test_original_payload_not_mutated(self):
        """The function should not mutate the input dict."""
        payload = {
            "source": "env_var",
            "access_token": "original-secret",
        }
        original = dict(payload)
        sanitize_borrowed_credential_payload(payload)
        assert payload == original

    def test_fingerprint_uses_agent_key_priority(self):
        """agent_key should be preferred for fingerprinting over other secrets."""
        payload = {
            "source": "borrowed",
            "agent_key": "ag-key-123",
            "access_token": "at-different",
        }
        result = sanitize_borrowed_credential_payload(payload)
        expected_fp = f"sha256:{hashlib.sha256(b'ag-key-123').hexdigest()[:16]}"
        assert result["secret_fingerprint"] == expected_fp
