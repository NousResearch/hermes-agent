"""Tests for the credential proxy server (proxy/server.py)."""

from __future__ import annotations

import pytest

from proxy.server import _substitute_placeholders, _PLACEHOLDER_RE
from proxy.store import CredentialStore


class TestPlaceholderRegex:
    """Regex must match hermes-proxy:// URIs correctly."""

    def test_simple_name(self):
        m = _PLACEHOLDER_RE.search("hermes-proxy://my_token")
        assert m and m.group(1) == "my_token"

    def test_hyphenated_name(self):
        m = _PLACEHOLDER_RE.search("hermes-proxy://cf-dns-token")
        assert m and m.group(1) == "cf-dns-token"

    def test_no_match_on_partial(self):
        m = _PLACEHOLDER_RE.search("hermes-proxy://")
        assert m is None

    def test_no_match_on_wrong_scheme(self):
        m = _PLACEHOLDER_RE.search("http://my_token")
        assert m is None

    def test_multiple_in_string(self):
        matches = _PLACEHOLDER_RE.findall(
            "Bearer hermes-proxy://tok1 and hermes-proxy://tok2"
        )
        assert matches == ["tok1", "tok2"]


class TestSubstitutePlaceholders:
    """_substitute_placeholders rewrites matched names from the store."""

    def test_single_header_value(self):
        store = CredentialStore()
        store.store("api_key", "real_secret_123")
        result = _substitute_placeholders(
            "Bearer hermes-proxy://api_key", store
        )
        assert result == "Bearer real_secret_123"

    def test_multiple_placeholders(self):
        store = CredentialStore()
        store.store("tok1", "val1")
        store.store("tok2", "val2")
        result = _substitute_placeholders(
            "hermes-proxy://tok1 and hermes-proxy://tok2", store
        )
        assert result == "val1 and val2"

    def test_unresolved_placeholder_unchanged(self):
        store = CredentialStore()
        result = _substitute_placeholders(
            "hermes-proxy://unknown_cred", store
        )
        assert result == "hermes-proxy://unknown_cred"

    def test_no_placeholders(self):
        store = CredentialStore()
        text = "Authorization: Bearer sk-real-token"
        assert _substitute_placeholders(text, store) == text

    def test_json_body_substitution(self):
        store = CredentialStore()
        store.store("slack_token", "xoxb-real-token")
        body = '{"token": "hermes-proxy://slack_token", "channel": "#general"}'
        result = _substitute_placeholders(body, store)
        assert result == '{"token": "xoxb-real-token", "channel": "#general"}'

    def test_mixed_resolved_unresolved(self):
        store = CredentialStore()
        store.store("known", "resolved_value")
        result = _substitute_placeholders(
            "hermes-proxy://known hermes-proxy://unknown", store
        )
        assert result == "resolved_value hermes-proxy://unknown"
