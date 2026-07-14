"""Tests for the Hermes operator authorization vocabulary."""

import pytest

from gateway.api_operator_auth import AuthPrincipal, normalize_scopes


def test_normalize_scopes_accepts_domain_read_write_and_deduplicates():
    assert normalize_scopes(["profiles:write", "profiles:read", "profiles:read"]) == (
        "profiles:read",
        "profiles:write",
    )


def test_normalize_scopes_rejects_unknown_domain():
    with pytest.raises(ValueError, match="unknown scope"):
        normalize_scopes(["filesystem:write"])


def test_normalize_scopes_rejects_unknown_scope_even_with_wildcard():
    with pytest.raises(ValueError, match="unknown scope"):
        normalize_scopes(["*", "filesystem:read"])


def test_normalize_scopes_wildcard_with_valid_scopes_collapses_to_wildcard():
    assert normalize_scopes(["*", "profiles:read"]) == ("*",)


def test_superuser_satisfies_every_scope():
    assert AuthPrincipal("legacy-api-key", ("*",), True).allows("profiles:write")
