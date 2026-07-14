"""Tests for the Hermes operator authorization vocabulary."""

import stat
import sys

import pytest

from gateway.api_operator_auth import (
    AuthPrincipal,
    OperatorCredentialStore,
    normalize_scopes,
)


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


def test_issue_authenticate_list_revoke_round_trip(tmp_path):
    store = OperatorCredentialStore(tmp_path / "operator_credentials.json")
    issued = store.issue("Galaxy S24", ["chat:write", "profiles:read"])

    assert issued.token.startswith("hop_")
    assert issued.token not in (tmp_path / "operator_credentials.json").read_text()
    assert store.authenticate(issued.token).credential_id == issued.credential_id
    assert store.list_credentials()[0].label == "Galaxy S24"
    assert store.revoke(issued.credential_id)
    assert store.authenticate(issued.token) is None


def test_authenticate_uses_full_hash_not_token_prefix(tmp_path):
    store = OperatorCredentialStore(tmp_path / "operator_credentials.json")
    issued = store.issue("phone", ["chat:read"])
    assert store.authenticate(issued.token[:-1] + "x") is None


def test_authenticate_rejects_unknown_token(tmp_path):
    store = OperatorCredentialStore(tmp_path / "operator_credentials.json")
    store.issue("phone", ["chat:read"])
    assert store.authenticate("hop_not-a-real-token") is None


def test_malformed_store_file_fails_closed(tmp_path):
    path = tmp_path / "operator_credentials.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = OperatorCredentialStore(path)

    assert store.list_credentials() == []
    assert store.authenticate("hop_anything") is None
    assert store.revoke("hoc_missing") is False


def test_duplicate_revocation_returns_false(tmp_path):
    store = OperatorCredentialStore(tmp_path / "operator_credentials.json")
    issued = store.issue("phone", ["chat:read"])

    assert store.revoke(issued.credential_id) is True
    assert store.revoke(issued.credential_id) is False


def test_revoke_unknown_credential_returns_false(tmp_path):
    store = OperatorCredentialStore(tmp_path / "operator_credentials.json")
    assert store.revoke("hoc_does-not-exist") is False


def test_label_is_trimmed_and_bounded_to_eighty_characters(tmp_path):
    store = OperatorCredentialStore(tmp_path / "operator_credentials.json")
    issued = store.issue("  padded label  ", ["chat:read"])
    assert issued.label == "padded label"

    long_label = "x" * 200
    issued_long = store.issue(long_label, ["chat:read"])
    assert issued_long.label == "x" * 80
    assert store.list_credentials()[-1].label == "x" * 80


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")
def test_store_file_has_owner_only_permissions(tmp_path):
    store = OperatorCredentialStore(tmp_path / "operator_credentials.json")
    store.issue("phone", ["chat:read"])

    mode = stat.S_IMODE((tmp_path / "operator_credentials.json").stat().st_mode)
    assert mode == 0o600
